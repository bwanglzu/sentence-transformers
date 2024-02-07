import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import PreTrainedTokenizerBase, Trainer, EvalPrediction, TrainerCallback
from transformers.trainer import TRAINING_ARGS_NAME

from datasets import DatasetDict, Dataset
from transformers.trainer_utils import EvalLoopOutput
from transformers.data.data_collator import DataCollator
from sentence_transformers.losses import CoSENTLoss

from sentence_transformers.training_args import TrainingArguments
from sentence_transformers.data_collator import SentenceTransformerDataCollator
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.sampler import ProportionalBatchSampler, RoundRobinBatchSampler

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer


class SentenceTransformerTrainer(Trainer):
    def __init__(
        self,
        model: Optional["SentenceTransformer"] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        loss: nn.Module = None,
        evaluator: SentenceEvaluator = None,
        data_collator: Optional[DataCollator] = None,
        tokenizer: Optional[Union[PreTrainedTokenizerBase, Callable]] = None,
        model_init: Optional[Callable[[], "SentenceTransformer"]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        elif not isinstance(args, TrainingArguments):
            raise ValueError("Please use `TrainingArguments` imported from `sentence_transformers`.")
        if tokenizer is None and isinstance(model.tokenizer, PreTrainedTokenizerBase):
            tokenizer = model.tokenizer
        if data_collator is None:
            data_collator = SentenceTransformerDataCollator(tokenize_fn=model.tokenize)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        if loss is None:
            logger.info("No `loss` passed, using `losses.CoSENTLoss` as a default option.")
            loss = CoSENTLoss(self.model)
        self.loss = loss
        if isinstance(loss, dict):
            self.loss = {dataset_name: loss_fn.to(self.model.device) for dataset_name, loss_fn in loss.items()}
            for dataset_name, dataset in zip(["train", "eval"], [train_dataset, eval_dataset]):
                if dataset is None:
                    continue
                if not isinstance(dataset, dict):
                    raise ValueError(
                        f"If the provided `loss` is a dict, then the `{dataset_name}_dataset` must be a `DatasetDict`."
                    )
                if missing := set(dataset.keys()) - set(loss.keys()):
                    raise ValueError(
                        f"If the provided `loss` is a dict, then all keys from the `{dataset_name}_dataset` dictionary must occur in `loss` also. "
                        f"Currently, {sorted(missing)} occur{'s' if len(missing) == 1 else ''} in `{dataset_name}_dataset` but not in `loss`."
                    )
        else:
            self.loss.to(self.model.device)
        self.evaluator = evaluator
        self.training_with_dataset_dict = isinstance(self.train_dataset, DatasetDict)
        if self.training_with_dataset_dict:
            self.dataset_names = list(self.train_dataset.keys())

    def compute_loss(
        self,
        model: "SentenceTransformer",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = self.collect_features(inputs)
        loss_fn = self.loss

        if self.training_with_dataset_dict and isinstance(loss_fn, dict):
            loss_fn = loss_fn[self.dataset_name]

        loss = loss_fn(features, labels)
        if return_outputs:
            output = torch.cat([model(row)["sentence_embedding"][:, None] for row in features], dim=1)
            return loss, output
        return loss

    def collect_features(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Tuple[List[Dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        """Turn the inputs from the dataloader into the separate model inputs & the labels.

        Example::

            >>> list(inputs.keys())
            ['return_loss', 'label', 'sentence_0_input_ids', 'sentence_0_token_type_ids', 'sentence_0_attention_mask', 'sentence_1_input_ids', 'sentence_1_token_type_ids', 'sentence_1_attention_mask']
            >>> features, labels = self.collect_features(inputs)
            >>> len(features)
            2
            >>> list(features[0].keys())
            ['input_ids', 'token_type_ids', 'attention_mask']
            >>> list(features[1].keys())
            ['input_ids', 'token_type_ids', 'attention_mask']
            >>> torch.equal(labels, inputs["label"])
            True
        """
        # All inputs ending with `_input_ids` or `_sentence_embedding` (for BoW only) are considered to correspond to a feature
        features = []
        for column in inputs:
            if column.endswith("_input_ids"):
                prefix = column[: -len("input_ids")]
            elif column.endswith("_sentence_embedding"):
                prefix = column[: -len("sentence_embedding")]
            else:
                continue
            features.append({key[len(prefix) :]: value for key, value in inputs.items() if key.startswith(prefix)})
        labels = inputs.get("label", None)
        return features, labels

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        if (
            self.training_with_dataset_dict
            and self.is_in_train
            and isinstance(self.eval_dataset, dict)
            and metric_key_prefix.startswith("eval_")
            and metric_key_prefix[5:] in list(self.eval_dataset.keys())
        ):
            self.dataset_name = metric_key_prefix[5:]

        output = super().evaluation_loop(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # If the evaluator is not defined, we can just return the output
        if self.evaluator is None:
            return output

        # If we are training and eval_dataset is a DatasetDict, then we should
        # 1) only run the evaluator for the first dataset
        # 2) prefix that only run as "eval", rather than e.g. "eval_multi_nli"
        if self.is_in_train and isinstance(self.eval_dataset, dict) and metric_key_prefix.startswith("eval_"):
            if metric_key_prefix[5:] == list(self.eval_dataset.keys())[0]:
                metric_key_prefix = "eval"
            else:
                return output

        evaluator_metrics = self.evaluator(self.model)
        if not isinstance(evaluator_metrics, dict):
            evaluator_metrics = {"evaluator": evaluator_metrics}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(evaluator_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                evaluator_metrics[f"{metric_key_prefix}_{key}"] = evaluator_metrics.pop(key)

        output.metrics.update(evaluator_metrics)

        return output

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, DatasetDict):
            sizes = [len(ds) for ds in train_dataset.values()]
            train_dataset = ConcatDataset(train_dataset.values())
        else:
            sizes = [len(train_dataset)]

        batch_sampler_class = RoundRobinBatchSampler if self.args.round_robin_sampler else ProportionalBatchSampler
        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "batch_sampler": batch_sampler_class(
                lengths=sizes,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                seed=self.args.seed,
                trainer=self,
            ),
        }

        self._train_dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        return self._train_dataloader

    def get_eval_dataloader(self, eval_dataset: Union[Dataset, None] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            # Prevent errors if the evaluator is set but no eval_dataset is provided
            if self.evaluator is not None:
                return DataLoader([])
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if isinstance(eval_dataset, DatasetDict):
            sizes = [len(ds) for ds in eval_dataset.values()]
            eval_dataset = ConcatDataset(eval_dataset.values())
        else:
            sizes = [len(eval_dataset)]

        batch_sampler_class = RoundRobinBatchSampler if self.args.round_robin_sampler else ProportionalBatchSampler
        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "batch_sampler": batch_sampler_class(
                lengths=sizes,
                batch_size=self.args.train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                seed=self.args.seed,
            ),
        }

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    # TODO: Also override the test_dataloader

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir, safe_serialization=self.args.save_safetensors)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
