import json
import logging
import os
import functools
import inspect
from pathlib import Path
import shutil
from typing import Any, List, Dict, Tuple, Iterable, Type, Callable, Optional, TYPE_CHECKING
import transformers
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm.autonotebook import trange
from datasets import Dataset, DatasetDict
from transformers import TrainerCallback, TrainerState, TrainerControl
from sentence_transformers.training_args import TrainingArguments

from .evaluation import SentenceEvaluator
from .util import (
    batch_to_device,
    fullname,
)
from .model_card_templates import ModelCardTemplate

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer
    from sentence_transformers.readers.InputExample import InputExample


class SaveModelCallback(TrainerCallback):
    """A Callback to save the model to the `output_dir`.

    There are two cases:
    1. save_best_model is True and evaluator is defined:
        We save on evaluate, but only if the new model is better than the currently saved one
        according to the evaluator.
    2. If evaluator is not defined:
        We save after the model has been trained.
    """

    def __init__(self, output_dir: str, evaluator: Optional[SentenceEvaluator], save_best_model: bool) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.evaluator = evaluator
        # TODO: ^ has to implement `greater_is_better` and `primary_metric`
        self.save_best_model = save_best_model
        self.best_metric = None

    def is_better(self, new_metric: float) -> bool:
        if getattr(self.evaluator, "greater_is_better", True):
            return new_metric > self.best_metric
        return new_metric < self.best_metric

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, Any],
        model: "SentenceTransformer",
        **kwargs,
    ):
        if self.evaluator is not None and self.save_best_model:
            metric_key = getattr(self.evaluator, "primary_metric", "evaluator")
            for key, value in metrics.items():
                if key.endswith(metric_key):
                    if self.best_metric is None or self.is_better(value):
                        self.best_metric = value
                        model.save(self.output_dir)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: "SentenceTransformer",
        **kwargs,
    ):
        if self.evaluator is None:
            model.save(self.output_dir)


class EvaluatorCallback(TrainerCallback):
    """The SentenceTransformers.fit method always ran the evaluator on every epoch,
    in addition to every "evaluation_steps". This callback is responsible for that.

    The `.trainer` must be provided after the trainer has been created.
    """

    def __init__(self, evaluator: SentenceEvaluator) -> None:
        super().__init__()
        self.evaluator = evaluator
        self.metric_key_prefix = "eval"
        self.trainer = None

    def on_epoch_end(
        self,
        args: transformers.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: "SentenceTransformer",
        **kwargs,
    ):
        evaluator_metrics = self.evaluator(model, epoch=state.epoch)
        if not isinstance(evaluator_metrics, dict):
            evaluator_metrics = {"evaluator": evaluator_metrics}

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(evaluator_metrics.keys()):
            if not key.startswith(f"{self.metric_key_prefix}_"):
                evaluator_metrics[f"{self.metric_key_prefix}_{key}"] = evaluator_metrics.pop(key)

        if self.trainer is not None:
            self.trainer.callback_handler.on_evaluate(args, state, control, metrics=evaluator_metrics)


class OriginalCallback(TrainerCallback):
    """A Callback to invoke the original callback function that was provided to SentenceTransformer.fit()

    This callback has the following signature: `(score: float, epoch: int, steps: int) -> None`
    """

    def __init__(self, callback: Callable[[float, int, int], None], evaluator: SentenceEvaluator) -> None:
        super().__init__()
        self.callback = callback
        self.evaluator = evaluator

    def on_evaluate(
        self,
        args: transformers.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, Any],
        **kwargs,
    ):
        metric_key = getattr(self.evaluator, "primary_metric", "evaluator")
        for key, value in metrics.items():
            if key.endswith(metric_key):
                return self.callback(value, state.epoch, state.global_step)


class FitMixin:
    """Mixin class for injecting the `fit` method into Sentence Transformers"""

    def fit(
        self,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch=None,
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
    ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """
        # Delayed import to counter the SentenceTransformers -> FitMixin -> SentenceTransformerTrainer -> SentenceTransformers circular import
        from sentence_transformers.trainer import SentenceTransformerTrainer

        data_loaders, loss_fns = zip(*train_objectives)

        # Clear the dataloaders from collate functions as we just want raw InputExamples
        def identity(batch):
            return batch

        for data_loader in data_loaders:
            data_loader.collate_fn = identity

        batch_size = None
        # Convert dataloaders into a DatasetDict
        train_dataset_dict = {}
        for loader_idx, data_loader in enumerate(data_loaders, start=1):
            batch_size = getattr(data_loader, "batch_size", batch_size)
            texts = []
            labels = []
            for batch in data_loader:
                batch_texts, batch_labels = zip(*[(example.texts, example.label) for example in batch])
                texts += batch_texts
                labels += batch_labels
            dataset = Dataset.from_dict({f"sentence_{idx}": text for idx, text in enumerate(zip(*texts))})
            if set(labels) != {0}:
                dataset = dataset.add_column("label", labels)
            train_dataset_dict[f"dataset_{loader_idx}"] = dataset
        train_dataset_dict = DatasetDict(train_dataset_dict)

        def _default_checkpoint_dir() -> str:
            dir_name = "checkpoints/model"
            idx = 1
            while Path(dir_name).exists() and len(list(Path(dir_name).iterdir())) != 0:
                dir_name = f"checkpoints/model_{idx}"
                idx += 1
            return dir_name

        # Convert loss_fns into a dict with `dataset_{idx}` keys
        loss_fn_dict = {f"dataset_{idx}": loss_fn for idx, loss_fn in enumerate(loss_fns, start=1)}
        # TODO: Test model checkpointing & loading

        # Use steps_per_epoch to perhaps set max_steps
        max_steps = -1
        if steps_per_epoch is not None and steps_per_epoch > 0:
            if epochs == 1:
                max_steps = steps_per_epoch
            else:
                logger.warning(
                    "Setting `steps_per_epoch` alongside `epochs` > 1 no longer works. "
                    "We will train with the full datasets per epoch."
                )
                steps_per_epoch = None

        args = TrainingArguments(
            output_dir=checkpoint_path or _default_checkpoint_dir(),
            round_robin_sampler=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            max_steps=max_steps,
            evaluation_strategy="steps" if evaluation_steps is not None and evaluation_steps > 0 else "no",
            eval_steps=evaluation_steps,
            # load_best_model_at_end=save_best_model, # <- TODO: Look into a good solution for save_best_model
            max_grad_norm=max_grad_norm,
            fp16=use_amp,
            disable_tqdm=not show_progress_bar,
            save_strategy="steps" if checkpoint_path is not None else "no",
            save_steps=checkpoint_save_steps,
            save_total_limit=checkpoint_save_total_limit,
        )

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(train_dataset) // batch_size for train_dataset in train_dataset_dict.values()])
        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizer & scheduler
        param_optimizer = list(self.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler_obj = self._get_scheduler(
            optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
        )

        # Create callbacks
        callbacks = []
        if output_path is not None:
            callbacks.append(SaveModelCallback(output_path, evaluator, save_best_model))
        if evaluator is not None:
            callbacks.append(EvaluatorCallback(evaluator))
            if callback is not None:
                callbacks.append(OriginalCallback(callback, evaluator))

        trainer = SentenceTransformerTrainer(
            model=self,
            args=args,
            train_dataset=train_dataset_dict,
            eval_dataset=None,
            loss=loss_fn_dict,
            evaluator=evaluator,
            optimizers=(optimizer, scheduler_obj),
            callbacks=callbacks,
        )
        # Set the trainer on the EvaluatorCallback, required for logging the metrics
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, EvaluatorCallback):
                callback.trainer = trainer

        trainer.train()

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == "constantlr":
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == "warmupconstant":
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == "warmuplinear":
            return transformers.get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosine":
            return transformers.get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        elif scheduler == "warmupcosinewithhardrestarts":
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
            )
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def smart_batching_collate(self, batch: List["InputExample"]) -> Tuple[List[Dict[str, Tensor]], Tensor]:
        """
        Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
        Here, batch is a list of InputExample instances: [InputExample(...), ...]

        :param batch:
            a batch from a SmartBatchingDataset
        :return:
            a batch of tensors for the model
        """
        texts = [example.texts for example in batch]
        sentence_features = [self.tokenize(sentence) for sentence in zip(*texts)]
        labels = torch.tensor([example.label for example in batch])
        return sentence_features, labels

    """
    Temporary methods that will be removed when this refactor is complete:
    """

    def old_fit(
        self,
        train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        steps_per_epoch=None,
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
        checkpoint_path: str = None,
        checkpoint_save_steps: int = 500,
        checkpoint_save_total_limit: int = 0,
    ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_objectives: Tuples of (DataLoader, LossFunction). Pass more than one for multi-task learning
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param steps_per_epoch: Number of training steps per epoch. If set to None (default), one epoch is equal the DataLoader size from train_objectives.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        :param checkpoint_path: Folder to save checkpoints during training
        :param checkpoint_save_steps: Will save a checkpoint after so many steps
        :param checkpoint_save_total_limit: Total number of checkpoints to store
        """

        ##Add info to model card
        # info_loss_functions = "\n".join(["- {} with {} training examples".format(str(loss), len(dataloader)) for dataloader, loss in train_objectives])
        info_loss_functions = []
        for dataloader, loss in train_objectives:
            info_loss_functions.extend(ModelCardTemplate.get_train_objective_info(dataloader, loss))
        info_loss_functions = "\n\n".join([text for text in info_loss_functions])

        info_fit_parameters = json.dumps(
            {
                "evaluator": fullname(evaluator),
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "scheduler": scheduler,
                "warmup_steps": warmup_steps,
                "optimizer_class": str(optimizer_class),
                "optimizer_params": optimizer_params,
                "weight_decay": weight_decay,
                "evaluation_steps": evaluation_steps,
                "max_grad_norm": max_grad_norm,
            },
            indent=4,
            sort_keys=True,
        )
        self._model_card_text = None
        self._model_card_vars["{TRAINING_SECTION}"] = ModelCardTemplate.__TRAINING_SECTION__.replace(
            "{LOSS_FUNCTIONS}", info_loss_functions
        ).replace("{FIT_PARAMETERS}", info_fit_parameters)

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        self.to(self.device)

        dataloaders = [dataloader for dataloader, _ in train_objectives]

        # Use smart batching
        for dataloader in dataloaders:
            dataloader.collate_fn = self.smart_batching_collate

        loss_models = [loss for _, loss in train_objectives]
        for loss_model in loss_models:
            loss_model.to(self.device)

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        # Prepare optimizers
        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]

            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = self._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]

        num_train_objectives = len(train_objectives)

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05, disable=not show_progress_bar):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self.device)
                    features = list(map(lambda batch: batch_to_device(batch, self.device), features))

                    if use_amp:
                        with autocast():
                            loss_value = loss_model(features, labels)

                        scale_before_step = scaler.get_scale()
                        scaler.scale(loss_value).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()

                        skip_scheduler = scaler.get_scale() != scale_before_step
                    else:
                        loss_value = loss_model(features, labels)
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                        optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(
                        evaluator, output_path, save_best_model, epoch, training_steps, callback
                    )

                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if (
                    checkpoint_path is not None
                    and checkpoint_save_steps is not None
                    and checkpoint_save_steps > 0
                    and global_step % checkpoint_save_steps == 0
                ):
                    self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

            self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

        if evaluator is None and output_path is not None:  # No evaluator, but output path: save final model version
            self.save(output_path)

        if checkpoint_path is not None:
            self._save_checkpoint(checkpoint_path, checkpoint_save_total_limit, global_step)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        eval_path = output_path
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            eval_path = os.path.join(output_path, "eval")
            os.makedirs(eval_path, exist_ok=True)

        if evaluator is not None:
            score = evaluator(self, output_path=eval_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def _save_checkpoint(self, checkpoint_path, checkpoint_save_total_limit, step):
        # Store new checkpoint
        self.save(os.path.join(checkpoint_path, str(step)))

        # Delete old checkpoints
        if checkpoint_save_total_limit is not None and checkpoint_save_total_limit > 0:
            old_checkpoints = []
            for subdir in os.listdir(checkpoint_path):
                if subdir.isdigit():
                    old_checkpoints.append({"step": int(subdir), "path": os.path.join(checkpoint_path, subdir)})

            if len(old_checkpoints) > checkpoint_save_total_limit:
                old_checkpoints = sorted(old_checkpoints, key=lambda x: x["step"])
                shutil.rmtree(old_checkpoints[0]["path"])


class GradientCheckpointingMixin:
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `torch.utils.checkpoint.checkpoint` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": True}

        gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        else:
            self.apply(functools.partial(self._set_gradient_checkpointing, value=True))
            logger.warn(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PreTrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True
            logger.debug(
                f"Gradient checkpointing enabled for {self.__class.__name__}."
            )

        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            # For old GC format (transformers < 4.35.0) for models that live on the Hub
            # we will fall back to the overwritten `_set_gradient_checkpointing` methid
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                logger.warn(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                self.apply(functools.partial(self._set_gradient_checkpointing, value=False))

        if getattr(self, "_hf_peft_config_loaded", False):
            self.disable_input_require_grads()

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing for m in self.modules())
