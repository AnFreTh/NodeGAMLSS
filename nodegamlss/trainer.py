"""The trainer to optimize the model."""

import glob
import os
import time
from collections import OrderedDict
from copy import deepcopy
from os.path import join as pjoin, exists as pexists
import re
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

try:
    IS_AMP_EXISTS = True
    from apex import amp
except ModuleNotFoundError:
    print("WARNING! The apex is not installed so fp16 is not available.")
    IS_AMP_EXISTS = False

from .utils import get_latest_file, check_numpy, process_in_chunks


class Trainer(nn.Module):
    def __init__(
        self,
        models,
        family,
        experiment_name=None,
        problem="LSS",
        warm_start=False,
        Optimizer=torch.optim.Adam,
        optimizer_params={},
        lr=0.01,
        lr_warmup_steps=-1,
        verbose=False,
        n_last_checkpoints=5,
        step_callbacks=[],
        fp16=0,
        pretraining_ratio=0.15,
        masks_noise=0.1,
        opt_only_last_layer=False,
        freeze_steps=0,
        **kwargs,
    ):
        """Trainer.

        Args:
                model (torch.nn.Module): the model.
                experiment_name: a path where all logs and checkpoints are saved.
                warm_start: when set to True, loads the last checkpoint.
                Optimizer: function(parameters) -> optimizer. Default: torch.optim.Adam.
                optimizer_params: parameter when intializing optimizer. Usage:
                        Optimizer(**optimizer_params).
                verbose: when set to True, produces logging information.
                n_last_checkpoints: the last few checkpoints to do model averaging.
                step_callbacks: function(step). Will be called after each optimization step.
                problem: problem type. Chosen from ['classification', 'regression', 'pretrain'].
                pretraining_ratio: the percentage of feature to mask for reconstruction. Between 0 and
                        1. Only used when problem == 'pretrain'.
        """
        super().__init__()
        self.models = models
        self.verbose = verbose
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps

        # When using fp16, there are some params if not filtered out by requires_grad
        # will produce error
        params = []
        if opt_only_last_layer:
            print("Only optimize last layer!")
            for model in self.models:
                # Assuming each model has an attribute pointing to its last layer's parameters
                params.extend([getattr(model, "last_layer_attr")])
        else:
            for model in self.models:
                params.extend([p for p in model.parameters() if p.requires_grad])

        self.opt = Optimizer(params, lr=self.lr, **optimizer_params)

        # Initialize the optimizer with the collected parameters
        self.opt = Optimizer(params, lr=lr, **optimizer_params)
        self.step = 0
        self.n_last_checkpoints = n_last_checkpoints
        self.step_callbacks = step_callbacks
        self.fp16 = fp16
        self.pretraining_ratio = pretraining_ratio
        self.masks_noise = masks_noise
        self.opt_only_last_layer = opt_only_last_layer
        self.freeze_steps = freeze_steps
        self.family = family

        if experiment_name is None:
            experiment_name = "untitled_{:%Y.%m.%d_%H_%M}".format(
                datetime.now()
            ).replace(":", "_")
            if self.verbose:
                print("using automatic experiment name: " + experiment_name)
        else:
            experiment_name = experiment_name.replace(":", "_")

        self.experiment_path = os.path.join("logs", experiment_name)

        if fp16 and IS_AMP_EXISTS:
            self.models, self.opt = [
                amp.initialize(model, self.opt, opt_level="O1") for model in self.models
            ]
        if warm_start:
            self.load_checkpoint()

        if problem.startswith("pretrain"):
            self.freeze_steps = 0
            self.loss_function = None

        elif problem == "regression":
            self.loss_function = lambda y1, y2: F.mse_loss(y1.float(), y2.float())

        elif problem == "LSS":
            self.loss_function = lambda predictions, y_true: self.family.compute_loss(
                predictions, y_true
            )

        self.problem = problem

    def save_checkpoint(self, tag=None, path=None, mkdir=True, **kwargs):
        assert (
            tag is None or path is None
        ), "please provide either tag or path or nothing, not both"

        if tag is not None:
            # Replace invalid characters with an underscore
            tag = re.sub(r'[<>:"/\\|?*]', "_", tag)

        if tag is None and path is None:
            # Format the timestamp without colons
            timestamp = datetime.now().strftime("temp_%Y.%m.%d_%H-%M")
            tag = f"{timestamp}_{self.step}"

        if path is None:
            path = os.path.join(self.experiment_path, f"checkpoint_{tag}.pth")

        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Sometimes happen there is a checkpoint already existing. Then overwrite!
        if pexists(path):
            os.remove(path)

        model_states = [model.state_dict() for model in self.models]
        checkpoint = {
            "models": model_states,
            "opt": self.opt.state_dict(),
            "step": self.step,
        }
        if self.fp16 and IS_AMP_EXISTS:
            checkpoint["amp"] = amp.state_dict()

        torch.save(checkpoint, path)
        if self.verbose:
            print("Saved " + path)
        return path

    def load_checkpoint(self, tag=None, path=None, **kwargs):
        assert (
            tag is None or path is None
        ), "please provide either tag or path or nothing, not both"
        if tag is None and path is None:
            path = self.get_latest_file(
                pjoin(self.experiment_path, "checkpoint_temp_[0-9]*.pth")
            )
            if path is None:
                return self

        if tag is not None and path is None:
            path = os.path.join(self.experiment_path, f"checkpoint_{tag}.pth")

        checkpoint = torch.load(path)
        for model, state in zip(self.models, checkpoint["models"]):
            model.load_state_dict(state, **kwargs)

        self.opt.load_state_dict(checkpoint["opt"])
        self.step = checkpoint["step"]

        if self.fp16 and IS_AMP_EXISTS and "amp" in checkpoint:
            amp.load_state_dict(checkpoint["amp"])

        # Set the temperature
        for c in self.step_callbacks:
            c(self.step)

        if self.verbose:
            print("Loaded " + path)
        return self

    def get_latest_file(self, pattern):
        list_of_files = glob.glob(pattern)

        if not list_of_files:  # If no files found
            if self.verbose:
                print("No previous checkpoints found. Train from scratch.")
            return None

        # Find the latest file based on creation time
        latest_file = max(list_of_files, key=os.path.getctime)

        # Check if the latest file is valid
        if (
            os.stat(latest_file).st_size == 0
            or len(list_of_files) > self.n_last_checkpoints
        ):
            os.remove(latest_file)  # Remove invalid file
            return self.get_latest_file(
                pattern
            )  # Recursively find the next latest file

        return latest_file

    def average_checkpoints(self, tags=None, paths=None, out_tag="avg", out_path=None):
        assert (
            tags is None or paths is None
        ), "Please provide either tags or paths, not both."
        assert (
            out_tag is not None or out_path is not None
        ), "Please provide either out_tag or out_path."

        # Ensure paths are correctly generated or specified
        if paths is None:
            if tags:
                paths = [
                    os.path.join(self.experiment_path, f"checkpoint_{tag}.pth")
                    for tag in tags
                ]
            else:
                paths = self.get_latest_checkpoints(
                    pjoin(self.experiment_path, "checkpoint_temp_*").replace(":", "_"),
                    self.n_last_checkpoints,
                )

        # Load checkpoints
        checkpoints = [torch.load(path) for path in paths]

        # Initialize a structure for the averaged model states
        averaged_ckpts = [
            deepcopy(checkpoints[0]["models"][i]) for i in range(len(self.models))
        ]

        # Iterate through each model's parameters and average them across checkpoints
        for model_idx, _ in enumerate(self.models):
            for param_key in averaged_ckpts[model_idx]:
                # Aggregate the same parameter across all checkpoints for the current model
                param_values = [
                    ckpt["models"][model_idx][param_key] for ckpt in checkpoints
                ]

                # Average the parameter values and update the averaged checkpoint
                averaged_ckpts[model_idx][param_key] = torch.mean(
                    torch.stack(param_values, dim=0), dim=0
                )

        # Prepare the final averaged checkpoint dictionary
        averaged_checkpoint = {
            "models": averaged_ckpts,
            "opt": checkpoints[0][
                "opt"
            ],  # Optionally, you might want to handle averaging or resetting optimizer states
            "step": checkpoints[0][
                "step"
            ],  # Assuming step should be consistent across checkpoints; adjust as needed
        }

        # Handle output path
        if out_path is None:
            out_path = os.path.join(
                self.experiment_path, f"checkpoint_{out_tag}.pth"
            ).replace(":", "_")

        # Save the averaged checkpoint
        torch.save(averaged_checkpoint, out_path)
        if self.verbose:
            print(f"Averaged checkpoint saved to {out_path}")

    def get_latest_checkpoints(self, pattern, n_last=None):
        list_of_files = glob.glob(pattern)
        if len(list_of_files) == 0:
            return []

        assert len(list_of_files) > 0, "No latest checkpoint found: " + pattern
        return sorted(list_of_files, key=os.path.getctime, reverse=True)[:n_last]

    def remove_old_temp_checkpoints(self, number_ckpts_to_keep=None):
        if number_ckpts_to_keep is None:
            number_ckpts_to_keep = self.n_last_checkpoints
        paths = self.get_latest_checkpoints(
            pjoin(self.experiment_path, "checkpoint_temp_[0-9]*.pth")
        )
        paths_to_delete = paths[number_ckpts_to_keep:]

        for ckpt in paths_to_delete:
            os.remove(ckpt)

    def train_on_batch(self, *batch, device, update=True):
        # Tune temperature in choice function
        for c in self.step_callbacks:
            c(self.step)

        # Tune the learning rate
        if self.lr_warmup_steps > 0 and self.step < self.lr_warmup_steps:
            cur_lr = self.lr * (self.step + 1) / self.lr_warmup_steps
            self.set_lr(cur_lr)

        if self.freeze_steps > 0 and self.step == 0 and update:
            for model in self.models:
                model.freeze_all_but_lastw()

        if 0 < self.freeze_steps == self.step:
            for model in self.models:
                model.unfreeze()

        x_batch, y_batch = batch
        x_batch = torch.as_tensor(x_batch, device=device)
        if not self.problem.startswith("pretrain"):  # Save some memory
            y_batch = torch.as_tensor(y_batch, device=device)

        for model in self.models:
            model.train()

        # Read that it's faster...
        for group in self.opt.param_groups:
            for p in group["params"]:
                p.grad = None
        # self.opt.zero_grad()

        if not self.problem.startswith("pretrain"):  # Normal training
            model_predictions = []
            model_penalties = []

            # Iterate over each model to get predictions and penalties
            for model in self.models:
                prediction, penalty = model(x_batch, return_outputs_penalty=True)
                model_predictions.append(
                    prediction.unsqueeze(-1)
                )  # Add a new dimension to concatenate along
                model_penalties.append(penalty)

            # Concatenate predictions along the new dimension to get a shape of (n, k)
            predictions = torch.cat(model_predictions, dim=-1)

            # Sum the penalties
            total_penalty = sum(model_penalties)

            loss = self.loss_function(predictions, y_batch).mean()

        else:
            raise ValueError(
                "please do not use pre-train objective for NodeGAMLSS as it is not yet implemented."
            )

        loss += total_penalty

        if self.fp16 and IS_AMP_EXISTS:
            with amp.scale_loss(loss, self.opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if update:
            self.opt.step()
            self.step += 1

        return {"loss": loss.item()}

    def mask_input(self, x_batch):
        masks = torch.bernoulli(self.pretraining_ratio * torch.ones(x_batch.shape)).to(
            x_batch.device
        )

        infills = 0.0
        # To make it more difficult, 10% of the time we do not mask the inputs! Similar to BERT
        # tricks.
        new_masks = masks
        if self.masks_noise > 0.0:
            new_masks = torch.bernoulli((1.0 - self.masks_noise) * masks)
        x_batch = (1.0 - new_masks) * x_batch + new_masks * infills
        return x_batch, masks, new_masks

    def pretrain_loss(self, outputs, masks, targets):
        if self.problem.startswith("pretrain_recon"):
            nb_masks = torch.sum(masks, dim=1, keepdim=True)
            nb_masks[nb_masks == 0] = 1
            loss = (((outputs - targets) * masks) ** 2) / nb_masks
            loss = torch.mean(loss)
        else:
            raise NotImplementedError("Unknown problem: " + self.problem)

        return loss

    def evaluate_mse(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        for model in self.models:
            model.train(False)
        with torch.no_grad():
            prediction = process_in_chunks(
                self.models[0], X_test, batch_size=batch_size
            )
            prediction = check_numpy(prediction)
        if prediction.ndim == 1 or (prediction.ndim == 2 and prediction.shape[1] == 1):
            # This handles the case where 'prediction' is 1D or 2D with a single column
            error_rate = (
                (y_test - prediction.squeeze()) ** 2
            ).mean()  # Use 'squeeze()' to ensure it works for both 1D and 2D cases
        else:
            # This handles the case where 'prediction' is 2D with more than one column
            error_rate = ((y_test - prediction[:, 0]) ** 2).mean()
        error_rate = float(error_rate)  # To avoid annoying JSON unserializable bug
        return error_rate

    def evaluate_LSS(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        for model in self.models:
            model.train(False)
        with torch.no_grad():
            model_predictions = []
            for model in self.models:
                model_predictions.append(
                    process_in_chunks(model, X_test, batch_size=batch_size)
                )
            prediction = np.array(model_predictions).T
            prediction = check_numpy(prediction)
            error_rate = self.family.evaluate_nll(prediction, y_test).mean()
        error_rate = float(error_rate)  # To avoid annoying JSON unserializable bug
        return error_rate

    def evaluate_multiple_mse(self, X_test, y_test, device, batch_size=4096):
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        for model in self.models:
            model.train(False)
        with torch.no_grad():
            prediction = [
                process_in_chunks(model, X_test, batch_size=batch_size)
                for model in self.models
            ]
            prediction = check_numpy(prediction)
            error_rate = ((y_test - prediction) ** 2).mean(axis=0)
        return error_rate.astype(float).tolist()

    def evaluate_ce_loss(self, X_test, y_test, device, batch_size=512):
        """Evaluate cross entropy loss for binary or multi-class targets.

        Args:
                X_test: input features.
                y_test (numpy Int array or torch Long tensor): the target classes.

        Returns:
                celoss (float): the average cross entropy loss.
        """
        X_test = torch.as_tensor(X_test, device=device)
        y_test = check_numpy(y_test)
        for model in self.models:
            model.train(False)
        with torch.no_grad():
            logits = [
                process_in_chunks(model, X_test, batch_size=batch_size)
                for model in self.models
            ]
            y_test = torch.tensor(y_test, device=device)

            if logits.ndim == 1:
                celoss = F.binary_cross_entropy_with_logits(
                    logits, y_test.float()
                ).item()
            else:
                celoss = F.cross_entropy(logits, y_test).item()
        celoss = float(celoss)  # To avoid annoying JSON unserializable bug
        return celoss

    def decrease_lr(self, ratio=0.1, min_lr=1e-6):
        if self.lr <= min_lr:
            return

        self.lr *= ratio
        if self.lr < min_lr:
            self.lr = min_lr
        self.set_lr(self.lr)

    def set_lr(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr
