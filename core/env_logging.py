import os
import numpy as np
import pandas as pd
import torch
from core.environment import ALGame

class EnvironmentLogger:

    def __init__(self, environment:ALGame, out_path:str, is_cluster, save_checkpoints=False):
        self.is_cluster = is_cluster
        self.out_path = out_path
        self.env = environment
        self.save_checkpoints = save_checkpoints
        self.accuracies_path = os.path.join(out_path, "accuracies.csv")
        self.losses_path = os.path.join(self.out_path, "losses.csv")
        if save_checkpoints:
            self.ckpt_dir = os.path.join(out_path, "checkpoints")
            os.makedirs(self.ckpt_dir, exist_ok=True)

    def __enter__(self):
        self.current_run = 0
        self.accuracies = dict()
        self.losses = dict()
        return self

    def __exit__(self, type, value, traceback):
        # create base dirs
        os.makedirs(self.out_path, exist_ok=True)
        # Check if this was a test run, or if all runs finished
        if not self.is_cluster and \
           self.env.added_images < self.env.budget:
            resp = input(f"Only {self.env.added_images}/{self.env.budget} iterations computed. Do you want to overwrite existing results? (y/n)")
            if resp != "y":
                print("Keeping old results...")
                return
        # clear old files
        if os.path.exists(self.accuracies_path):
            os.remove(self.accuracies_path)
        if os.path.exists(self.losses_path):
            os.remove(self.losses_path)
        # save new files
        acc_df = pd.DataFrame(self.accuracies)
        acc_df.to_csv(self.accuracies_path)
        loss_df = pd.DataFrame(self.losses)
        loss_df.to_csv(self.losses_path)


    def reset(self, *args, **kwargs)->list:
        return_values = self.env.reset(*args, **kwargs)
        self.current_run += 1
        self.current_timestep = 0
        self.al_round = 0
        self.accuracies[self.current_run] = [self.env.current_test_accuracy]
        self.losses[self.current_run] = [self.env.current_test_loss]
        if self.save_checkpoints:
            self._save_checkpoint()
        return return_values


    def step(self, *args, **kwargs):
        return_values = self.env.step(*args, **kwargs)
        while self.current_timestep < self.env.added_images - 1:
            self.accuracies[self.current_run].append(np.nan)
            self.losses[self.current_run].append(np.nan)
            self.current_timestep += 1
        self.accuracies[self.current_run].append(self.env.current_test_accuracy)
        self.losses[self.current_run].append(self.env.current_test_loss)
        self.current_timestep += 1
        self.al_round += 1
        if self.save_checkpoints:
            self._save_checkpoint()
        return return_values


    def _save_checkpoint(self):
        ckpt = {
            "round": self.al_round,
            "model_state_dict": self.env.classifier.state_dict(),
            "labeled_indices": list(self.env.labeled_indices),
            "test_accuracy": self.env.current_test_accuracy,
            "test_loss": getattr(self.env, "current_test_loss", None),
            "added_images": self.env.added_images,
        }
        path = os.path.join(self.ckpt_dir, f"round_{self.al_round:03d}.pt")
        torch.save(ckpt, path)
