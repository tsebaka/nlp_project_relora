import os
import sys
import math
import torch
import torch.nn as nn
import lightning as L
from torch import optim
from torch.utils.data import DataLoader
from typing import Dict, Any
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from .utils import create_module_lists, get_weight_names


import torch._dynamo
torch._dynamo.config.suppress_errors = True
peft_supported_layers = [""]

class ReloraModule(L.LightningModule):
    def __init__(self,
                 model_class: nn.Module,
                 model_path: str,
                 lora_config: Any = None,
                 lora_merge_freq: int = 0,
                 train_dataset: Any = None,
                 eval_dataset: Any = None,
                 batch_size: int = 1,
                 learning_rate: float = 1e-6,
                 num_workers: int = 4,
                 **kwargs):
        
        super().__init__()
        checkpoint_folder = "relora_checkpoints"
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        self.model_class = model_class
        self.model_path = model_path
        self.save_path = f"./{checkpoint_folder}/{model_path}"

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.workers = num_workers

        self.train_set = train_dataset
        self.eval_set = eval_dataset

        self.num_workers = num_workers
        self.batch_size = batch_size
        
        self.torch_compile = False 
        self.config = lora_config
        self.merge_freq = lora_merge_freq
        self.merge_precision = torch.float32 
        self.load_precision = torch.float16 
        self.train_all_params = True

    def setup(self, stage: str):
        if stage == 'fit':
            self.load_model(self.model_path)
            self.prepare_lora_training()

            if torch.__version__ >= "2" and sys.platform != "win32" and self.torch_compile:
                self.model = torch.compile(self.model)

            checkpoint_folder = self.trainer.default_root_dir
            if not os.path.exists(checkpoint_folder):
                os.makedirs(checkpoint_folder)
            self.save_path = f"./{checkpoint_folder}/{self.model_path}"

    def prepare_lora_training(self):
        if self.merge_freq > 0:
            if self.config.target_modules is None:
                self.config.target_modules, self.base_modules = create_module_lists(self.model, "lora")
            else:
                self.base_modules = [item for item in get_weight_names() if item not in self.config.target_modules]
            self.base_modules += ["lora_"]
            self.init_lora()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.eval_set, batch_size=32, num_workers=self.workers)

    def forward(self, x):
        return self.model(input_ids=x[0], attention_mask=x[1])

    def merge_lora_weights(self):
        self.model = self.model.merge_and_unload()

    def load_model(self, path):
        self.model = self.model_class.from_pretrained(path, num_labels=1)

    def init_lora(self):
        if self.train_all_params is True:
            for name, param in self.model.named_parameters():
                if any(substring in name for substring in self.base_modules):
                    param.requires_grad = True

    def reset_optimizer(self, in_place=False):
        for optimizer in self.trainer.optimizers:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    param_state = optimizer.state[p]
                    if in_place is False:
                        param_state["exp_avg"] = torch.zeros_like(p.data)
                        param_state["exp_avg_sq"] = torch.zeros_like(p.data)
                    else:
                        param_state["exp_avg"].zero_()
                        param_state["exp_avg_sq"].zero_()

    def on_train_epoch_end(self):
        if self.exists("accuracy"):
            self.accuracy.reset()
    
    def lora_checkpoint_reset(self, checkpoint: Dict[str, Any]):
        self.reset_optimizer()
        self.merge_lora_weights
        checkpoint["state_dict"] = self.model.state_dict()
        self.reset_optimizer(in_place=True) if self.trainer.scaler is not None else self.reset_optimizer()    
        self.init_lora()

        return checkpoint

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        if self.exists("merge_freq") and self.trainer.current_epoch % self.merge_freq == 0:
            checkpoint = self.lora_checkpoint_reset(checkpoint)
        return checkpoint

    def exists(self, attribute):
        return bool(hasattr(self, attribute) and bool(getattr(self, attribute)))
    
    @property
    def max_steps(self):
        return self.trainer.max_steps
    @property
    def total_steps(self):
        return int(self.trainer.max_steps if self.trainer.max_steps != -1
            else self.trainer.max_epochs * self.trainer.limit_train_batches if isinstance(self.trainer.limit_train_batches, int)
            else self.trainer.max_epochs * (math.ceil(self.train_set.__len__() / self.batch_size) * self.trainer.limit_train_batches) if isinstance(self.trainer.limit_train_batches, float)
            else self.trainer.max_epochs * math.ceil(self.train_set.__len__() / self.batch_size) if self.trainer.max_epochs != -1
            else 57600)