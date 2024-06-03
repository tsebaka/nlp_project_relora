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
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import DebertaV2Config, DebertaV2ForSequenceClassification


import torch._dynamo
torch._dynamo.config.suppress_errors = True
peft_supported_layers = [""]

class ReloraModule(L.LightningModule):
    def __init__(self,
                 model_class: nn.Module,
                 model_path: str,
                 lora_config: Any = None,
                 lora_merge_freq: int = 5,
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
        
        model = AutoModelForSequenceClassification.from_pretrained("src/models")
        self.lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.01, bias="none", inference_mode=False)
        self.model = get_peft_model(model, self.lora_config)
        self.count_parameters(self.model)
        
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
        # pass
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
        return DataLoader(self.train_set, batch_size=128, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.eval_set, batch_size=128, num_workers=self.workers)

    def forward(self, x):
        return self.model(input_ids=x[0], attention_mask=x[1])

    def merge_lora_weights(self):
        self.model = self.model.merge_and_unload()
        
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    def count_parameters(self, model):
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        
        for param in model.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            else:
                non_trainable_params += num_params
        
        print(total_params, trainable_params, non_trainable_params)
        
        
    def load_model(self, path):
#         config = AutoConfig.from_pretrained(path, num_labels=1)
#         model = AutoModelForSequenceClassification.from_config(config)
        
#     def load_model(self, path):
#         config = AutoConfig.from_pretrained(path, num_labels=1)

#         # Создать новую модель с той же конфигурацией, но с необученными весами
#         self.model = AutoModelForSequenceClassification(config)
        # self.model = self.model_class.from_pretrained(path, num_labels=1)
        #         # Создаем конфигурацию для DeBERTa v3 large
        # config = DebertaV2Config(
        #     vocab_size=50265,  # Обычно размер словаря для больших моделей
        #     hidden_size=1024,  # Размер скрытого слоя для large модели
        #     num_hidden_layers=24,  # Количество скрытых слоев
        #     num_attention_heads=16,  # Количество голов в механизме внимания
        #     intermediate_size=4096,  # Размер промежуточного слоя
        #     hidden_dropout_prob=0.1,  # Dropout для скрытого слоя
        #     attention_probs_dropout_prob=0.1,  # Dropout для механизма внимания
        #     max_position_embeddings=512,  # Максимальное количество позиций для эмбеддингов
        #     num_labels=1  # Количество классов для классификации
        # )
        pass
        # for layer in model.modules():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
        #         if layer.bias is not None:
        #             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        #             bound = 1 / math.sqrt(fan_in)
        #             nn.init.uniform_(layer.bias, -bound, bound)
        #     elif isinstance(layer, nn.Conv2d):
        #         nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        #         if layer.bias is not None:
        #             nn.init.constant_(layer.bias, 0)
        #     elif isinstance(layer, nn.Embedding):
        #         nn.init.normal_(layer.weight, mean=0, std=layer.embedding_dim ** -0.5)
        #         if layer.padding_idx is not None:
        #             nn.init.constant_(layer.weight[layer.padding_idx], 0)
        # self.model = model
        # self.freeze_model
        
        # self.model = DebertaV2ForSequenceClassification(config)
        
        # self.count_parameters(self.model)
        # self.freeze_model()

    def init_lora(self):
        self.model = get_peft_model(self.model, self.lora_config)
        self.count_parameters(self.model)

        # lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.01, bias="none", inference_mode=False)
        # self.model = get_peft_model(self.model, lora_config)
        self.count_parameters(self.model)
        if self.train_all_params is True:
            for name, param in self.model.named_parameters():
                if any(substring in name for substring in self.base_modules):
                    param.requires_grad = True
        # self.count_parameters(self.model)
        # self.model = AutoModelForSequenceClassification.from_pretrained("src/models")
        # lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.01, bias="none", inference_mode=False)
        # self.model = get_peft_model(self.model, lora_config)
        # self.count_parameters(self.model)
        pass

        # self.freeze_model()

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
        print("=="*10)
        self.reset_optimizer()
        self.merge_lora_weights
        self.model = get_peft_model(self.model, self.lora_config)
        self.count_parameters(self.model)
        # self.model.save_pretrained("src/models")
        checkpoint["state_dict"] = self.model.state_dict()
        self.reset_optimizer(in_place=True) if self.trainer.scaler is not None else self.reset_optimizer()    
        self.init_lora()
        self.count_parameters(self.model)

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
