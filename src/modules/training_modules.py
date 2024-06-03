import torch
from torch.nn import CrossEntropyLoss
from .relora_module import ReloraModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReloraModuleForClassification(ReloraModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.MSELoss()
        self.val_losses = []

    def training_step(self, batch, batch_idx):
        
        output = self([batch["input_ids"], batch["attention_mask"]])
        logits = output["logits"].view(-1).to(device)
        labels = batch["labels"]

        loss = self.loss(logits.to(torch.float32).view(-1), labels.to(torch.float32).view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self([batch["input_ids"], batch["attention_mask"]])
        logits = output["logits"].view(-1).to(device)
        labels = batch["labels"].to(device)
        
        val_loss = self.loss(logits.to(device), labels.to(device))
        self.val_losses.append(val_loss)
        self.log("val_loss", val_loss)
        return val_loss
    
    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.val_losses).mean()
        self.log("avg_val_loss", avg_val_loss, prog_bar=True)
        self.val_losses.clear()
