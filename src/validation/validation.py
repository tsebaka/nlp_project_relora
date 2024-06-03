import torch
import logging
import numpy as np

from tqdm import tqdm
from src.metric.metric import quadratic_weighted_kappa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(config, model, valid_dataloader, criterion, step, logger):
        model.eval()
        valid_loss = []
        valid_metric = []
        valid_preds = []
        valid_labels = []
        for batch in tqdm(valid_dataloader):
            input_ids, attention_mask, labels = batch
            with torch.no_grad():
                output = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
            
            loss = torch.sqrt(criterion(output["logits"].view(-1).to(device), labels.to(device).float()))
            
            valid_preds.append(output["logits"].cpu().detach())
            valid_loss.append(loss.item())
            valid_labels.append(labels.cpu().detach())
            
        logger.add_scalar("Valid loss:", np.sum(valid_loss) / len(valid_dataloader))
        logger.add_scalar("Valid metric", quadratic_weighted_kappa(torch.cat(valid_labels).view(-1), torch.cat(valid_preds).view(-1),
                                                       task=config.criterion.criterion_type), step)