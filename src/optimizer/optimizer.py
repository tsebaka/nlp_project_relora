from torch.optim import AdamW


def get_optimizer(model):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    return optimizer