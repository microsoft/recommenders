from typing import Type

import torch
from torch import nn, optim

OPTIM_DICT: dict[str, Type[optim.Optimizer]] = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'sgd': optim.SGD,
    'sparseadam': optim.SparseAdam,
}

LOSS_DICT: dict[str, Type[nn.Module]] = {
    'l1': nn.L1Loss,
    'mse': nn.MSELoss,
    'cross_entropy': nn.CrossEntropyLoss,
    'nll': nn.NLLLoss,
    'bce': nn.BCELoss,
    'bce_with_logits': nn.BCEWithLogitsLoss,
    'hinge': nn.HingeEmbeddingLoss,
    'kl_div': nn.KLDivLoss,
    'huber': nn.HuberLoss,
    'smooth_l1': nn.SmoothL1Loss,
    'soft_margin': nn.SoftMarginLoss,
    'multi_margin': nn.MultiMarginLoss,
    'cosine_embedding': nn.CosineEmbeddingLoss,
    'margin_ranking': nn.MarginRankingLoss,
    'triplet_margin': nn.TripletMarginLoss,
    'ctc': nn.CTCLoss,
}