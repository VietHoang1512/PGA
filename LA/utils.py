"""
Author: Haoran Chen
Date: 2022.08.15
"""

import os
from clip_custom import clip
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from torchvision import datasets
import tqdm
import numpy as np


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)


def target_text(target_path):
    target_classes = os.listdir(target_path)
    target_classes = [name.replace("_", " ") for name in target_classes]
    target_classes.sort()
    for i in range(len(target_classes)):
        target_classes[i] = "A photo of a " + target_classes[i]
    return target_classes


def Prompt(classnames, clip_model, prompt, args):
    dtype = torch.float32
    prompt_prefix = " ".join(["X"] * (args.M1 + args.M2))

    classnames = [name.replace("_", " ") for name in classnames]
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(args.device)

    with torch.no_grad():
        embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

    prefix = embedding[:, :1, :]
    suffix = embedding[:, 1 + args.M1 + args.M2 :, :]

    source_prompts = torch.cat(
        [
            prefix,  # (n_cls, 1, dim)
            prompt,  # (n_cls, M1 + 1, dim)
            suffix,  # (n_cls, *, dim)
        ],
        dim=1,
    )

    return source_prompts, tokenized_prompts


def l1(logits_list):
    l1_loss = 0
    while len(logits_list) > 1:
        logits1 = logits_list.pop()
        for logits2 in logits_list:
            l1_loss += torch.mean(
                torch.abs(
                    torch.nn.functional.softmax(logits1, dim=-1)
                    - torch.nn.functional.softmax(logits2, dim=-1)
                )
            )
    return l1_loss
