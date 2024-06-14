"""
Author: Haoran Chen
Date: 2022.08.15
"""

import torch
from torch import nn
import tqdm
from clip_custom import clip
import os
import numpy as np
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset
from model import PromptGenerator, Custom_Clip


def make_dataset(target_name, path, clip_model, transform, args):
    class_list = os.listdir(path)
    class_list.sort()
    class_list_tokenize = []
    for i in range(len(class_list)):
        class_list_tokenize.append(f"A photo of a {class_list[i]}")
    text = clip.tokenize(class_list_tokenize).to(args.device)
    logit_scale = clip_model.logit_scale.exp()

    clip_model = nn.DataParallel(clip_model)

    data = datasets.ImageFolder(root=path, transform=transform)
    data_loader = torch.utils.data.DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    instances = []
    correct = 0
    tot = 0
    with torch.no_grad():
        print("Generating pseudo labels for {} data".format(target_name))
        for image, label in tqdm.tqdm(data_loader):
            image = image.to(args.device)
            label = label.cpu().numpy()

            image_features, text_features = clip_model(image, text)
            logits_per_image = logit_scale * image_features @ text_features.t()
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            pseudo_label = np.argmax(probs, axis=1)

            tot += image.shape[0]

            correct += np.count_nonzero(pseudo_label == label)
            for i in range(pseudo_label.shape[0]):
                data_image = image[i].cpu()
                data_label = pseudo_label[i]
                confidence = probs[i][data_label]
                if confidence > args.threshold:
                    item = data_image, data_label
                    instances.append(item)
                else:
                    item = data_image, -1
                    instances.append(item)

    print(f"pseudo label correct rate is {correct/tot}")
    return instances


class Pseudolabeldata(Dataset):
    def __init__(self, target_name, path, clip_model, transform, args):
        self.instances = make_dataset(target_name, path, clip_model, transform, args)

    def __getitem__(self, index):
        image, label = self.instances[index]
        return image, label

    def __len__(self):
        return len(self.instances)


def load_data(path, preprocess, args):
    data = datasets.ImageFolder(root=path, transform=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    return data_loader


def load_pseudo_label_data(target_name, path, preprocess, clip_model, args):
    target_pseudo_data = Pseudolabeldata(
        target_name, path, clip_model, preprocess, args
    )
    data_loader = torch.utils.data.DataLoader(
        target_pseudo_data,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    return data_loader
