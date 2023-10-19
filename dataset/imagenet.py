import os
from collections import Counter

import numpy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class ImageNetLT(Dataset):

    def __init__(self, root, txt,args, transform=None, train=True, class_balance=False):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.num_classes = 1000
        self.train = train
        self.class_balance = class_balance
        self.args=args
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y = self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]
        self.targets = self.labels  # Sampler needs to use targets
        print("data init finish")
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if  self.args.Background_sampler == "balance":
            label = random.randint(0, self.num_classes - 1)
            A_index = random.choice(self.class_data[label])
            A_path = self.img_path[A_index]
        elif self.args.Background_sampler == "uniform":
            A_path = self.img_path[index]
            A_label = self.labels[index]

        if  self.train:
            assert self.args.Foreground_sampler in ["balance"]
            if self.args.Foreground_sampler == "balance":
                B_label = random.randint(0, self.num_classes - 1)
                B_index = random.choice(self.class_data[B_label])
                B_path = self.img_path[B_index]



        with open(A_path, 'rb') as f:
            sample_A = Image.open(f).convert('RGB')

        with open(B_path, 'rb') as f:
            sample_B = Image.open(f).convert('RGB')
        if self.transform is not None:
            if self.train:
                sample_A1 = self.transform[0](sample_A)
                sample_A2 = self.transform[1](sample_A)
                sample_A3 = self.transform[2](sample_A)
                sample_B1 = self.transform[0](sample_B)
                sample_B2 = self.transform[1](sample_B)
                sample_B3 = self.transform[2](sample_B)
                return [sample_A1, sample_A2, sample_A3], [sample_B1, sample_B2, sample_B3], A_label, B_label  # , index
            else:
                return self.transform(sample_A), A_label



