import time
from collections import Counter

import numpy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
np.random.seed()
class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self,args, root, imb_type='exp', imb_factor=0.01, rand_number=10, train=True,
                 transform=None, target_transform=None,
                 download=False):
        t = time.time()
        self.args=args
        self.labels = []
        self.imb_factor = imb_factor
        self.imb_type = imb_type
        np.random.seed(rand_number)
        random.seed(rand_number)
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)
        self.cls_num_list = self.get_cls_num_list()

        t = time.time()
        if (self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
            self.class_dict = self._get_class_dict()

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        self.labels = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])

        return cls_num_list

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight


    def get_annotations(self):
        annos = []
        for target in self.targets:
            annos.append({'category_id': int(target)})
        return annos




    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.args.Background_sampler == "reverse":
            sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
        elif self.args.Background_sampler == "balance":
            sample_class = random.randint(0, self.cls_num - 1)
            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
        elif self.args.Background_sampler == "uniform":
            # sample_index = random.randint(0, self.__len__() - 1)
            sample_index = index
        img_A, target_A = self.data[sample_index], self.targets[sample_index]


        if  self.train:
            assert self.args.Foreground_sampler in ["balance", "reverse"]
            if  self.args.Foreground_sampler == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
            elif self.args.Foreground_sampler == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
            img_B, target_B = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image



        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            if self.train:

                sample_A = Image.fromarray(img_A)
                sample_A1 = self.transform[0](sample_A)
                sample_A2 = self.transform[1](sample_A)
                sample_A3 = self.transform[2](sample_A)
                sample_B = Image.fromarray(img_B)
                sample_B1 = self.transform[0](sample_B)
                sample_B2 = self.transform[1](sample_B)
                sample_B3 = self.transform[2](sample_B)
                if (len(self.transform) > 3):
                    sample_A4 = self.transform[3](sample_A)

                    return [sample_A1, sample_A2, sample_A3,sample_A4], [sample_B1, sample_B2, sample_B3], target_A, target_B

                else:
                    return [sample_A1, sample_A2, sample_A3], [sample_B1, sample_B2, sample_B3], target_A, target_B

            else:
                sample = Image.fromarray(img_A)

                return self.transform(sample), target_A

        if self.target_transform is not None:
            target_A = self.target_transform(target_A)
            return img_A ,target_A


# new dataset with cutmix before aug, ouput [Sample_A1, Sample_A2, Sample_A3], ta_onehot 

class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


def rand_bbox_without_bs(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class Args():

    def __init__(self):
        self.SAMPLER_TYPE="weighted sampler"
        self.Foreground_sampler="reverse"
        self.Background_sampler="uniform"
        self.beta=48
        self.cutmix_prob=48
        self.cutmix=1


if __name__ == '__main__':
    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        transforms.ToTensor(),

    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),

    ]
    transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                       transforms.Compose(augmentation_sim), ]


    args=Args()
    trainset = IMBALANCECIFAR10_weight(args,root="/home/pc/utils/datasets/", train=True,
                                 download=True, transform=transform_train,imb_factor=0.01)
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=(train_sampler is None),
         pin_memory=True,num_workers=12)
    arr=np.zeros(10)
    arr1=np.zeros(10)
    time1=time.time()
    A =None
    B =None
    for i,data in  enumerate (train_loader):
        [sample_A1, sample_A2, sample_A3], sample_B1, target_A, target_B=data
        if(A==None):
            A=Counter(target_A.numpy())
            B=Counter(target_B.numpy())
        else:
            A += Counter(target_A.numpy())
            B += Counter(target_B.numpy())
        print("A",A)
        print("B",B)
        print("AB",A+B)


