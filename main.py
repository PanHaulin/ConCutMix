import argparse
import os
import random
import shutil
import time
import warnings
import math
import neptune.new as neptune
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models_office
import csv
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from dataset.cifar import IMBALANCECIFAR10
from dataset.cifar import IMBALANCECIFAR100
from dataset.imagenet import ImageNetLT
from dataset.inaturalist import INaturalist
from dataset.PlacesLT import PlacesLT
from loss.contrastive import BalSCL
from loss.logitadjust import LogitAdjust, cutmix_cross_entropy
from models.resnet32 import BCLModel_32
from models.resnext import BCLModel
from utils import rand_augment_transform
from utils import shot_acc, GaussianBlur
from utils import CIFAR10Policy
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='imagenet', choices=['inat', 'imagenet', 'cifar10', 'cifar100','Places_LT'])
parser.add_argument('--data', default='/DATACENTER/raid5/zjg/imagenet', metavar='DIR')
parser.add_argument('--arch', default='resnext50', choices=['resnet50', 'resnext50', 'resnet32', 'resnet152' ,'resnext101'])
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--epochs', default=90, type=int)
parser.add_argument('--temp', default=0.07, type=float, help='scalar temperature for contrastive learning')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[160, 180], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--alpha', default=1.0, type=float, help='cross entropy loss weight')
parser.add_argument('--beta', default=0.35, type=float, help='supervised contrastive loss weight')
parser.add_argument('--randaug', default=True, type=bool, help='use RandAugmentation for classification branch')
parser.add_argument('--cl_views', default='sim-sim', type=str,
                    choices=['sim-sim', 'sim-rand', 'rand-rand', 'cutout-sim', 'none-sim', "cutout-none",
                             "uncutout-sim", "unauto-sim", "cutmix-sim", "cutoutmix-sim", "uncutout-cutmix_sim"],
                    help='Augmentation strategy for contrastive learning views')
parser.add_argument('--feat_dim', default=1024, type=int, help='feature dimension of mlp head')
parser.add_argument('--warmup_epochs', default=0, type=int,
                    help='warmup epochs')
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--cos', default=False, action='store_true',
                    help='lr decays by cosine scheduler. ')
parser.add_argument('--use_norm', action='store_true',
                    help='cosine classifier.')
parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
parser.add_argument('--reload', default=False, type=bool, help='load supervised model')
parser.add_argument('--imb_factor', default=1, type=float)
parser.add_argument('--grad_c', action='store_true', )
parser.add_argument('--file_name', default="", type=str)
parser.add_argument('--device_ids', default=[0, 1, 2, 3], type=int, nargs="*")
parser.add_argument('--save_epoch', default=None, type=int)
parser.add_argument('--auto_resume', action='store_true')
parser.add_argument('--reload_torch', default=None, type=str,
                    help='load supervised model from torchvision')
parser.add_argument('--num_classes', default=None, type=int,
                    help='num_classes')
# neptune
parser.add_argument('--logger', default="none", type=str, choices=["neptune", "none"])
parser.add_argument('--ne_token', default="", type=str)
parser.add_argument('--ne_project', default="", type=str)
parser.add_argument('--ne_run', default=None, type=str)


# ablation
parser.add_argument('--Background_sampler', default="uniform", type=str, choices=["balance", "reverse", "uniform"])
parser.add_argument('--Foreground_sampler', default="balance", type=str, choices=["reverse", "balance", "uniform"])

# cutmix:

parser.add_argument('--cutmix_prob', default=0.5, type=float,
                    help='cutmix probability')


# Contrastive CutMix
parser.add_argument('--l_d_warm', default=0, type=int)
parser.add_argument('--scaling_factor', default=[2, 256], nargs='*', type=int,
                    help='scaling_factor=[a,b]=a/b')
parser.add_argument('--tau', default=1, type=float)
parser.add_argument('--topk', default=1, type=int)


def main():





    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.file_name, args.dataset, args.arch, 'batchsize', str(args.batch_size), 'epochs', str(args.epochs), 'temp',
         str(args.temp),"cutmix_prob",str(args.cutmix_prob), "topk", str(args.topk), "scaling_factor", str(args.scaling_factor[0]), str(args.scaling_factor[1]),"tau",str(args.tau)
         ,'lr', str(args.lr), args.cl_views])
    print(args.store_name)
    if args.seed is not None:
        random.seed(args.seed)
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')


    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    logger_run = None
    print(args.logger)
    if (args.logger == "neptune"):
        if (args.ne_run != None):

            logger_run = neptune.init_run(with_id=args.ne_run,project=args.ne_project,
                                      api_token=args.ne_token,
                                     description=args.file_name)
        else:
            logger_run = neptune.init_run(project=args.ne_project,
                                      api_token=args.ne_token,
                                      description=args.file_name
                                      )

        logger_run["dataset"] = args.dataset
        logger_run["arch"] = args.arch
        logger_run["epoch"] = args.epochs
        logger_run["scaling_factor"] = args.scaling_factor
        logger_run["l_d_warm"] = args.l_d_warm
        logger_run["topk"] = args.topk
        logger_run["prob"] = args.cutmix_prob
        logger_run["args"] = args
        logger_run["tau"] = args.tau



    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))


    if args.arch == 'resnet50':
        model = BCLModel(name='resnet50', feat_dim=args.feat_dim,
                                 num_classes=args.num_classes ,

                                 use_norm=args.use_norm)
    elif args.arch == 'resnext50':
        model = BCLModel(name='resnext50', feat_dim=args.feat_dim,num_classes=args.num_classes,
                                 use_norm=args.use_norm)
    elif args.arch == 'resnet32':

        model = BCLModel_32(name='resnet32', feat_dim=args.feat_dim,
                                   num_classes=args.num_classes,
                                   use_norm=args.use_norm)
    elif args.arch =="resnet152":
        model = BCLModel(name='resnet152', feat_dim=args.feat_dim,
                                 num_classes=args.num_classes ,
                                 use_norm=args.use_norm)
    elif args.arch == 'resnext101':
        model = BCLModel(name='resnext101', feat_dim=args.feat_dim,
                                 num_classes=args.num_classes ,
                                 use_norm=args.use_norm)
    else:
        raise NotImplementedError('This model is not supported')
    # print(model)



    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids).cuda()

    # model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    best_acc1 = 0.0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            print("best_acc1",best_acc1)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.auto_resume:
            filename = os.path.join(args.root_log, args.store_name, 'ConCutMix_ckpt.pth.tar')
            if os.path.isfile(filename):
                print("=> auto loading checkpoint '{}'".format(filename))
                checkpoint = torch.load(filename, map_location='cuda:0')
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                print("best_acc1",best_acc1)
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(filename, checkpoint['epoch']))
            else:
                print("=> no auto checkpoint found at '{}'".format(filename))
    elif args.reload_torch:
            state_dict = model.state_dict()
            state_dict_imagenet = torch.load(args.reload_torch)
            for key in state_dict.keys():
                    newkey = key[8:]
                    if newkey in state_dict_imagenet.keys() and state_dict[key].shape == state_dict_imagenet[newkey].shape:
                        state_dict[key]=state_dict_imagenet[newkey]
                        print(newkey+" ****loaded******* ")
                    else:
                        print(key+" ****unloaded******* ")
            model.load_state_dict(state_dict)
    # cudnn.benchmark = True

    normalize = transforms.Normalize((0.466, 0.471, 0.380), (0.195, 0.194, 0.192)) if args.dataset == 'inat' \
        else transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    rgb_mean = (0.485, 0.456, 0.406)
    ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
    if not os.path.exists('{}/'.format(os.path.join(args.root_log, args.store_name))):  # 判断所在目录下是否有该文件名的文件夹
        os.mkdir(os.path.join(args.root_log, args.store_name))

    augmentation_randnclsstack = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation_sim = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize
    ]
    augmentation_sim_cifar = [
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    Uncut_augmentation_regular = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # add AutoAug
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    augmentation_randncls = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0),
        rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(args.randaug_n, args.randaug_m), ra_params),
        transforms.ToTensor(),
        normalize,
    ]

    if args.cl_views == 'sim-sim':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'sim-rand':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_sim), ]
    elif args.cl_views == 'randstack-randstack':
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_randnclsstack),
                           transforms.Compose(augmentation_randnclsstack), ]
    elif args.cl_views == "uncutout-sim":
        transform_train = [transforms.Compose(Uncut_augmentation_regular), transforms.Compose(augmentation_sim_cifar),
                           transforms.Compose(augmentation_sim_cifar), ]
    else:
        raise NotImplementedError("This augmentations strategy is not available for contrastive learning branch!")

    if (args.dataset == 'inat'):
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_train = f'../dataset/iNaturalist18/iNaturalist18_train.txt'
        txt_val = f'../dataset/iNaturalist18/iNaturalist18_val.txt'
        val_dataset = INaturalist(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False,args=args
        )

        train_dataset = INaturalist(
                root=args.data,
                txt=txt_train,
                args=args,
                transform=transform_train
            )
    elif args.dataset == 'imagenet':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_val = f'../dataset/ImageNet_LT/ImageNet_LT_val.txt'
        txt_train = f'../dataset/ImageNet_LT/ImageNet_LT_train.txt'
        train_dataset = ImageNetLT(
                root=args.data,
                args=args,
                txt=txt_train,
                transform=transform_train)
        val_dataset = ImageNetLT(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False,args=args)

    elif args.dataset == 'cifar10':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_dataset = IMBALANCECIFAR10(root=args.data, args=args,
                                       transform=val_transform,
                                       train=False, imb_factor=1,download=True)
        train_dataset = IMBALANCECIFAR10(
                root=args.data, args=args,download=True,
                imb_factor=args.imb_factor,
                transform=transform_train)
    elif args.dataset == 'Places_LT':
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        txt_val = f'../dataset/Places_LT/Places_LT_val.txt'
        txt_train = f'../dataset/Places_LT/Places_LT_train.txt'
        train_dataset = PlacesLT(
                root=args.data,
                args=args,
                txt=txt_train,
                transform=transform_train)
        val_dataset = PlacesLT(
            root=args.data,
            txt=txt_val,
            transform=val_transform, train=False,args=args)
    else:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        val_dataset = IMBALANCECIFAR100(root=args.data, args=args,
                                        download=True,
                                        transform=val_transform,
                                        train=False, imb_factor=1)
        train_dataset = IMBALANCECIFAR100(
                root=args.data, args=args,
                download=True,
                imb_factor=args.imb_factor,
                transform=transform_train)

    cls_num_list = train_dataset.cls_num_list
    args.cls_num = len(cls_num_list)
    print(len(cls_num_list))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    criterion_scl = BalSCL(cls_num_list, args.temp).cuda(args.gpu)
    criterion_ce = LogitAdjust(cls_num_list, tau=args.tau).cuda(args.gpu)
    criterion_ce_cutmix = cutmix_cross_entropy(cls_num_list, args.tau).cuda(args.gpu)

    if args.reload:
        if (args.dataset == 'inat'):
            txt_test = f'../dataset/iNaturalist18/iNaturalist18_val.txt'
            test_dataset = INaturalist(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False,args=args)
        elif args.dataset == 'imagenet':
            txt_test = f'../dataset/ImageNet_LT/ImageNet_LT_test.txt'
            test_dataset = ImageNetLT(
                root=args.data,
                txt=txt_test,
                transform=val_transform, train=False,args=args)
        elif args.dataset == 'cifar10':
            test_dataset = IMBALANCECIFAR10(root=args.data, args=args, transform=val_transform, train=False,
                                            imb_factor=1,download=True)
        elif args.dataset == 'Places_LT':
            txt_train = f'../dataset/Places_LT/Places_LT_val.txt'

            test_dataset = PlacesLT(
                root=args.data,
                txt=txt_val,
                transform=val_transform, train=False,args=args)

        else:
            test_dataset = IMBALANCECIFAR100(root=args.data, args=args, transform=val_transform, train=False,
                                             imb_factor=1,
                                             download=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        acc1, many, med, few, class_acc = validate(train_loader, test_loader, model, criterion_ce, 1, args)
        print('Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'
              .format(acc1, many, med, few,))

        return
    print("start train")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(optimizer, epoch, args)
        ce_loss_all,scl_loss_all,top1,loss=train(train_loader, model, criterion_ce, criterion_ce_cutmix, criterion_scl, optimizer,
              epoch, args,
              logger_run, cls_num_list)
        # evaluate on validation set
        acc1, many, med, few, class_acc = validate(train_loader, val_loader, model, criterion_ce, epoch, args,
                                             )
        if (args.logger == "neptune"):
            logger_run["few_acc"].log(few,step=epoch)
            logger_run["val_acc"].log(acc1,step=epoch)
            logger_run["many_acc"].log(many,step=epoch)
            logger_run["median_acc"].log(med,step=epoch)
            logger_run["CE_loss/train"].log(ce_loss_all, step=epoch, )
            logger_run["SCL_loss/train"].log(scl_loss_all, step=epoch)
            logger_run["train_acc"].log(top1, step=epoch)
            logger_run["train_loss"].log(loss, step=epoch)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_many = many
            best_med = med
            best_few = few
            best_class_acc = class_acc
            if(logger_run!=None):
                logger_run["few_acc_top1"].log(best_few,step=epoch)
                logger_run["val_acc_top1"].log(best_acc1,step=epoch)
                logger_run["many_acc_top1"].log(best_many,step=epoch)
                logger_run["median_acc_top1"].log(best_med,step=epoch)
            print(
                'Best Prec@1: {:.3f}, Many Prec@1: {:.3f}, Med Prec@1: {:.3f}, Few Prec@1: {:.3f}'.format(
                    best_acc1,
                    best_many,
                    best_med,
                    best_few,
                    ))
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion_ce, criterion_ce_cutmix, criterion_scl, optimizer, epoch,
          args,
          logger_run, cls_num_list):
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    scl_loss_all = AverageMeter('SCL_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    end = time.time()

    model.train()
    for i, data in enumerate(train_loader):
        sample_A, sample_B, target_A, target_B = data  # !Modified: n_views args testing
        batch_size = target_A.shape[0]
        target_A, target_B = target_A.cuda(), target_B.cuda()
        # cutmix
        sample_A[0], sample_A[1], sample_A[2], = sample_A[0].cuda(), sample_A[1].cuda(), sample_A[2].cuda()
        sample_B[0], sample_B[1], sample_B[2], = sample_B[0].cuda(), sample_B[1].cuda(), sample_B[2].cuda()
        lam = np.random.beta(1, 1)
        rand_index = torch.randperm(sample_B[0].size()[0]).cuda()
        r = np.random.rand(1)

        if r < args.cutmix_prob:
                target_a = target_A
                target_b = target_B[rand_index]
                ta = torch.nn.functional.one_hot(target_a, num_classes=args.num_classes)
                tb = torch.nn.functional.one_hot(target_b, num_classes=args.num_classes)
                bbx1, bby1, bbx2, bby2 = rand_bbox(sample_B[0].size(), lam)
                cutmix_sample1 = sample_A[0].clone()
                cutmix_sample1[:, :, bbx1:bbx2, bby1:bby2] = sample_B[0][rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (sample_B[0].size()[-1] * sample_B[0].size()[-2]))
                target_cutmix = (torch.tensor([lam]).cuda() * ta) + ((1 - torch.tensor([lam])).cuda() * tb)
                inputs = torch.cat([cutmix_sample1, sample_A[1], sample_A[2]], dim=0)
                inputs = inputs.cuda()
                feat_mlp, logits, centers, uncenters, unfeat = model(inputs)
                uncenters = uncenters[:args.cls_num]
                logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
                f1, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
                unfeat1, unfeat2, unfeat3 = torch.split(unfeat, [batch_size, batch_size, batch_size], dim=0)

                if ((epoch > args.l_d_warm) ):

                        target_lam = get_semantically_consistent_label(unfeat1, uncenters, target_cutmix, args.scaling_factor,
                                                             cls_num_list, args.topk)
                        ce_loss = criterion_ce_cutmix(logits, target_lam)
                else:
                        ce_loss = criterion_ce(logits, target_a) * lam + criterion_ce(logits, target_b,
                                                                                  ) * (1. - lam)

                features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
                centers = centers[:args.cls_num]
                scl_loss = criterion_scl(centers, features, target_A, )
        else:
            inputs = torch.cat([sample_A[0], sample_A[1], sample_A[2]], dim=0)
            inputs = inputs.cuda()
            feat_mlp, logits, centers, uncenter, unfeat = model(inputs)
            logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
            _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            centers = centers[:args.cls_num]
            uncenter = uncenter[:args.cls_num]
            ce_loss = criterion_ce(logits, target_A)

            scl_loss = criterion_scl(centers, features, target_A,)

        loss = args.alpha * ce_loss + args.beta * scl_loss

        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)
        # 梯度累积
        if (args.grad_c):
            loss = loss / (256 / batch_size)
            loss.backward()
            if ((i + 1) % (256 / batch_size) == 0):
                optimizer.step()
                optimizer.zero_grad()


        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}] \t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'SCL_Loss {scl_loss.val:.4f} ({scl_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'loss {loss:.4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                ce_loss=ce_loss_all, scl_loss=scl_loss_all, top1=top1, loss=loss))  # TODO
            print(output)


        ce_loss_all.update(ce_loss.item(), batch_size)
        scl_loss_all.update(scl_loss.item(), batch_size)

        acc1 = accuracy(logits, target_A, topk=(1,))
        top1.update(acc1[0].item(), batch_size)


    return  ce_loss_all.avg,scl_loss_all.avg,top1.avg,loss



def validate(train_loader, val_loader, model, criterion_ce, epoch, args, flag='val'):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    total_logits = torch.empty((0, args.cls_num)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = targets.size(0)

            feat_mlp, logits, centers, _, __ = model(inputs)

            ce_loss = criterion_ce(logits, targets)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
            ce_loss_all.update(ce_loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)

            batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      .format(
                i, len(val_loader), batch_time=batch_time, ce_loss=ce_loss_all, top1=top1,
                )) 
            print(output)
        probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        many_acc_top1, median_acc_top1, low_acc_top1, class_acc = shot_acc(preds, total_labels, train_loader,
                                                                        acc_per_cls=False)
        return top1.avg, many_acc_top1, median_acc_top1, low_acc_top1, class_acc


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


def save_checkpoint(args, state, is_best):
    filename = os.path.join(args.root_log, args.store_name,'ConCutMix_ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))



def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.epochs - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
            # lr *= 0.1 if epoch == milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




def get_semantically_consistent_label(feature, center, target, scaling_factor,cls,k):
    #get the scaling factor omega
    scaling_factor=scaling_factor[0]/scaling_factor[1]
    #get N
    cls_num_list = torch.cuda.FloatTensor(cls)
    #sum(log(N_i))
    weight=torch.log((cls_num_list*target).sum(1))
    #N /sum(log(N_i))
    weight=(weight/(torch.log(cls_num_list).sum())).reshape(-1,1)
    target_de = target.detach()
    center_de = center.detach()
    feature_de = feature.detach()
    # get the euclidean distance
    sim = torch.sqrt(torch.sum((feature_de[:, None, :] - center_de) ** 2, dim=2))
    sim = 1 / sim
    # top K
    indices_to_remove = sim < torch.topk(sim, k)[0][..., -1, None]
    sim[indices_to_remove] = 0
    final_sim = sim
    # normlaization
    label = F.normalize(final_sim, p=1, dim=1)
    label = (weight*scaling_factor) * label + (1 - weight*scaling_factor) * target_de
    return label


if __name__ == '__main__':
    main()
    # center=torch.tensor([[1.21,1],[1,4],[2,5],[3,5]])
    # target=torch.tensor([0,2,0,1,2])
    # feature=torch.tensor([[1.1,1],[2,6],[2,50],[33,5],[34,1]])
    # target=F.one_hot(target,4)
    # print(get_semantically_consistent_label(feature,center,target,0.01,[100,10,5,1],2))
    # targetA = torch.tensor([0, 2, 0, 3])
    # targetB = torch.tensor([1, 0, 2, 1])
    # lam=0.1
    # center = torch.tensor([[1.21, 1], [1, 4], [2, 5], [3, 5]])
    # print(get_distance_cutmix_5(center, targetA,targetB,lam, 0.01, [9, 6, 3, 2]))
    exit(0)