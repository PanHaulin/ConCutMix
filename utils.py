from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20, acc_per_cls=False):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    class_shot=[]
    for i in range(len(train_class_count)):
        class_shot.append((class_correct[i] / test_class_count[i]))

        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot) ,class_shot



class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(gs[1], alpha=0.587 ).add_(gs[2],alpha=0.114)
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)


import math
import re
from PIL import Image
import PIL

import torchvision.transforms as transforms

_PIL_VER = tuple([int(x) for x in PIL.__version__.split('.')[:2]])

_FILL = (128, 128, 128)

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.

_HPARAMS_DEFAULT = dict(
    translate_const=250,
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _interpolation(kwargs):
    interpolation = kwargs.pop('resample', Image.BILINEAR)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    if 'fillcolor' in kwargs and _PIL_VER < (5, 0):
        kwargs.pop('fillcolor')
    kwargs['resample'] = _interpolation(kwargs)


def shear_x(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, factor, 0, 0, 1, 0), **kwargs)


def shear_y(img, factor, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, factor, 1, 0), **kwargs)


def translate_x_rel(img, pct, **kwargs):
    pixels = pct * img.size[0]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_rel(img, pct, **kwargs):
    pixels = pct * img.size[1]
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def translate_x_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), **kwargs)


def translate_y_abs(img, pixels, **kwargs):
    _check_args_tf(kwargs)
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), **kwargs)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    if _PIL_VER >= (5, 2):
        return img.rotate(degrees, **kwargs)
    elif _PIL_VER >= (5, 0):
        w, h = img.size
        post_trans = (0, 0)
        rotn_center = (w / 2.0, h / 2.0)
        angle = -math.radians(degrees)
        matrix = [
            round(math.cos(angle), 15),
            round(math.sin(angle), 15),
            0.0,
            round(-math.sin(angle), 15),
            round(math.cos(angle), 15),
            0.0,
        ]

        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        matrix[2], matrix[5] = transform(
            -rotn_center[0] - post_trans[0], -rotn_center[1] - post_trans[1], matrix
        )
        matrix[2] += rotn_center[0]
        matrix[5] += rotn_center[1]
        return img.transform(img.size, Image.AFFINE, matrix, **kwargs)
    else:
        return img.rotate(degrees, resample=kwargs['resample'])


def auto_contrast(img, **__):
    return ImageOps.autocontrast(img)


def invert(img, **__):
    return ImageOps.invert(img)


def identity(img, **__):
    return img


def equalize(img, **__):
    return ImageOps.equalize(img)


def solarize(img, thresh, **__):
    return ImageOps.solarize(img, thresh)


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def posterize(img, bits_to_keep, **__):
    if bits_to_keep >= 8:
        return img
    return ImageOps.posterize(img, bits_to_keep)


def contrast(img, factor, **__):
    return ImageEnhance.Contrast(img).enhance(factor)


def color(img, factor, **__):
    return ImageEnhance.Color(img).enhance(factor)


def brightness(img, factor, **__):
    return ImageEnhance.Brightness(img).enhance(factor)


def sharpness(img, factor, **__):
    return ImageEnhance.Sharpness(img).enhance(factor)


def _randomly_negate(v):
    """With 50% prob, negate the value"""
    return -v if random.random() > 0.5 else v


def _rotate_level_to_arg(level, _hparams):
    # range [-30, 30]
    level = (level / _MAX_LEVEL) * 30.
    level = _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, _hparams):
    # range [0.1, 1.9]
    return (level / _MAX_LEVEL) * 1.8 + 0.1,


def _shear_level_to_arg(level, _hparams):
    # range [-0.3, 0.3]
    level = (level / _MAX_LEVEL) * 0.3
    level = _randomly_negate(level)
    return level,


def _translate_abs_level_to_arg(level, hparams):
    translate_const = hparams['translate_const']
    level = (level / _MAX_LEVEL) * float(translate_const)
    level = _randomly_negate(level)
    return level,


def _translate_rel_level_to_arg(level, _hparams):
    # range [-0.45, 0.45]
    level = (level / _MAX_LEVEL) * 0.45
    level = _randomly_negate(level)
    return level,


def _posterize_original_level_to_arg(level, _hparams):
    # As per original AutoAugment paper description
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    return int((level / _MAX_LEVEL) * 4) + 4,


def _posterize_research_level_to_arg(level, _hparams):
    # As per Tensorflow models research and UDA impl
    # range [4, 0], 'keep 4 down to 0 MSB of original image'
    return 4 - int((level / _MAX_LEVEL) * 4),


def _posterize_tpu_level_to_arg(level, _hparams):
    # As per Tensorflow TPU EfficientNet impl
    # range [0, 4], 'keep 0 up to 4 MSB of original image'
    return int((level / _MAX_LEVEL) * 4),


def _solarize_level_to_arg(level, _hparams):
    # range [0, 256]
    return int((level / _MAX_LEVEL) * 256),


def _solarize_add_level_to_arg(level, _hparams):
    # range [0, 110]
    return int((level / _MAX_LEVEL) * 110),


LEVEL_TO_ARG = {
    'AutoContrast': None,
    'Equalize': None,
    'Invert': None,
    'Identity': None,
    'Rotate': _rotate_level_to_arg,
    # There are several variations of the posterize level scaling in various Tensorflow/Google repositories/papers
    'PosterizeOriginal': _posterize_original_level_to_arg,
    'PosterizeResearch': _posterize_research_level_to_arg,
    'PosterizeTpu': _posterize_tpu_level_to_arg,
    'Solarize': _solarize_level_to_arg,
    'SolarizeAdd': _solarize_add_level_to_arg,
    'Color': _enhance_level_to_arg,
    'Contrast': _enhance_level_to_arg,
    'Brightness': _enhance_level_to_arg,
    'Sharpness': _enhance_level_to_arg,
    'ShearX': _shear_level_to_arg,
    'ShearY': _shear_level_to_arg,
    'TranslateX': _translate_abs_level_to_arg,
    'TranslateY': _translate_abs_level_to_arg,
    'TranslateXRel': _translate_rel_level_to_arg,
    'TranslateYRel': _translate_rel_level_to_arg,
}


NAME_TO_OP = {
    'AutoContrast': auto_contrast,
    'Equalize': equalize,
    'Invert': invert,
    'Identity': identity,
    'Rotate': rotate,
    'PosterizeOriginal': posterize,
    'PosterizeResearch': posterize,
    'PosterizeTpu': posterize,
    'Solarize': solarize,
    'SolarizeAdd': solarize_add,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x_abs,
    'TranslateY': translate_y_abs,
    'TranslateXRel': translate_x_rel,
    'TranslateYRel': translate_y_rel,
}


class AutoAugmentOp:

    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fillcolor=hparams['img_mean'] if 'img_mean' in hparams else _FILL,
            resample=hparams['interpolation'] if 'interpolation' in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce some randomness
        # in the usually fixed policy and sample magnitude from a normal distribution
        # with mean `magnitude` and std-dev of `magnitude_std`.
        # NOTE This is my own hack, being tested, not in papers or reference impls.
        self.magnitude_std = self.hparams.get('magnitude_std', 0)

    def __call__(self, img):
        if random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std and self.magnitude_std > 0:
            magnitude = random.gauss(magnitude, self.magnitude_std)
        magnitude = min(_MAX_LEVEL, max(0, magnitude)) # clip to valid range
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)


_RAND_TRANSFORMS = [
    'AutoContrast',
    'Equalize',
    'Invert',
    'Rotate',
    'PosterizeTpu',
    'Solarize',
    'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    #'Cutout'  # FIXME I implement this as random erasing separately
]

_RAND_TRANSFORMS_CMC = [
    'AutoContrast',
    'Identity',
    'Rotate',
    'Sharpness',
    'ShearX',
    'ShearY',
    'TranslateXRel',
    'TranslateYRel',
    #'Cutout'  # FIXME I implement this as random erasing separately
]


# These experimental weights are based loosely on the relative improvements mentioned in paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    'Rotate': 0.3,
    'ShearX': 0.2,
    'ShearY': 0.2,
    'TranslateXRel': 0.1,
    'TranslateYRel': 0.1,
    'Color': .025,
    'Sharpness': 0.025,
    'AutoContrast': 0.025,
    'Solarize': .005,
    'SolarizeAdd': .005,
    'Contrast': .005,
    'Brightness': .005,
    'Equalize': .005,
    'PosterizeTpu': 0,
    'Invert': 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    """rand augment ops for RGB images"""
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AutoAugmentOp(
        name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


def rand_augment_ops_cmc(magnitude=10, hparams=None, transforms=None):
    """rand augment ops for CMC images (removing color ops)"""
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS_CMC
    return [AutoAugmentOp(
        name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(
            self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img


def rand_augment_transform(config_str, hparams, use_cmc=False):
    """
    Create a RandAugment transform
    :param config_str: String defining configuration of random augmentation. Consists of multiple sections separated by
    dashes ('-'). The first section defines the specific variant of rand augment (currently only 'rand'). The remaining
    sections, not order sepecific determine
        'm' - integer magnitude of rand augment
        'n' - integer num layers (number of transform ops selected per image)
        'w' - integer probabiliy weight index (index of a set of weights to influence choice of op)
        'mstd' -  float std deviation of magnitude noise applied
    Ex 'rand-m9-n3-mstd0.5' results in RandAugment with magnitude 9, num_layers 3, magnitude_std 0.5
    'rand-mstd1-w0' results in magnitude_std 1.0, weights 0, default magnitude of 10 and num_layers 2
    :param hparams: Other hparams (kwargs) for the RandAugmentation scheme
    :param use_cmc: Flag indicates removing augmentation for coloring ops.
    :return: A PyTorch compatible Transform
    """
    magnitude = _MAX_LEVEL  # default to _MAX_LEVEL for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    weight_idx = None  # default to no probability weights for op choice
    config = config_str.split('-')
    assert config[0] == 'rand'
    config = config[1:]
    for c in config:
        cs = re.split(r'(\d.*)', c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == 'mstd':
            # noise param injected via hparams for now
            hparams.setdefault('magnitude_std', float(val))
        elif key == 'm':
            magnitude = int(val)
        elif key == 'n':
            num_layers = int(val)
        elif key == 'w':
            weight_idx = int(val)
        else:
            assert False, 'Unknown RandAugment config section'
    if use_cmc:
        ra_ops = rand_augment_ops_cmc(magnitude=magnitude, hparams=hparams)
    else:
        ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams)
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import torch


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]


    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]


    def __call__(self, img):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img