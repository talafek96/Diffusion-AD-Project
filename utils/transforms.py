import random
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, args):
        self.base_transform = base_transform
        self.args = args

    def trans(self, transform, x):
        if not isinstance(transform, tuple):
            q = transform(x)
            k = transform(x)
        else:
            q = transform[0](x)
            k = transform[1](x)
        return q, k

    def __call__(self, x):
        q, k = self.trans(self.base_transform, x)

        return [q, k]


def get_transforms(args):
    test_transforms = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=args.mean_train,
                             std=args.std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((args.load_size, args.load_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(args.input_size)])

    # defining train transform
    if isinstance(args.augment,str):
        args.augment = [args.augment]

    train_transforms = []
    for  augment in args.augment:
        if augment == 'basic':
            train_transforms.append(test_transforms)
        
        elif augment == 'rigid':
            train_transforms.append(transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(90,fill=args.fill_value),
                transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train,
                                     std=args.std_train)]))
        
        elif augment == 'non_rigid':
            train_transforms.append(transforms.Compose([
                transforms.RandomPerspective(p=1,fill=args.fill_value),
                transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train,
                                     std=args.std_train)]))
        
        elif augment == 'non_rigid+rigid':
            train_transforms.append(transforms.Compose([
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(90, fill=args.fill_value),
                transforms.RandomPerspective(p=1,fill=args.fill_value),
                transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train,
                                     std=args.std_train)]))
        
        elif augment == 'non_rigid+rigid+color':
            train_transforms.append(transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.5),
                transforms.RandomPosterize(6, p=0.3),
                transforms.RandomAdjustSharpness(2, p=0.5),
                transforms.RandomAdjustSharpness(0.5, p=0.5),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(90, fill=args.fill_value),
                transforms.RandomPerspective(p=1,fill=args.fill_value),
                transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train,
                                     std=args.std_train)]))
        
        elif augment == 'rigid+color':
            train_transforms.append(transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8),
                transforms.RandomPosterize(6, p=0.3),
                transforms.RandomAdjustSharpness(2, p=0.3),
                transforms.RandomAdjustSharpness(0.5, p=0.3),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(90, fill=args.fill_value),
                transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train, std=args.std_train)
             ]))

        elif augment == 'non_rigid+color':
            train_transforms.append(transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8),
                transforms.RandomPosterize(6, p=0.3),
                transforms.RandomAdjustSharpness(2, p=0.3),
                transforms.RandomAdjustSharpness(0.5, p=0.3),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomPerspective(p=1,fill=args.fill_value),
                transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train,
                                     std=args.std_train)]))
        
        elif augment == 'color':
            train_transforms.append(transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1) ], p=0.8),
                transforms.RandomPosterize(6, p=0.3),
                transforms.RandomAdjustSharpness(2, p=0.3),
                transforms.RandomAdjustSharpness(0.5, p=0.3),
                transforms.RandomAutocontrast(p=0.3),
                transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train,
                                     std=args.std_train)]))

        elif augment == 'extra':
            train_transforms.append(transforms.Compose([
                #transforms.AutoAugment(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(p=0.8,fill=args.fill_value),
                transforms.RandomRotation(90,fill=args.fill_value),
                transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train,
                                     std=args.std_train)]))
        
        elif augment == 'autoaugment+extra':
            rand = random.random()
            trans = []
            if rand>0.33:
                if args.auto_augment_policy == 'IMAGENET':
                    policy = transforms.AutoAugmentPolicy.IMAGENET
                elif args.auto_augment_policy == 'CIFAR10':
                    policy = transforms.AutoAugmentPolicy.CIFAR10
                elif args.auto_augment_policy == 'SVHN':
                    policy = transforms.AutoAugmentPolicy.SVHN
                else:
                    raise NotImplementedError
                trans.append(transforms.AutoAugment(policy=policy,fill=args.fill_value))
            if rand>0.66 or rand<0.33:
                trans.extend([transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomPerspective(p=0.8,fill=args.fill_value),
                transforms.RandomRotation(90,fill=args.fill_value)])
            trans.extend([transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train, std=args.std_train)])
            train_transforms.append(transforms.Compose(trans))
        
        elif augment == 'autoaugment':
            if args.auto_augment_policy == 'IMAGENET':
                policy = transforms.AutoAugmentPolicy.IMAGENET
            elif args.auto_augment_policy == 'CIFAR10':
                policy = transforms.AutoAugmentPolicy.CIFAR10
            elif args.auto_augment_policy == 'SVHN':
                policy = transforms.AutoAugmentPolicy.SVHN
            else:
                raise NotImplementedError
            train_transforms.append(transforms.Compose([
                transforms.AutoAugment(policy=policy,fill=args.fill_value),
                transforms.Resize((args.load_size, args.load_size), InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.RandomCrop(args.input_size),
                transforms.Normalize(mean=args.mean_train,
                                     std=args.std_train)]))
        else:
            raise NotImplementedError

    if len(train_transforms)==1:
        train_transforms = train_transforms[0]

    return train_transforms, test_transforms, gt_transforms
