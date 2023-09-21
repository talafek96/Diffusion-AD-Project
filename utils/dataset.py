import torch
from torch.utils.data import Dataset
import os
import glob
import csv
from PIL import Image
import random
import torchvision.transforms as transforms

"""
Note: 
This module is a revamped version of extern/guided_diffusion/image_datasets.py
"""

class FilelistDataset(Dataset):
    def __init__(self, root, transform, phase, category):
        self.class_list = ['good', 'anomaly']

        self.csv_path = os.path.join(root, f'{phase}_{category}.csv')
        self.transform = transform
        self.img_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        tot_types = []

        with open(self.csv_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                img_tot_paths.append(row[0])
                if int(row[1]) > 0:
                    row[1] = '1'
                tot_labels.append(int(row[1]))
                tot_types.append(self.class_list[int(row[1])])

        return img_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        # gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        gt = -1

        return img, gt, label, os.path.basename(img_path[:-4]), img_type, img_path

class FolderDataset(Dataset):
    def __init__(self, dir_path, transform):
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(dir_path,'*.*'))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, -1, 0, os.path.basename(img_path[:-4]), [], img_path

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
            self.gt_path = None
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
            if not os.path.exists(self.gt_path):
                self.gt_path = None
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                img_tot_paths.extend(img_paths)
                if self.gt_path:
                    gt_tot_paths.extend([0] * len(img_paths))
                else:
                    gt_tot_paths.extend([-1] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.jpg") + \
                            glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                if self.gt_path:
                    gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png") + \
                               glob.glob(os.path.join(self.gt_path, defect_type) + "/*.jpg") + \
                               glob.glob(os.path.join(self.img_path, defect_type) + "/*.bmp")
                else:
                    gt_paths = [-1] * len(img_paths)
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if isinstance(gt, str):
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        elif gt == 0:
            gt = torch.zeros_like(img[0:1, :, :])

        # assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type, img_path


class ListDataset(Dataset): # relevant only for normals
    def __init__(self, img_root, transform, image_list):
        self.class_list = ['good', 'anomaly']
        self.transform = transform
        self.img_paths = [os.path.join(img_root,file) for file in image_list]
        self.labels = [0 for _ in self.img_paths]
        self.types = ['good' for _ in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        if isinstance(self.transform,list):
            r =random.random()
            p = 1/len(self.transform)
            for i,trans in enumerate(self.transform):
                if r<p*(i+1):
                    img = trans(img)
                    break
        else:
            img = self.transform(img)
        # gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        gt = -1

        return img, gt, label, os.path.basename(img_path[:-4]), img_type, img_path
