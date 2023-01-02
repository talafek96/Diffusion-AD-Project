import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from PIL import Image
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from scipy.ndimage import gaussian_filter
from abc import ABC, abstractmethod
from dataset import MVTecDataset, FilelistDataset, ListDataset, syntheticListDataset,  FolderDataset
import shutil
from utils.transforms import get_transforms

def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name) and full_file_name.endswith('.py'):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)

def prep_dirs(root):
    output_path = root
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE']) # copy source code
    return output_path, sample_path, source_code_save_path

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

class BaseAlgo(pl.LightningModule):
    def __init__(self, args):
        super(BaseAlgo, self).__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.init_results_list()

        self.train_data_transforms, self.data_transforms, self.gt_transforms = get_transforms(self.args)

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

        self.train_list = []
        self.test_list = []

        self.test_image_folder = ''

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type, score):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        # anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm = (anomaly_map-1) / 3.5
        anomaly_map_norm[anomaly_map_norm>0.9] = 0.9
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        score = float(score)
        cv2.imwrite(os.path.join(self.sample_path, f'score{score:.2f}_{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'score{score:.2f}_{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'score{score:.2f}_{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)
        if gt_img is not None:
            cv2.imwrite(os.path.join(self.sample_path, f'score{score:.2f}_{x_type}_{file_name}_gt.jpg'), gt_img)

    def train_dataloader(self):
        if self.train_list:
            self.train_datasets = ListDataset(self.args.dataset_path, self.train_data_transforms, self.train_list)
        else:
            if os.path.isfile(os.path.join(self.args.dataset_path, f'train_{self.args.category}.csv')):
                self.train_datasets = FilelistDataset(root=self.args.dataset_path,
                                                transform=self.train_data_transforms,
                                                phase='train',
                                                category=self.args.category)
            else:
                self.train_datasets = MVTecDataset(root=os.path.join(self.args.dataset_path,self.args.category), transform=self.train_data_transforms, gt_transform=self.gt_transforms, phase='train')
        if self.args.max_train_imgs and self.args.max_train_imgs<len(self.train_datasets):
            self.train_datasets = torch.utils.data.random_split(
                self.train_datasets,
                [self.args.max_train_imgs, len(self.train_datasets)-self.args.max_train_imgs],
                generator=torch.Generator().manual_seed(self.args.seed))[0]
        train_loader = DataLoader(self.train_datasets, batch_size=self.args.batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        if self.args.test_on_train_data:
            if self.train_list:
                self.test_datasets = syntheticListDataset(self.args.dataset_path, self.data_transforms,
                                                          self.gt_transforms, self.train_list, self.args)
            else:
                if os.path.isfile(os.path.join(self.args.dataset_path, f'train_{self.args.category}.csv')):
                    self.test_datasets = FilelistDataset(root=self.args.dataset_path,
                                                         transform=self.data_transforms,
                                                         phase='train',
                                                         category=self.args.category)
                else:
                    self.test_datasets = MVTecDataset(root=os.path.join(self.args.dataset_path, self.args.category),
                                                      transform=self.data_transforms, gt_transform=self.gt_transforms,
                                                      phase='train')
        else:
            if self.test_list:
                self.test_datasets = syntheticListDataset(self.args.dataset_path, self.data_transforms, self.gt_transforms, self.test_list,self.args)
            else:
                if os.path.isfile(os.path.join(self.args.dataset_path, f'test_{self.args.category}.csv')):
                    self.test_datasets = FilelistDataset(root=self.args.dataset_path,
                                                    transform=self.data_transforms,
                                                    phase='test',
                                                    category=self.args.category)
                else:
                    self.test_datasets = MVTecDataset(root=os.path.join(self.args.dataset_path,self.args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        if self.args.max_test_imgs and self.args.max_test_imgs<len(self.test_datasets):
            self.test_datasets = torch.utils.data.random_split(
                self.test_datasets,
                [self.args.max_test_imgs, len(self.test_datasets)-self.args.max_test_imgs],
                generator=torch.Generator().manual_seed(self.args.seed))[0]
        test_loader = DataLoader(self.test_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def predict_dataloader(self):
        predict_dataset = FolderDataset(self.args.dataset_path,self.data_transforms)
        predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers=0)  # , pin_memory=True) # only work on batch_size=1, now.
        return predict_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.output_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        if self.args.saved_model_path:
            self.output_path = self.args.saved_model_path
        os.makedirs(self.output_path,exist_ok=True)
        with open(os.path.join(self.logger.log_dir,'run.txt'),'w') as f:
            f.write(str(self.args))
            f.write('\n')


    # should be implemented
    @abstractmethod
    def training_step(self, batch, batch_idx): # save locally aware patch features
        pass

    def training_epoch_end(self, outputs):
        pass

    def on_test_start(self):
        self.init_results_list()
        self.output_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        if self.args.saved_model_path:
            self.output_path = self.args.saved_model_path

    @abstractmethod
    def predict_scores(self, x):
    # Should be implemented by the spesific algo for computing test time score prediction
    # returns per pixel scores (B,H,W) and image scores (B,) (numpy arrays)
    # example with random scores:
        # image_scores = np.random.uniform(0, 1, (x.shape[0],)) # single value per image
        # pixel_scores = np.repeat(np.repeat(image_scores[:,None,None], x.shape[2], axis=1), x.shape[3], axis=2) # score per pixel
        # return image_scores, pixel_scores
        pass



    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        x, gt, label, file_name, x_type, img_path = batch
        score, anomaly_map = self.predict_scores(x)
        score = score[0] # assuming test_batch_size==1
        anomaly_map = anomaly_map[0] # test_batch_size==1

        if len(gt.shape) == 4:
            gt_np = (gt.cpu().numpy()[0,0]>0).astype(int)
            self.gt_list_px_lvl.extend(gt_np.ravel())
        else:# either good image or no pixel level GT
            gt_np = np.zeros((self.args.input_size, self.args.input_size))
            if gt[0]==-1: #no pixel level GT
                self.gt_list_px_lvl.extend([0])
            else:
                self.gt_list_px_lvl.extend(gt_np.ravel())

        anomaly_map_resized = cv2.resize(anomaly_map, (self.args.input_size, self.args.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)

        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())

        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # save images
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
        if not self.args.dont_save_images:
            self.save_anomaly_map(anomaly_map_resized_blur, input_x, gt_np*255, file_name[0], x_type[0], score)

    def test_epoch_end(self, outputs):
        pixel_auc = -1
        # if not self.args.no_pix_level_auc_roc and  any(self.gt_list_px_lvl):
        if any(self.gt_list_px_lvl):
            print("Total pixel-level auc-roc score :")
            pixel_auc = roc_auc_score(self.gt_list_px_lvl, self.pred_list_px_lvl)
            print(pixel_auc)
        if any(self.gt_list_img_lvl):
            print("Total image-level auc-roc score :")
            img_auc = roc_auc_score(self.gt_list_img_lvl, self.pred_list_img_lvl)
            print(img_auc)
        else: #relevant for automatic selection of augmentation
            print("Maximal anomaly score:")
            img_auc = max(self.pred_list_img_lvl)
            print(img_auc)
        print('test_epoch_end')
        self.values = {'pixel_auc': pixel_auc, 'img_auc': img_auc}
        self.log_dict(self.values)
        with open(os.path.join(self.logger.log_dir,'run.txt'),'a') as f:
            f.write(str(self.values))

