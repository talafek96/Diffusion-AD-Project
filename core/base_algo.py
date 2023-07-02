import os
import shutil
from typing import Tuple
import torch
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from typing import Set
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from utils.dataset import MVTecDataset, FilelistDataset, ListDataset, FolderDataset
from utils.data_frame_manager import PersistentDataFrame
from utils.transforms import get_transforms
from config.configuration import MAGIC_NORMALIZE_MEAN, MAGIC_NORMALIZE_STD, CATEGORY_TO_V_MIN_MAX, DEFAULT_AUGMENT_NAME, \
    UNLIMITED_MAX_TEST_IMAGES, CATEGORY_TO_TYPE, DEFAULT_RESULTS_COLUMNS


ALL_CATEGORIES = set(CATEGORY_TO_TYPE.keys())  # since keys() returns a view and not a set


def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name) and full_file_name.endswith('.py'):
            shutil.copy(full_file_name, os.path.join(dst, file_name))
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
    copy_files('./', source_code_save_path, ['.git', '.vscode', '__pycache__',
               'logs', 'README', 'samples', 'LICENSE'])  # copy source code
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
    """
    A basic abstract class for training and testing an
    anomaly detection model.

    Attributes:
    -----------
    hparams: (dict-like parameter)
    - phase:
        Can be 'test' or 'train', we will probably only use 'test'.
    - category:
        The folder name from the data set. ('bottle', 'hazelnut', ...)
    - coreset_sampling_ratio: 
        Used for batch selection while training. (NOT NEEDED FOR TEST)
    - dataset_path:
        The path to the dataset. (category will be concat)
    - num_epochs:
        Number of training epochs, used by pl.Trainer,
        default=1. Not relevant for test.
    - batch_size:
        Used when initializing the data loaders.
        '1' for test, batch_size for train.
    - load_size:
        IMPORTANT, The size of the image in pixels, used when preparing the 
        transforms used: data_transforms, gt_transforms. default=256
    - input_size:
        Used for the center crop transform. default=256 (256x256 center crop)
    - root_output_dir:
        Used when saving the results like the anomaly maps.  (default='./test')
        scores, etc.
        Required for initializing the Trainer.
        The trainer will export output files into `${root_output_dir}/${category}`
    - save_anomaly_map:
        A boolean flag, if True saves the heatmaps and scores.
    """

    def __init__(self, hparams, model_name: str = "Unknown model name"):
        super(BaseAlgo, self).__init__()
        self.args = hparams
        
        if 'mean_train' not in self.args:
            self.args.mean_train = MAGIC_NORMALIZE_MEAN
        if 'std_train' not in self.args:
            self.args.std_train = MAGIC_NORMALIZE_STD
        if 'augment' not in self.args:
            self.args.augment = DEFAULT_AUGMENT_NAME
        if 'test_on_train_data' not in self.args:
            self.args.test_on_train_data = False
        if 'max_test_imgs' not in self.args:
            self.args.max_test_imgs = UNLIMITED_MAX_TEST_IMAGES
        
        self.experiment_results_manager = PersistentDataFrame(self.args.results_csv_path, columns=DEFAULT_RESULTS_COLUMNS)
        self.model_name = model_name

        self.save_hyperparameters(hparams)
        self.init_results_list()

        self.train_data_transforms, self.data_transforms, self.gt_transforms = get_transforms(self.args)

        self.inv_normalize = transforms.Normalize(mean=(-self.args.mean_train / self.args.std_train), std=(1 / self.args.std_train))

        self.train_list = []

        self.test_image_folder = ''

    def init_results_list(self):
        """
        Initializing the datamembers used as output.
        """
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type, score):
        """
        Saving the input image, the anomaly map, the AM on the IMG, 
        with an option of also saving the Ground Truth.
        """
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(
                anomaly_map, (input_img.shape[0], input_img.shape[1]))

        anomaly_map_hm = cvt2heatmap(anomaly_map*255)

        # Calculate anomaly map on image
        hm_on_img = heatmap_on_image(anomaly_map_hm, input_img)

        # Save images
        score = float(score)
        cv2.imwrite(os.path.join(self.sample_path,
                    f'{self.args.category}_{x_type}_score{score:.2f}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(
            self.sample_path, f'{self.args.category}_{x_type}_score{score:.2f}_{file_name}_amap.jpg'), anomaly_map_hm)
        cv2.imwrite(os.path.join(
            self.sample_path, f'{self.args.category}_{x_type}_score{score:.2f}_{file_name}_amap_on_img.jpg'), hm_on_img)
        if gt_img is not None:
            cv2.imwrite(os.path.join(
                self.sample_path, f'{self.args.category}_{x_type}_score{score:.2f}_{file_name}_gt.jpg'), gt_img)


    def train_dataloader(self):
        """
        Basically a method for getting the dataloader for the training phase
        """
        if self.train_list:
            self.train_datasets = ListDataset(
                self.args.dataset_path, self.train_data_transforms, self.train_list)
        else:
            if os.path.isfile(os.path.join(self.args.dataset_path, f'train_{self.args.category}.csv')):
                self.train_datasets = FilelistDataset(root=self.args.dataset_path,
                                                      transform=self.train_data_transforms,
                                                      phase='train',
                                                      category=self.args.category)
            else:
                self.train_datasets = MVTecDataset(root=os.path.join(
                    self.args.dataset_path, self.args.category), transform=self.train_data_transforms, gt_transform=self.gt_transforms, phase='train')
        if self.args.max_train_imgs and self.args.max_train_imgs < len(self.train_datasets):
            self.train_datasets = torch.utils.data.random_split(
                self.train_datasets,
                [self.args.max_train_imgs, len(
                    self.train_datasets)-self.args.max_train_imgs],
                generator=torch.Generator().manual_seed(self.args.seed))[0]
        train_loader = DataLoader(self.train_datasets, batch_size=self.args.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        """
        Basically a method for getting the dataloader for the test phase
        """
        if self.args.test_on_train_data:
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
            if os.path.isfile(os.path.join(self.args.dataset_path, f'test_{self.args.category}.csv')):
                self.test_datasets = FilelistDataset(root=self.args.dataset_path,
                                                        transform=self.data_transforms,
                                                        phase='test',
                                                        category=self.args.category)
            else:
                self.test_datasets = MVTecDataset(root=os.path.join(
                    self.args.dataset_path, self.args.category), transform=self.data_transforms, gt_transform=self.gt_transforms, phase='test')
        if self.args.max_test_imgs and self.args.max_test_imgs < len(self.test_datasets):
            self.test_datasets = torch.utils.data.random_split(
                self.test_datasets,
                [self.args.max_test_imgs, len(
                    self.test_datasets)-self.args.max_test_imgs],
                generator=torch.Generator().manual_seed(self.args.seed))[0]

        test_loader = DataLoader(
            self.test_datasets, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        return test_loader

    def predict_dataloader(self):
        """
        Most likely irrelevant, but left untouched for future generations.
        """
        predict_dataset = FolderDataset(
            self.args.dataset_path, self.data_transforms)

        predict_loader = DataLoader(
            predict_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        return predict_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        """
        Invoked right before training phase and only once per training epoch.
        """
        self.output_path, self.sample_path, self.source_code_save_path = prep_dirs(
            self.logger.log_dir)
        if self.args.saved_model_path:
            self.output_path = self.args.saved_model_path
        os.makedirs(self.output_path, exist_ok=True)
        with open(os.path.join(self.logger.log_dir, 'run.txt'), 'w') as f:
            f.write(str(self.args))
            f.write('\n')

    def training_step(self, batch, batch_idx):  # save locally aware patch features
        pass

    def training_epoch_end(self, outputs):
        pass

    def on_test_start(self):
        """
        Invoked right before test phase and only once per test epoch (there is only one test epoch).
        """
        self.init_results_list()
        self.output_path, self.sample_path, self.source_code_save_path = prep_dirs(
            self.logger.log_dir)
        if self.args.saved_model_path:
            self.output_path = self.args.saved_model_path

    @abstractmethod
    def predict_scores(self, x):
        '''
        Should be implemented by the spesific algo for computing test time score prediction.
        Returns per pixel scores (B,H,W) and image scores (B,) (numpy arrays).

        Example with random scores:
        >>> image_scores = np.random.uniform(0, 1, (x.shape[0],)) # single value per image
        >>> pixel_scores = np.repeat(np.repeat(image_scores[:,None,None], x.shape[2], axis=1), x.shape[3], axis=2) # score per pixel
        >>> return image_scores, pixel_scores
        '''
        pass


    def test_step(self, batch, batch_idx):
        """
        The Business Logic (BL) of every test step.
        we will eventually call predict_scores from here for every test image in the test set.
        """
        x, gt, label, file_name, x_type, img_path = batch
        anomaly_map, score = self.predict_scores(x)
        score = score[0]  # assuming test_batch_size==1

        if self.args.verbosity >= 1:
            print(f'Anomaly Map values:\nvmin={anomaly_map.min()}, vmax={anomaly_map.max()}')

        if len(gt.shape) == 4:
            gt_np = (gt.cpu().numpy()[0, 0] > 0).astype(int)
            self.gt_list_px_lvl.extend(gt_np.ravel())
        else:  # either good image or no pixel level GT
            gt_np = np.zeros((self.args.input_size, self.args.input_size))
            if gt[0] == -1:  # no pixel level GT
                self.gt_list_px_lvl.extend([0])
            else:
                self.gt_list_px_lvl.extend(gt_np.ravel())

        # Resize and blur for noise reduction
        anomaly_map_resized = cv2.resize(
            anomaly_map, (self.args.input_size, self.args.input_size))
        anomaly_map_resized_blur = gaussian_filter(
            anomaly_map_resized, sigma=4)
        anomaly_map_resized_blur = anomaly_map_resized  # TODO: Check if blurring the anomaly map works better or worse

        # Normalize anomaly map
        vmin, vmax = CATEGORY_TO_V_MIN_MAX[self.args.category]
        anomaly_map = (anomaly_map - vmin) / (vmax - vmin)
        anomaly_map = anomaly_map.clip(0, 1)

        self.pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())

        self.gt_list_img_lvl.append(label.cpu().numpy()[0])
        self.pred_list_img_lvl.append(score)
        self.img_path_list.extend(file_name)
        # save images
        x = self.inv_normalize(x)
        input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
        if self.args.save_anomaly_map:
            self.save_anomaly_map(
                anomaly_map, input_x, gt_np*255, file_name[0], x_type[0], score)

    def test_epoch_end(self, outputs):
        """
        Callback to be called at the end of a test epoch (there should only be one epoch)
        """
        pixel_auc = -1
        # if not self.args.no_pix_level_auc_roc and  any(self.gt_list_px_lvl):
        if any(self.gt_list_px_lvl):
            print("Total pixel-level auc-roc score :")
            pixel_auc = roc_auc_score(
                self.gt_list_px_lvl, self.pred_list_px_lvl)
            print(pixel_auc)
        if any(self.gt_list_img_lvl):
            print("Total image-level auc-roc score :")
            img_auc = roc_auc_score(
                self.gt_list_img_lvl, self.pred_list_img_lvl)
            print(img_auc)
        else:  # relevant for automatic selection of augmentation
            print("Maximal anomaly score:")
            img_auc = max(self.pred_list_img_lvl)
            print(img_auc)
            
        print('test_epoch_end')
        self.values = {'pixel_auc': float(pixel_auc), 'img_auc': float(img_auc)}
        self.log_dict(self.values)

        self._update_results_csv(self.values)
        
        with open(os.path.join(self.logger.log_dir, 'run.txt'), 'a') as f:
            f.write(str(self.values))

    def get_remaining_categories(self) -> Set:
        """
        Getting a set of the remaining categories that do not yet exist 
        in the experiment data file.

        Return:
        -------
        A set of strings denoting the categories who don't exist in the file
        """
        categories_in_file = self._get_categories_in_data_file()

        return ALL_CATEGORIES - categories_in_file

    def _update_results_csv(self, values_dict) -> None:
        key = [self.args.category, self.model_name]
        if self.experiment_results_manager.data[['category', 'model_name']].isin(key).any(axis=1).any():
            # Key match found, overwriting
            mask = (self.experiment_results_manager.data[['category', 'model_name']] != key)
            self.experiment_results_manager.data = self.experiment_results_manager.data[mask]
        
        new_dict = values_dict.copy()
        new_dict['category'] = self.args.category
        new_dict['category_type'] = CATEGORY_TO_TYPE[self.args.category]
        new_dict['model_name'] = self.model_name
        self.experiment_results_manager.data = \
            pd.concat(
                [self.experiment_results_manager.data, pd.DataFrame(new_dict, index=[0])]
            ).reset_index(drop=True).sort_values(['model_name', 'category_type', 'category'])

    def _get_categories_in_data_file(self) -> Set:
        """
        Getting a list of the categories from experiment_results_manager that correspond
        to the `self.model_name` model.

        Return: Set[str]
        -------
        A set of strings denoting the categories stored in the data file.
        """
        categories = set(
            self.experiment_results_manager.data.loc[self.experiment_results_manager.data["model_name"] == self.model_name, "category"].unique())
        
        return categories
