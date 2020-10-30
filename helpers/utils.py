import numpy as np
from torch.utils.data import Dataset
from skimage import io, transform
import os
import pandas as pd
from ast import literal_eval
from sklearn import preprocessing
import torchvision.transforms as transforms
import torch


class MeshImageDataset(Dataset):
    """
    input: a csv file contains all locs to input to the model,
    iteration: return the center image at the lat_lo
    return: image_mat, labels, lat_lon
    """

    def __init__(self, x, y, img_dir_path, normalize=False):
        # x is the mesh code
        self.x = x
        self.y = y
        self.img_dir_path = img_dir_path
        self.norm = normalize

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir_path,
                                self.x[idx] + '.png')
        # img_path = os.path.join(self.img_dir_path,
        #                         self.x[idx] + '.png')
        image = io.imread(img_path)[:, :, :3]
        compose = transforms.Compose([transforms.ToPILImage()])
        image = np.array(compose(image))
        image = image / 255
        img_mat = np.array(image).transpose((2, 0, 1))
        if self.norm:
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_mat = norm(torch.FloatTensor(img_mat))

        return img_mat, np.array(self.y[idx], dtype=np.float32), self.x[idx]

    def __len__(self):
        return len(self.x)


class BikeImageDataset(Dataset):
    """
    input: a csv file contains all locs to input to the model,
    iteration: return the center image at the lat_lon
    return: image_mat, labels, lat_lon

    """

    def __init__(self, x, y, img_dir_path, normalize=False):
        # x is the lat_lon
        self.x = x
        self.y = y
        self.img_dir_path = img_dir_path
        self.norm = normalize

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir_path,
                                self.x[idx] + '.png')
        image = io.imread(img_path)[:, :, :3]
        compose = transforms.Compose([transforms.ToPILImage()])
        image = np.array(compose(image))
        image = image / 255
        img_mat = np.array(image).transpose((2, 0, 1))
        if self.norm:
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_mat = norm(torch.FloatTensor(img_mat))
        return img_mat, np.array(self.y[idx], dtype=np.float32), self.x[idx]

    def __len__(self):
        return len(self.x)


def csv_to_x_y(df):
    """
        Preprocessing the csv file with meshcode and labels, convert to x, y
    :param df:
    :return:
    """
    x = df.mesh.map(int).map(str).values.tolist()
    y = df.drop(['mesh'], axis=1).values.tolist()
    return x, y


def bikecsv_to_x_y(df):
    """
        Preprocessing the csv file with meshcode and labels, convert to x, y
    :param df:
    :return:
    """
    x = df.lat_lon.values.tolist()
    y = df.drop(['lat_lon'], axis=1).values.tolist()
    return x, y
