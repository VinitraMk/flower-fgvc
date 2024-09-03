from torch.utils.data import Dataset
import os
from scipy.io import loadmat
import torch
from torchvision.io import read_image


class FlowerDataset(Dataset):

    def __init__(self, data_dir, data_filepaths, transforms):
        img_dir = os.path.join(data_dir, 'jpg')
        #seg_mask_dir = os.path.join(data_dir, 'segmim')
        labels_path = os.path.join(data_dir, 'imagelabels.mat')
        labels_mat = loadmat(labels_path)
        #id_path = os.path.join(data_dir, 'setid.mat')
        #ids = loadmat(id_path)
        #print(ids)
        self.data_filepaths = data_filepaths
        self.img_dir = img_dir
        self.labels_tensor = torch.from_numpy(labels_mat['labels'][0]).int() - 1
        self.num_classes = len(self.labels_tensor.unique())
        self.data_transform = transforms
        #print('unique ids', ids['tstid'].min(), ids['tstid'].max())

    def __len__(self):
        return len(self.data_filepaths)

    def __getitem__(self, idx):
        fn = self.data_filepaths[idx]
        si = fn.find("_")
        img_idx = int(fn[si+1:si+6]) - 1
        img_tensor = read_image(os.path.join(self.img_dir, fn)).float()
        img_tensor = self.data_transform(img_tensor)
        label = self.labels_tensor[img_idx]
        onehot_label = torch.zeros(self.num_classes, dtype=torch.float)
        onehot_label[label] = 1
        soft_label = torch.zeros(self.num_classes, dtype=torch.float)

        return {
            'img': img_tensor,
            'label': label,
            'olabel': onehot_label,
            'slabel': soft_label
        }

