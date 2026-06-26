import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import os 

class DataProcessor(Dataset):
    def __init__(self, imgs_dir, channel, split, transform=None, syn_root=None):
        """
        Args:
            imgs_dir (list[str]): list of image file paths
            channel (int): number of output channels (3, 6, or 12 supported)
            split (str): one of 'train', 'val', 'test'
            transform (callable, optional): torchvision-like transform to apply to each image
        """
        self.imgs_ids = imgs_dir
        self.channel = channel
        self.split = split
        self.transform = transform
        self.syn_root = os.path.abspath(syn_root) if syn_root else None

        random.seed(42)
        if split == 'train':
            random.shuffle(self.imgs_ids)  # shuffle only for training

    def __getitem__(self, i):
        img_file = self.imgs_ids[i]

        if self.syn_root is not None:
            is_syn = os.path.abspath(img_file).startswith(self.syn_root)
        else: 
            is_syn = False 

        # Assign labels based on file path keywords
        if 'lepidic' in img_file:
            label = torch.tensor(0) # good prognosis 
        elif 'acinar' in img_file:
            label = torch.tensor(1) # intermediate prognosis 
        elif 'papillary' in img_file:
            label = torch.tensor(1) # intermediate prognosis 
        elif 'micro' in img_file:
            label = torch.tensor(2) # poor prognosis
        elif 'solid' in img_file:
            label = torch.tensor(2) # poor prognosis
        elif 'nontumor' in img_file:
            label = torch.tensor(3) # non-tumor
        else:
            raise ValueError(f"Unknown class in filename: {img_file}")

        # Load image
        img = np.asarray(Image.open(img_file))

        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)
        else:
            # Fallback: convert to tensor without augmentation
            from torchvision import transforms as T
            img = T.ToTensor()(img)

        # Expand channels if needed
        if self.channel == 6:
            img = torch.cat((img, img), dim=0)
        elif self.channel == 12:
            img = torch.cat((img, img, img, img), dim=0)

        out = {
            "image": img, 
            "label": label,
            'is_syn': is_syn,
            'img_path': img_file,
        }

        return out 

    def __len__(self):
        return len(self.imgs_ids)
