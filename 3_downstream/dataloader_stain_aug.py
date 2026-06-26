# dataloader.py
# https://github.com/CielAl/torch-staintools 
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class DataProcessor(Dataset):
    def __init__(
        self,
        imgs_dir,
        channel,
        split,
        image_processor,
        num_image_channels=3,
        augmentor=None,
        syn_root=None,
    ):
        """
        Args:
            imgs_dir (list[str]): list of image file paths
            channel (int): number of output channels (3, 6, or 12)
            split (str): 'train', 'val', or 'test'
            image_processor: HF AutoImageProcessor (for size/mean/std)
            num_image_channels (int): usually 3 for ViT
            augmentor: torch-staintools augmentor (train only), or None
            syn_root (str): path prefix for synthetic images (optional)
        """
        self.imgs_ids = imgs_dir
        self.channel = channel
        self.split = split
        self.augmentor = augmentor      # CPU augmentor
        self.syn_root = os.path.abspath(syn_root) if syn_root else None
        self.num_image_channels = num_image_channels

        # --- figure out size / mean / std from image_processor ---
        size = 224
        if hasattr(image_processor, "size"):
            s = image_processor.size
            if isinstance(s, dict):
                size = s.get("height", s.get("shortest_edge", 224))
            elif isinstance(s, int):
                size = s
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])[:self.num_image_channels]
        std  = getattr(image_processor, "image_std",  [0.5, 0.5, 0.5])[:self.num_image_channels]

        # --- basic transforms ---
        self.resize = T.Resize((size, size))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=mean, std=std)

        random.seed(42)
        if split == 'train':
            random.shuffle(self.imgs_ids)

    def __len__(self):
        return len(self.imgs_ids)

    def _get_label_from_path(self, img_file: str) -> torch.Tensor:
        if 'good_prognosis' in img_file:
            return torch.tensor(0, dtype=torch.long)  # good prognosis
        elif 'intermediate_prognosis' in img_file:
            return torch.tensor(1, dtype=torch.long)  # intermediate prognosis
        elif 'poor_prognosis' in img_file:
            return torch.tensor(2, dtype=torch.long)  # poor prognosis
        elif 'nontumor' in img_file:
            return torch.tensor(3, dtype=torch.long)  # non-tumor
        else:
            raise ValueError(f"Unknown class in filename: {img_file}")

    def __getitem__(self, i):
        img_file = self.imgs_ids[i]

        # synthetic flag
        if self.syn_root is not None:
            is_syn = os.path.abspath(img_file).startswith(self.syn_root)
        else:
            is_syn = False

        label = self._get_label_from_path(img_file)

        # --- Load & preprocess image (CPU) ---
        img_pil = Image.open(img_file).convert("RGB")
        img_pil = self.resize(img_pil)
        img = self.to_tensor(img_pil)  # [3, H, W], float32 in [0,1]

        # --- Stain augmentation on CPU (train only) ---
        if (self.split == 'train') and (self.augmentor is not None):
            # augmentor expects BCHW on CPU
            img_bchw = img.unsqueeze(0).contiguous()  # [1, C, H, W]
            with torch.no_grad():
                # you can use cache_keys=[i] or [img_file]
                img_bchw = self.augmentor(img_bchw, cache_keys=[img_file])
            img = img_bchw.squeeze(0)

        # --- Normalize ---
        img = self.normalize(img)

        # --- Expand channels if needed ---
        if self.channel == 6:
            img = torch.cat((img, img), dim=0)
        elif self.channel == 12:
            img = torch.cat((img, img, img, img), dim=0)

        return {
            "image": img,        # CPU tensor
            "label": label,
            "is_syn": torch.tensor(is_syn, dtype=torch.float32),
            "img_path": img_file,
        }