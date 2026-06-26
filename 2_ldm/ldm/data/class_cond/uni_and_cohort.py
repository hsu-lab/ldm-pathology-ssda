import os
import io
import lmdb
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random
from glob import glob
import torchvision.transforms as T

class SamplingDataset(Dataset): 
    
    def __init__(self, config):
        self.image_root = config.get('image_root', None)
        self.fold_index = config.get('fold_index', None)
        self.target_n_csv = config.get('target_n_csv', None)

        # filter the rows for this fold 
        fold_name = f"fold{self.fold_index}"
        df = pd.read_csv(self.target_n_csv)
        df = df[df['fold'] == fold_name].copy()
        if df.empty:
            raise ValueError(f"No rows found in CSV for fold '{fold_name}'")
        
        # build conditions 
        conds = []
        for _, row in df.iterrows():
            n = int(row['n_to_generate'])
            if n <= 0:
                continue
            subtype_id = int(row['subtype_id'])
            subtype_text = str(row['subtype'])
            ## TODO: DOUBLE CHECK THIS! 
            ## ----- pre-train on NLST, fine-tune on NLST -----
            # cohort = 'nlst' 
            # cohort_id = 0
            ## ----- pre-train on NLST and TCGA, fine-tune on NLST -----
            cohort = row['cohort']
            cohort_id = int(row['cohort_id'])
            ## -------------------------------------------------------
            for _ in range(n):
                conds.append({
                    'fold': self.fold_index,
                    "cohort_id": cohort_id,
                    'cohort': cohort,
                    "subtype_id": subtype_id,
                    "subtype": subtype_text,
                    'human_label': f'{subtype_text}_{cohort}',
                })
        if len(conds) == 0:
            raise ValueError(f"All n_to_generate are zero for fold '{fold_name}' — nothing to sample.")

        self.conditions = conds
        print(f"[SamplingDataset] fold={fold_name} | total conditions={len(self.conditions)}")
        
        print(f"Loaded {len(self.conditions)} conditions")
        print("Done!")
    
    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, idx):
        # returns a dict containing conditions 
        return self.conditions[idx]
    

class CohortTissueUniDataset(Dataset):
    """
    Dataset for loading image tiles, UNI features (optional), and cohort/preservation labels from LMDB.
    """

    def __init__(self, config):
        self.split = config.get("split", "train")
        print(f"Initializing {self.split} dataset...")
        self.image_lmdb_path = config["image_lmdb_path"]
        self.feature_lmdb_path = config.get("feature_lmdb_path", None)  
        self.csv_path = config["csv_path"]
        self.cohort = config.get("cohort", "both")  # 'nlst', 'tcga', or 'both'
        
        self.env_img = None
        self.env_feat = None  # only initialized if path is given

        self.metadata = pd.read_csv(self.csv_path)

        if self.cohort in ['nlst', 'tcga']:
            orig_len = len(self.metadata)
            self.metadata = self.metadata[self.metadata['cohort'] == self.cohort].reset_index(drop=True)
            print(f"Filtered dataset to cohort='{self.cohort}', resulting in {len(self.metadata)} samples from {orig_len} samples.")

        # self.metadata = self.metadata[:50]
        self.keys = self.metadata['key'].tolist()
        
        self.crop_size = config.get("crop_size", None)
        self.resize = config.get("resize", None)
        self.p_uncond = config.get("p_uncond", 0.0)

        self.use_features = self.feature_lmdb_path is not None

        print(f"Loaded {len(self.keys)} samples")
        print(f"Drop conditions probability p_uncond = {self.p_uncond}")
        print(f"Using UNI features: {self.use_features}")
        print("Done!")

    def _init_lmdb(self):
        if self.env_img is None:
            self.env_img = lmdb.open(self.image_lmdb_path, readonly=True, lock=False, readahead=False)
        if self.use_features and self.env_feat is None:
            self.env_feat = lmdb.open(self.feature_lmdb_path, readonly=True, lock=False, readahead=False)

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def get_random_crop(img, size):
        if img.shape[0] == size and img.shape[1] == size:
            return img
        if img.shape[0] < size or img.shape[1] < size:
            raise ValueError(f"Image dimensions {img.shape} are smaller than the crop size {size}.")
        x = np.random.randint(0, img.shape[1] - size)
        y = np.random.randint(0, img.shape[0] - size)
        return img[y : y + size, x : x + size]

    def __getitem__(self, idx):
        self._init_lmdb() 

        key = self.keys[idx]

        # Load image
        with self.env_img.begin() as txn:
            img_bytes = txn.get(key.encode())

        image = Image.open(io.BytesIO(img_bytes))

        if self.resize:
            image = image.resize((self.resize, self.resize), Image.BICUBIC)

        image = np.array(image, dtype=np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)  # normalize to [-1, 1]

        if self.split == "train" and self.crop_size:
            image = self.get_random_crop(image, self.crop_size)
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=0).copy()
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()

        # Load UNI feature (only if available)
        uni_feature = None
        if self.use_features:
            with self.env_feat.begin() as txn:
                feat_bytes = txn.get(key.encode())
            uni_feature = np.frombuffer(feat_bytes, dtype=np.float32).copy()

        # Metadata/labels
        meta_row = self.metadata.iloc[idx]
        # cohort_id = int(meta_row['cohort_id']) - 1 ## MAY NEED TO UNCOMMENT THIS!!
        cohort_id = 0 # for NLST only or TCGA only, use 0 for that cohort and 1 for unknown. 
        preservation_method_id = 0
        # preservation_method_id = int(meta_row['preservation_method_id'])
        pid = str(meta_row['pid'])
        slide_id = str(meta_row['slide_id'])
        cohort = str(meta_row['cohort'])
        preservation_method = str(meta_row['preservation_method'])
        human_label = f"cohort={cohort}. tissue_prep={preservation_method}"

        if self.split == "train" and np.random.rand() < self.p_uncond:
            cohort_id = 2 ## NEED TO CHANGE THIS!!! if single cohort, change this to 1. if using nlst and tcga, change this to 2. 
            cohort = 'unconditional'
            preservation_method_id = 1 ## NEED TO CHANGE THIS!!!
            human_label = "unconditional"
            if self.use_features and uni_feature is not None:
                uni_feature = np.zeros_like(uni_feature)

        # Check for NaNs
        if np.isnan(image).any():
            raise RuntimeError(f"NaN detected in image for key {key}")
        if self.use_features and uni_feature is not None and np.isnan(uni_feature).any():
            raise RuntimeError(f"NaN detected in uni_feature for key {key}")
        if np.isnan(cohort_id):
            raise RuntimeError(f"NaN detected in cohort_id for key {key}")
        if np.isnan(preservation_method_id):
            raise RuntimeError(f"NaN detected in preservation_method_id for key {key}")

        out = {
            'image': image.astype(np.float32),
            'tile_name': key,
            'cohort_id': cohort_id,
            'preservation_method_id': preservation_method_id,
            'pid': pid,
            'slide_id': slide_id,
            'cohort': cohort,
            'preservation_method': preservation_method,
            'human_label': human_label,
            # 'subtype_id': 6, 
        }

        if self.use_features and uni_feature is not None:
            out['feature'] = uni_feature.astype(np.float32)
            out['human_label'] += f". feature_dim={uni_feature.shape[0]}"

        return out

