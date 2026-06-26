import os
import lmdb
import io
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import re
from trident.patch_encoder_models.load import encoder_factory

class SyntheticImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted([
            f for f in os.listdir(image_dir) if f.endswith('.png')
        ])
        print(f"Found {len(self.image_files)} synthetic images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, fname)
        image = Image.open(img_path).convert("RGB")

        parts = fname.replace(".png", "").split("_")
        # parts = ['sample', '0000123', tile_name_parts..., cohort, preservation]
        if len(parts) < 5:
            raise ValueError(f"Unexpected filename format: {fname}")

        sample_prefix = parts[0]
        sample_index = parts[1]
        cohort = parts[-2]
        preservation = parts[-1]
        tile_name = "_".join(parts[2:-2])

        return fname, image, tile_name, cohort, preservation

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        encoder = encoder_factory("conch_v15").to(self.device).eval()
        self.model = encoder 
        self.transform = encoder.eval_transforms

    def extract(self, image):
        with torch.inference_mode():
            x = self.transform(image).unsqueeze(0).to(self.device)
            feat = self.model(x).squeeze().cpu().numpy() 
        return feat

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic_dir', type=str, required=True, help='Path to synthetic image folder')
    parser.add_argument('--output_lmdb', type=str, required=True, help='Path to output LMDB for features')
    args = parser.parse_args()

    dataset = SyntheticImageDataset(args.synthetic_dir)
    extractor = FeatureExtractor()

    os.makedirs(os.path.dirname(args.output_lmdb), exist_ok=True)


    print(f"Extracting features and writing to LMDB: {args.output_lmdb}")
    env_out = lmdb.open(args.output_lmdb, map_size=1 << 40)

    with env_out.begin(write=True) as txn:
        for idx in tqdm(range(len(dataset))):
            fname, img, tile_name, cohort, preservation = dataset[idx]
            try:
                feat = extractor.extract(img)
                txn.put(tile_name.encode(), feat.astype(np.float32).tobytes())
            except Exception as e:
                print(f"Skipping {fname} due to error: {e}")

    env_out.close()
    print("Feature extraction complete.")
