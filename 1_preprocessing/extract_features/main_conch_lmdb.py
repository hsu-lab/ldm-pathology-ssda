# extract uni features from lmdb datasets 
"""
CUDA_VISIBLE_DEVICES=4 python main_conch_lmdb.py \
--input_lmdb /workspace/hsuraid/tengyuezhang/diffusion_luad/data_nlst/cond_ablation/cascaded_cross_lmdb/syn_img_lmdb \
--input_csv /workspace/hsuraid/tengyuezhang/diffusion_luad/data/dev_lmdb_25_tissue_split_by_pid/final_samplaed_10k_val_conditions.csv \
--output_lmdb /workspace/hsuraid/tengyuezhang/diffusion_luad/data_nlst/cond_ablation/cascaded_cross_conch
"""

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
from trident.patch_encoder_models.load import encoder_factory


class LMDBTileDataset(Dataset):
    def __init__(self, lmdb_path, csv_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False)
        self.metadata = pd.read_csv(csv_path)
        self.keys = self.metadata['key'].tolist()
        print(f"Loaded {len(self.keys)} keys from {lmdb_path}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin() as txn:
            img_bytes = txn.get(key.encode())
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return key, image

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        encoder = encoder_factory('conch_v15', weights_path="/hsuraid/tengyuezhang/tools/trident/hf_weights/conchv1_5/pytorch_model_vision.bin").to(self.device).eval()
        self.model = encoder 
        self.transform = encoder.eval_transforms

    def extract(self, image):
        with torch.inference_mode():
            x = self.transform(image).unsqueeze(0).to(self.device)
            feat = self.model(x).squeeze().cpu().numpy() 
        return feat

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_lmdb', type=str, required=True, help='Path to input LMDB of image tiles')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file of tile conditions')
    parser.add_argument('--output_lmdb', type=str, required=True, help='Path to output LMDB for features')
    args = parser.parse_args()

    print('Creating dataset...')
    dataset = LMDBTileDataset(args.input_lmdb, args.input_csv)
    print('Dataset ready!')

    extractor = FeatureExtractor()

    print(f"Extracting features and writing to LMDB: {args.output_lmdb}")
    env_out = lmdb.open(args.output_lmdb, map_size=1 << 40)
    with env_out.begin(write=True) as txn:
        for idx in tqdm(range(len(dataset))):
            key, img = dataset[idx]
            try:
                feat = extractor.extract(img)
                txn.put(key.encode(), feat.astype(np.float32).tobytes())
            except Exception as e:
                print(f"Skipping key {key} due to error: {e}")
    env_out.close()
    print("Feature extraction complete.")

