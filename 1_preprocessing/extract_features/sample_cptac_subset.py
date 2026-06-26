"""
Generate a subset of CPTAC conditions (keys) from which we will extract features. 
Used in t-SNE analysis.
"""
import pandas as pd 
import os 

df = pd.read_csv('/workspace/hsuraid/tengyuezhang/diffusion_luad/data_nlst_cptac/nlst_cptac_dev_lmdb_25_tissue/val_tiles_conditions.csv')
df = df[df['cohort'] == 'cptac']
df_sampled = df.sample(n=2500, random_state=42)

df_sampled.to_csv('/workspace/hsuraid/tengyuezhang/diffusion_luad/data_nlst_cptac/features/uni/cptac/2500_val/2500_val_samples_conditions.csv', index=False)