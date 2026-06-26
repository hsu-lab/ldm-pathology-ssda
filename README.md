# Semi-Supervised Domain Adaptation with Latent Diffusion for Pathology Image Classification

Deep learning models in computational pathology often fail to generalize across cohorts and institutions due to domain shift. Existing approaches either fail to leverage unlabeled data from the target domain or rely on image-to-image translation, which can distort tissue structures and compromise model accuracy. In this work, we propose a semi-supervised domain adaptation (SSDA) framework that utilizes a latent diffusion model trained on unlabeled data from both the source and target domains to generate morphology-preserving and target-aware synthetic images. By conditioning the diffusion model on foundation model features, cohort identity, and tissue preparation method, we preserve tissue structure in the source domain while introducing target-domain appearance characteristics. The target-aware synthetic images, combined with real, labeled images from the source cohort, are subsequently used to train a downstream classifier, which is then tested on the target cohort. The effectiveness of the proposed SSDA framework is demonstrated on the task of lung adenocarcinoma subtype classification. The proposed augmentation yielded substantially better performance on the held-out test set from the target cohort, without degrading source-cohort performance. The approach improved the weighted F1 score on the target-cohort held-out test set from 0.611 to 0.706 and the macro F1 score from 0.641 to 0.716. Our results demonstrate that target-aware diffusion-based synthetic data augmentation provides a promising and effective approach for improving domain generalization in computational pathology.

[pre-print] https://arxiv.org/abs/2601.17228 

# LDM training

We adapted the implementation of the latent diffusion model from [PathLDM](https://github.com/cvlab-stonybrook/PathLDM). 

To Train the VAE and the U-Net denoiser, create the conda environment by: 
```
cd 2_ldm 
conda env create -f environment_olivia.yaml
conda activate ldm
```

For VAE training, use: 
```
CUDA_VISIBLE_DEVICES=0,1,3 python main.py --base configs/autoencoder/vq_f4.yaml -t --gpus 0,1,2 
```

For LDM training, use:
```
python main.py -t \
--base configs/latent-diffusion/feature_cond/uni_and_others_from_scratch_concat.yaml \
--gpus 0,1
```

# LDM evaluation 

Use the following script to calculate FID statistics:
```
2_ldm/analysis/ldm/calculate_fid_stats.py
```

# Downstream classification 

Install dependencies:
```
cd 3_downstream 
chmod +x install_pkgs.sh 
./install_pkgs.sh
```

Grid search:
```
CUDA_VISIBLE_DEVICES=1 python vit_syn_grid_search_local.py \
--path_to_real_images /path/to/real \
--path_to_syn_images /path/to/synthetic \
--path_to_external_test_images /path/to/external \
--exp_folder 'exp' \
--patience 5 \
--dropout_rate 0.0 \
--learning_rate 1e-4 \
--num_classes 4
```



