#!/usr/bin/env python3
"""
Standalone evaluation script for 5-fold ViT models (no training).
Assumes:
  - Models saved as: <exp_folder>/<exp_name>/saved_models/vit_fold{0..4}.pth
  - Real images: path_to_real_images/fold{0..4}/test/class/*.png
  - Optional external images: path_to_external_test_images/class/*.png
"""

import os, sys, argparse, logging
from glob import glob
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score as sk_f1

from dataloader import DataProcessor
from transformers import AutoConfig, ViTForImageClassification, AutoImageProcessor

# -------------------------
# Config / utilities
# -------------------------
CLASS_LABELS = {
    0: "good_prognosis",
    1: "intermediate_prognosis",
    2: "poor_prognosis",
    3: "non_tumor",
}

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------
# Minimal Trainer (eval only)
# -------------------------
class ViTTrainer:
    def __init__(self,
                 num_classes,
                 batch_size,
                 learning_rate,
                 fold_index,
                 hf_model_name="google/vit-base-patch16-224-in21k",
                 freeze_encoder_layers=0,
                 dropout_rate=0.0,
                 num_image_channels=3):
        self.num_classes = num_classes
        self.batch = batch_size
        self.fold_index = fold_index
        self.learning_rate = learning_rate
        self.hf_model_name = hf_model_name
        self.freeze_encoder_layers = max(0, int(freeze_encoder_layers))
        self.dropout_rate = dropout_rate
        self.num_image_channels = num_image_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.last_avg_f1 = {"weighted": np.nan, "macro": np.nan}

        # Image processor (gives mean/std and size)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.hf_model_name, use_fast=True
        )

    def _make_model(self):
        cfg = AutoConfig.from_pretrained(self.hf_model_name)
        cfg.hidden_dropout_prob = self.dropout_rate
        cfg.attention_probs_dropout_prob = self.dropout_rate
        cfg.classifier_dropout = self.dropout_rate
        cfg.num_labels = self.num_classes

        model = ViTForImageClassification.from_pretrained(
            self.hf_model_name,
            config=cfg,
            ignore_mismatched_sizes=True,
        )

        # Optionally freeze the first K encoder blocks
        if self.freeze_encoder_layers > 0:
            enc = model.vit.encoder
            layers = enc.layer
            K = min(self.freeze_encoder_layers, len(layers))
            logging.info(f"Freezing first {K}/{len(layers)} ViT encoder layers.")
            for i in range(K):
                for p in layers[i].parameters():
                    p.requires_grad = False

            if K == len(layers):
                for p in model.vit.embeddings.parameters():
                    p.requires_grad = False

        return model

    def _build_transforms(self):
        size = 224
        if hasattr(self.image_processor, "size"):
            s = self.image_processor.size
            if isinstance(s, dict):
                size = s.get("height", s.get("shortest_edge", 224))
            elif isinstance(s, int):
                size = s

        mean = getattr(self.image_processor, "image_mean", [0.5, 0.5, 0.5])[:self.num_image_channels]
        std  = getattr(self.image_processor, "image_std",  [0.5, 0.5, 0.5])[:self.num_image_channels]

        tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return tfm

    def _plot_confmat(self, figures_dir, cmatrix, classes, test_name, normalize=True):
        import itertools
        if normalize:
            with np.errstate(invalid='ignore', divide='ignore'):
                cm_plot = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]
                cm_plot = np.nan_to_num(cm_plot)
        else:
            cm_plot = cmatrix

        plt.imshow(cm_plot, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=25)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm_plot.max() / 2.0 if cm_plot.size else 0.5
        for i, j in itertools.product(range(cm_plot.shape[0]), range(cm_plot.shape[1])):
            plt.text(j, i, format(cm_plot[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm_plot[i, j] > thresh else "black")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f'{test_name}_confusion_matrix.png'))
        plt.clf()
    def run_test(self, image_paths, test_name, figure_dir, model_dir, csv_path=None):
        logging.info("="*40)
        logging.info(f"Running test: {test_name}")
        os.makedirs(figure_dir, exist_ok=True)

        tfm = self._build_transforms()
        ds = DataProcessor(imgs_dir=image_paths, channel=NUM_CHANNEL, split='test', transform=tfm)
        logging.info(f"Images for testing: {len(ds)}")
        loader = DataLoader(
            ds,
            batch_size=self.batch,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True
        )

        # Rebuild model and load best weights
        model = self._make_model().to(self.device)
        ckpt_path = os.path.join(model_dir, f'vit_fold{self.fold_index}.pth')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        logging.info(f"Model weights loaded from {ckpt_path}")

        criterion = nn.CrossEntropyLoss()

        y_true, y_pred = [], []
        all_probs = []
        all_paths = []

        running_test = 0.0

        with torch.no_grad():
            tbar = tqdm(loader, desc=f"Test ({test_name})", leave=False)
            for batch in tbar:
                images = batch['image'].to(self.device, dtype=torch.float)
                labels = batch['label'].to(self.device, dtype=torch.long)

                batch_paths = batch['img_path']

                logits = model(images).logits
                loss = criterion(logits, labels)
                running_test += float(loss.item()) * images.size(0)

                probs = F.softmax(logits.detach().cpu(), dim=1).numpy()
                preds = probs.argmax(axis=1)

                y_pred.extend(preds.tolist())
                y_true.extend(labels.cpu().numpy().tolist())
                all_probs.extend(probs.tolist())

                if batch_paths is not None:
                    if isinstance(batch_paths, (list, tuple)):
                        all_paths.extend(list(batch_paths))
                    else:
                        # DataLoader default collate makes list-of-str -> list[str],
                        # so this branch is mostly a safety net.
                        all_paths.extend(batch_paths)

        avg_test = running_test / max(1, len(loader))
        cnf = cm(y_true, y_pred, labels=list(range(self.num_classes)))
        logging.info("Confusion matrix:")
        logging.info(cnf)
        self._plot_confmat(figure_dir, cnf, list(CLASS_LABELS.values()), test_name, normalize=True)

        # Per-class metrics via sklearn
        per_class_f1 = sk_f1(
            y_true, y_pred,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0
        )

        macro_f1 = sk_f1(
            y_true, y_pred,
            labels=list(range(self.num_classes)),
            average='macro',
            zero_division=0
        )
        weighted_f1 = sk_f1(
            y_true, y_pred,
            labels=list(range(self.num_classes)),
            average='weighted',
            zero_division=0
        )
        self.last_avg_f1 = {"weighted": float(weighted_f1), "macro": float(macro_f1)}

        logging.info(f"Test loss: {avg_test:.6f}")
        logging.info(f"Per-class F1: {per_class_f1}")
        logging.info(f"F1 (avg) weighted/macro: {weighted_f1:.4f} / {macro_f1:.4f}")

        # --------------------------------------------------
        # Build prediction dataframe & optionally save to CSV
        # --------------------------------------------------
        n_samples = len(y_true)
        # If we didn't manage to grab paths, fill with empty strings
        if len(all_paths) != n_samples:
            all_paths = [""] * n_samples

        prob_cols = [f"prob_{CLASS_LABELS[i]}" for i in range(self.num_classes)]

        df = pd.DataFrame({
            "image_path": all_paths,
            "true_label": y_true,
            "pred_label": y_pred,
        })
        # add probability columns
        probs_array = np.array(all_probs, dtype=float)
        for i, col in enumerate(prob_cols):
            df[col] = probs_array[:, i]

        if csv_path is not None:
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved predictions to: {csv_path}")

        return per_class_f1

# -------------------------
# main (evaluation only)
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_real_images', type=str, required=True,
                        help='parent path to all the images (expects fold{0..4}/{train,val,test}/class/*.png)')
    parser.add_argument('--path_to_external_test_images', type=str, default=None,
                        help='parent path to images for external validation (class/*.png)')
    parser.add_argument('--num_image_channels', type=int, default=3,
                        help='number of channels (ViT expects 3)')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='should match len(CLASS_LABELS)')
    parser.add_argument('--batches', type=int, default=32)

    parser.add_argument('--exp_folder', type=str, default='exp')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='name of the existing experiment (subfolder under exp_folder)')

    # ViT config (must match training)
    parser.add_argument('--hf_model_name', type=str, default='google/vit-base-patch16-224-in21k')
    parser.add_argument('--freeze_encoder_layers', type=int, default=0)
    parser.add_argument('--dropout_rate', type=float, default=0.0)

    args = parser.parse_args()

    global NUM_CHANNEL
    NUM_CHANNEL = args.num_image_channels
    num_classes = args.num_classes

    # experiment folders + logging
    exp_folder = os.path.join(args.exp_folder, args.exp_name)
    if not os.path.isdir(exp_folder):
        raise FileNotFoundError(f"Experiment folder not found: {exp_folder}")

    log_path = os.path.join(exp_folder, 'eval_log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    )

    FIGURE_DIR = os.path.join(exp_folder, 'eval_figures')
    MODEL_DIR  = os.path.join(exp_folder, 'saved_models')
    os.makedirs(FIGURE_DIR, exist_ok=True)

    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    seed_everything(42)

    real_root = args.path_to_real_images

    cv_test_f1_scores = []
    external_test_f1_scores = []
    cv_test_avg_f1_rows = []
    external_test_avg_f1_rows = []

    # 5-fold evaluation
    for fold_index in range(5):
        logging.info("=" * 40)
        logging.info(f"Fold {fold_index} evaluation")
        logging.info("=" * 40)

        seed_everything(42 + fold_index)

        test_images = glob(os.path.join(real_root, f'fold{fold_index}', 'test', '*', '*.png'))
        if len(test_images) == 0:
            logging.warning(f"No test images found for fold {fold_index}")
            cv_test_f1_scores.append(np.full((num_classes,), np.nan))
            cv_test_avg_f1_rows.append({"fold": fold_index, "weighted": np.nan, "macro": np.nan})
        else:
            trainer = ViTTrainer(
                num_classes=num_classes,
                batch_size=args.batches,
                learning_rate=1e-4,   # dummy for eval
                fold_index=fold_index,
                hf_model_name=args.hf_model_name,
                freeze_encoder_layers=args.freeze_encoder_layers,
                dropout_rate=args.dropout_rate,
                num_image_channels=NUM_CHANNEL,
            )
            fold_fig_dir = os.path.join(FIGURE_DIR, f'fold{fold_index}')
            csv_path = os.path.join(exp_folder, f'cv_test_fold{fold_index}_predictions.csv')

            f1_vec = trainer.run_test(
                image_paths=test_images,
                test_name='cv_test_eval',
                figure_dir=fold_fig_dir,
                model_dir=MODEL_DIR,
                csv_path=csv_path,
            )
            cv_test_f1_scores.append(f1_vec)
            cv_test_avg_f1_rows.append({"fold": fold_index, **trainer.last_avg_f1})

        # External evaluation
        if args.path_to_external_test_images is not None:
            external_images = glob(os.path.join(args.path_to_external_test_images, '*', '*.png'))
            if len(external_images) == 0:
                logging.warning("No external images found.")
                external_test_f1_scores.append(np.full((num_classes,), np.nan))
                external_test_avg_f1_rows.append({"fold": fold_index, "weighted": np.nan, "macro": np.nan})
            else:
                trainer = ViTTrainer(
                    num_classes=num_classes,
                    batch_size=args.batches,
                    learning_rate=1e-4,
                    fold_index=fold_index,
                    hf_model_name=args.hf_model_name,
                    freeze_encoder_layers=args.freeze_encoder_layers,
                    dropout_rate=args.dropout_rate,
                    num_image_channels=NUM_CHANNEL,
                )
                ext_csv_path = os.path.join(exp_folder, f'external_fold{fold_index}_predictions.csv')
                f1_vec_ext = trainer.run_test(
                    image_paths=external_images,
                    test_name='external_eval',
                    figure_dir=os.path.join(FIGURE_DIR, f'fold{fold_index}'),
                    model_dir=MODEL_DIR,
                    csv_path=ext_csv_path,
                )
                external_test_f1_scores.append(f1_vec_ext)
                external_test_avg_f1_rows.append({"fold": fold_index, **trainer.last_avg_f1})

    # ==============================
    # Summaries for CV test
    # ==============================
    logging.info("="*40)
    logging.info("**Final Summary (CV Test)**")
    logging.info("="*40)

    cv_test_f1_scores = np.asarray(cv_test_f1_scores, dtype=float)
    for fold_idx, scores in enumerate(cv_test_f1_scores):
        readable = {CLASS_LABELS[i]: round(scores[i], 3)
                    for i in range(min(len(scores), len(CLASS_LABELS)))}
        logging.info(f"  Fold {fold_idx}: {readable}")
        avg_row = cv_test_avg_f1_rows[fold_idx]
        logging.info(f"  Fold {fold_idx} (avg) weighted: {avg_row['weighted']:.3f}, "
                     f"macro: {avg_row['macro']:.3f}")

    df_cv = pd.DataFrame(
        cv_test_f1_scores,
        columns=[CLASS_LABELS[i] for i in range(cv_test_f1_scores.shape[1])]
    )
    df_cv['weighted'] = [r['weighted'] for r in cv_test_avg_f1_rows]
    df_cv['macro']    = [r['macro'] for r in cv_test_avg_f1_rows]
    df_cv.index.name = "fold"
    df_cv.to_csv(os.path.join(exp_folder, 'cv_test_f1_scores_eval.csv'))

    with np.errstate(invalid='ignore', divide='ignore'):
        per_class_mean = np.nanmean(cv_test_f1_scores, axis=0)
        per_class_std  = np.nanstd(cv_test_f1_scores, axis=0)

    logging.info("Per-class F1 mean ± std across folds (CV test, eval):")
    for i in range(len(CLASS_LABELS)):
        logging.info(f"  {CLASS_LABELS[i]}: {per_class_mean[i]:.3f} ({per_class_std[i]:.3f})")

    cv_w = np.array([r['weighted'] for r in cv_test_avg_f1_rows], dtype=float)
    cv_m = np.array([r['macro'] for r in cv_test_avg_f1_rows], dtype=float)
    logging.info(
        f"Average F1 across folds (CV test, eval) — "
        f"weighted: {np.nanmean(cv_w):.3f} ({np.nanstd(cv_w):.3f}), "
        f"macro: {np.nanmean(cv_m):.3f} ({np.nanstd(cv_m):.3f})"
    )

    # ==============================
    # Summaries for External test
    # ==============================
    if len(external_test_avg_f1_rows) > 0:
        logging.info("="*40)
        logging.info("**Final Summary (External Test)**")
        logging.info("="*40)

        external_test_f1_scores = np.asarray(external_test_f1_scores, dtype=float)

        for fold_idx, scores in enumerate(external_test_f1_scores):
            readable = {CLASS_LABELS[i]: round(scores[i], 3)
                        for i in range(min(len(scores), len(CLASS_LABELS)))}
            logging.info(f"  Fold {fold_idx}: {readable}")
            avg_row = external_test_avg_f1_rows[fold_idx]
            logging.info(f"  Fold {fold_idx} (avg) weighted: {avg_row['weighted']:.3f}, "
                         f"macro: {avg_row['macro']:.3f}")

        df_ext = pd.DataFrame(
            external_test_f1_scores,
            columns=[CLASS_LABELS[i] for i in range(external_test_f1_scores.shape[1])]
        )
        df_ext['weighted'] = [r['weighted'] for r in external_test_avg_f1_rows]
        df_ext['macro']    = [r['macro'] for r in external_test_avg_f1_rows]
        df_ext.index.name = "fold"
        df_ext.to_csv(os.path.join(exp_folder, 'external_test_f1_scores_eval.csv'))

        with np.errstate(invalid='ignore', divide='ignore'):
            per_class_mean_ext = np.nanmean(external_test_f1_scores, axis=0)
            per_class_std_ext  = np.nanstd(external_test_f1_scores, axis=0)

        logging.info("Per-class F1 mean ± std across folds (External Test, eval):")
        for i in range(len(CLASS_LABELS)):
            logging.info(f"  {CLASS_LABELS[i]}: {per_class_mean_ext[i]:.3f} "
                         f"({per_class_std_ext[i]:.3f})")

        ext_w = np.array([r['weighted'] for r in external_test_avg_f1_rows], dtype=float)
        ext_m = np.array([r['macro'] for r in external_test_avg_f1_rows], dtype=float)
        logging.info(
            f"Average F1 across folds (External test, eval) — "
            f"weighted: {np.nanmean(ext_w):.3f} ({np.nanstd(ext_w):.3f}), "
            f"macro: {np.nanmean(ext_m):.3f} ({np.nanstd(ext_m):.3f})"
        )
