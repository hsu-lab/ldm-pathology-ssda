"""
Fine-tune the Hugging Face implementation of ViT using locally stored weights.
"""
import os, sys, argparse, logging, random
from glob import glob
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score as sk_f1

from dataloader import DataProcessor
from util import FocalLoss

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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def setup_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # remove and close existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fmt = logging.Formatter('%(message)s')
    fh = logging.FileHandler(log_path, mode='w')
    sh = logging.StreamHandler(sys.stdout)
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


def get_experiment_name(prefix="exp"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# -------------------------
# Trainer
# -------------------------
class ViTTrainer:
    def __init__(
        self,
        num_classes,
        num_epochs,
        batch_size,
        learning_rate,
        fold_index,
        syn_root=None,
        synth_loss_weight=0.5,
        hf_model_name="google/vit-base-patch16-224-in21k",
        hf_weights_root="./hf_weights",
        freeze_encoder_layers=0,
        dropout_rate=0.0,
        num_image_channels=3,
        patience=10,
        loss='bce',  # 'bce' or 'focal' or 'focal_w_weights'
    ):
        self.num_classes = num_classes
        self.epochs = num_epochs
        self.batch = batch_size
        self.fold_index = fold_index
        self.syn_root = syn_root
        self.synth_loss_weight = synth_loss_weight
        self.learning_rate = learning_rate
        self.hf_model_name = hf_model_name
        self.hf_weights_root = hf_weights_root
        self.freeze_encoder_layers = max(0, int(freeze_encoder_layers))
        self.dropout_rate = dropout_rate
        self.num_image_channels = num_image_channels
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'Using device: {self.device}')
        self.last_avg_f1 = {"weighted": np.nan, "macro": np.nan}
        self.loss = loss
        self.local_model_path = self._resolve_local_model_path()

        # Image processor (gives mean/std and size)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.local_model_path,
            use_fast=True,
            local_files_only=True,
        )

    def _resolve_local_model_path(self):
        if os.path.isdir(self.hf_model_name):
            model_path = self.hf_model_name
        else:
            model_path = os.path.join(self.hf_weights_root, self.hf_model_name)
        model_path = os.path.abspath(model_path)

        required_files = [
            "config.json",
            "preprocessor_config.json",
        ]
        missing_files = [fname for fname in required_files if not os.path.isfile(os.path.join(model_path, fname))]
        has_model_weights = any(
            os.path.isfile(os.path.join(model_path, fname))
            for fname in ("model.safetensors", "pytorch_model.bin")
        )

        if not os.path.isdir(model_path) or missing_files or not has_model_weights:
            raise FileNotFoundError(
                "Local Hugging Face model files not found. "
                f"Expected a full model snapshot under '{model_path}' containing "
                "'config.json', 'preprocessor_config.json', and either "
                "'model.safetensors' or 'pytorch_model.bin'."
            )

        logging.info(f"Loading Hugging Face assets locally from: {model_path}")
        return model_path

    def _make_model(self):
        # Load ViT with pretrained weights; replace head for our num_classes
        cfg = AutoConfig.from_pretrained(self.local_model_path, local_files_only=True)
        cfg.hidden_dropout_prob = self.dropout_rate  # MLP/patch-embedding dropout
        cfg.attention_probs_dropout_prob = self.dropout_rate  # attention dropout
        cfg.classifier_dropout = self.dropout_rate  # head dropout
        cfg.num_labels = self.num_classes

        model = ViTForImageClassification.from_pretrained(
            self.local_model_path,
            config=cfg,
            ignore_mismatched_sizes=True,  # adapt head
            local_files_only=True,
            use_safetensors=False,
        )

        # Optionally freeze the first K encoder blocks (like freezing early resnet layers)
        if self.freeze_encoder_layers > 0:
            enc = model.vit.encoder
            layers = enc.layer  # ModuleList
            K = min(self.freeze_encoder_layers, len(layers))
            logging.info(f"Freezing first {K}/{len(layers)} ViT encoder layers.")
            for i in range(K):
                for p in layers[i].parameters():
                    p.requires_grad = False

            # Freeze the patch+pos embeddings too if K == len(layers) (fully frozen encoder)
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
        std = getattr(self.image_processor, "image_std", [0.5, 0.5, 0.5])[:self.num_image_channels]

        train_tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomVerticalFlip(0.5),
            # transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        eval_tfms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        return train_tfms, eval_tfms

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
            plt.text(
                j,
                i,
                format(cm_plot[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm_plot[i, j] > thresh else "black",
            )
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f'{test_name}_confusion_matrix.png'))
        plt.clf()

    def _ce_per_sample(logits, labels):
        return F.cross_entropy(logits, labels, reduction='none')

    def start_training(self, train_paths, valid_paths):
        # Data
        train_tfms, eval_tfms = self._build_transforms()
        train_ds = DataProcessor(imgs_dir=train_paths, channel=NUM_CHANNEL, split='train', transform=train_tfms, syn_root=self.syn_root)
        val_ds = DataProcessor(imgs_dir=valid_paths, channel=NUM_CHANNEL, split='val', transform=eval_tfms)

        logging.info("=" * 40)
        logging.info(f"Fold {self.fold_index}")
        logging.info(f"Images for Training: {len(train_ds)}")
        logging.info(f"Images for Validation: {len(val_ds)}")

        class_weights_tensor = None
        if self.loss == 'focal_w_weights':
            all_labels = [train_ds[i]['label'].item() for i in range(len(train_ds))]
            class_counts = Counter(all_labels)
            logging.info(f"Class counts: {class_counts}")
            num_classes = self.num_classes
            counts = np.array([class_counts.get(i, 0) for i in range(num_classes)])
            weights = 1.0 / (counts + 1e-6)
            weights = weights / weights.sum() * num_classes  # normalize
            logging.info(f"Normalized class weights for focal loss: {weights.tolist()}")
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        logging.info("=" * 40)

        trainloader = DataLoader(train_ds, batch_size=self.batch, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        validloader = DataLoader(val_ds, batch_size=self.batch, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

        # Model
        model = self._make_model().to(self.device)
        model.train()

        optimizer = torch.optim.Adam(
            (p for p in model.parameters() if p.requires_grad),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)  # , verbose=True

        # define loss functions
        use_focal = (self.loss in ['focal', 'focal_w_weights'])

        def ce_per_sample(logits, labels):
            return F.cross_entropy(logits, labels, reduction='none')

        def focal_per_sample(logits, labels):
            if class_weights_tensor is None:
                fl = FocalLoss(alpha=1.0, gamma=1.5, reduction='none')
            else:
                fl = FocalLoss(alpha=class_weights_tensor, gamma=2.0, reduction='none')
            return fl(logits, labels)

        def per_sample_loss(logits, labels):
            return focal_per_sample(logits, labels) if use_focal else ce_per_sample(logits, labels)

        def map_weights(is_synth_tensor: torch.Tensor):
            return 1.0 + (self.synth_loss_weight - 1.0) * is_synth_tensor.to(self.device)

        train_losses, val_losses = [], []
        metrics = {
            'accuracy': {i: [] for i in range(self.num_classes)},
            'sensitivity': {i: [] for i in range(self.num_classes)},
            'specificity': {i: [] for i in range(self.num_classes)},
            'f1': {i: [] for i in range(self.num_classes)},
        }

        best_val = np.inf
        early = 0
        epoch = 0
        global_step = 0

        while early < self.patience and epoch < self.epochs:
            epoch += 1
            logging.info(f"Epoch {epoch}")
            model.train()
            running_train = 0.0
            n_batches = max(1, len(trainloader))
            pbar = tqdm(trainloader, desc=f"Train | epoch {epoch}", leave=False)
            for step, batch in enumerate(pbar, 1):
                images = batch['image'].to(self.device, dtype=torch.float)
                labels = batch['label'].to(self.device, dtype=torch.long)
                is_syn = batch['is_syn'].to(self.device)

                optimizer.zero_grad()
                logits = model(images).logits
                per_s = per_sample_loss(logits, labels)
                w = map_weights(is_syn)
                loss = (per_s * w).mean()

                loss.backward()
                optimizer.step()

                running_train += float(loss.item()) * images.size(0)
                avg_so_far = running_train / step
                current_lr = optimizer.param_groups[0]["lr"]
                pbar.set_postfix(loss=f"{avg_so_far:.4f}", lr=f"{current_lr:.2e}")

            # Validation
            model.eval()
            running_val = 0.0
            y_true, y_pred, scores = [], [], []
            with torch.no_grad():
                vbar = tqdm(validloader, desc=f"Valid | epoch {epoch}", leave=False)
                for batch in vbar:
                    images = batch['image'].to(self.device, dtype=torch.float)
                    labels = batch['label'].to(self.device, dtype=torch.long)
                    is_syn = batch['is_syn'].to(self.device)

                    logits = model(images).logits
                    per_s = per_sample_loss(logits, labels)
                    w = map_weights(is_syn)
                    loss = (per_s * w).mean()

                    running_val += float(loss.item()) * images.size(0)

                    prob = F.softmax(logits.detach().cpu(), dim=1)
                    top1 = prob.argmax(dim=1)
                    y_pred.extend(top1.numpy().tolist())
                    y_true.extend(labels.cpu().numpy().tolist())
                    scores.extend(prob.numpy().tolist())

            avg_train = running_train / max(1, len(trainloader))
            avg_val = running_val / max(1, len(validloader))
            scheduler.step(avg_val)

            cnf = cm(y_true, y_pred, labels=list(range(self.num_classes)))
            FP = cnf.sum(axis=0) - np.diag(cnf)
            FN = cnf.sum(axis=1) - np.diag(cnf)
            TP = np.diag(cnf)
            TN = cnf.sum() - (FP + FN + TP)
            f_p, f_n, t_p, t_n = FP.astype(float), FN.astype(float), TP.astype(float), TN.astype(float)

            acc = (t_p + t_n) / (f_p + f_n + t_p + t_n + 1e-12)
            rec = t_p / (t_p + f_n + 1e-12)
            spe = t_n / (t_n + f_p + 1e-12)
            pre = t_p / (t_p + f_p + 1e-12)
            f1 = 2 * (rec * pre / (rec + pre + 1e-12))

            train_losses.append(avg_train)
            val_losses.append(avg_val)
            for i in range(self.num_classes):
                metrics['accuracy'][i].append(acc[i])
                metrics['sensitivity'][i].append(rec[i])
                metrics['specificity'][i].append(spe[i])
                metrics['f1'][i].append(f1[i])

            logging.info(f"Epoch:{epoch}/{self.epochs} - Training Loss:{avg_train:.6f} | Validation Loss:{avg_val:.6f}")
            logging.info(f"Accuracy:{acc}\nF1:{f1}\nSensitivity:{rec}\nSpecificity:{spe}")

            if avg_val <= best_val:
                logging.info(f"Validation loss decreased ({best_val:.6f} --> {avg_val:.6f}). Saving model ...")
                best_val = avg_val
                early = 0
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, f'vit_fold{self.fold_index}.pth'))
                logging.info("-" * 40)
            else:
                early += 1
                logging.info("-" * 40)

            fig_dir = os.path.join(FIGURE_DIR, f'fold{self.fold_index}')
            os.makedirs(fig_dir, exist_ok=True)

            plt.plot(train_losses, label='Training loss')
            plt.plot(val_losses, label='Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(frameon=False)
            plt.savefig(os.path.join(fig_dir, 'losses.png'))
            plt.clf()

            for metric_name in ["accuracy", "sensitivity", "specificity", "f1"]:
                for i, name in enumerate(CLASS_LABELS.values()):
                    plt.plot(metrics[metric_name][i], label=name)
                plt.xlabel('Epoch')
                plt.ylabel('Score')
                plt.legend(frameon=False, ncol=3)
                plt.savefig(os.path.join(fig_dir, f'{metric_name}.png'))
                plt.clf()

    def run_test(self, image_paths, test_name):
        logging.info("=" * 40)
        logging.info("Running test...")
        fig_dir = os.path.join(FIGURE_DIR, f'fold{self.fold_index}')
        os.makedirs(fig_dir, exist_ok=True)

        _, tfm = self._build_transforms()
        ds = DataProcessor(imgs_dir=image_paths, channel=NUM_CHANNEL, split='test', transform=tfm)
        logging.info(f"Images for testing: {len(ds)}")
        loader = DataLoader(ds, batch_size=self.batch, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

        # Rebuild model and load best weights
        model = self._make_model().to(self.device)
        ckpt_path = os.path.join(MODEL_DIR, f'vit_fold{self.fold_index}.pth')
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        model.eval()
        logging.info("Model Weights Loaded")

        criterion = nn.CrossEntropyLoss()
        y_true, y_pred = [], []
        running_test = 0.0

        with torch.no_grad():
            tbar = tqdm(loader, desc="Test", leave=False)
            for batch in tbar:
                images = batch['image'].to(self.device, dtype=torch.float)
                labels = batch['label'].to(self.device, dtype=torch.long)
                logits = model(images).logits
                loss = criterion(logits, labels)
                running_test += float(loss.item()) * images.size(0)

                preds = logits.detach().argmax(dim=1)
                y_pred.extend(preds.cpu().numpy().tolist())
                y_true.extend(labels.cpu().numpy().tolist())

        avg_test = running_test / max(1, len(loader))
        cnf = cm(y_true, y_pred, labels=list(range(self.num_classes)))
        logging.info("Confusion matrix:")
        logging.info(cnf)
        self._plot_confmat(fig_dir, cnf, list(CLASS_LABELS.values()), test_name, normalize=True)

        per_class_f1 = sk_f1(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0,
        )

        macro_f1 = sk_f1(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            average='macro',
            zero_division=0,
        )
        weighted_f1 = sk_f1(
            y_true,
            y_pred,
            labels=list(range(self.num_classes)),
            average='weighted',
            zero_division=0,
        )
        self.last_avg_f1 = {"weighted": float(weighted_f1), "macro": float(macro_f1)}

        logging.info(f"Test loss: {avg_test:.6f}")
        logging.info(f"Per-class F1: {per_class_f1}")
        logging.info(f"F1 (avg) weighted/macro: {weighted_f1:.4f} / {macro_f1:.4f}")

        return per_class_f1


# -------------------------
# main function
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_to_real_images',
        type=str,
        required=True,
        help='parent path to all the images (expects fold{0..4}/{train,val,test}/class/*.png)',
    )
    parser.add_argument('--path_to_syn_images', type=str, default=None)
    parser.add_argument(
        '--path_to_external_test_images',
        type=str,
        default=None,
        help='parent path to images for external validation (class/*.png)',
    )
    parser.add_argument('--num_image_channels', type=int, default=3, help='number of channels (ViT expects 3)')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    # parser.add_argument('--synthetic_weight', type=float, default=0.25)
    parser.add_argument('--batches', type=int, default=32)
    # 1e-4 for bce loss
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--exp_folder', type=str, default='exp')
    # parser.add_argument('--exp_name', type=str, default=None)

    # ViT
    parser.add_argument(
        '--hf_model_name',
        type=str,
        default='./models--google--vit-base-patch16-224-in21k/snapshots/b4569560a39a0f1af58e3ddaf17facf20ab919b0',
        help='local model subdirectory under --hf_weights_root, or an absolute/relative path to a local Hugging Face model snapshot',
    )
    parser.add_argument(
        '--hf_weights_root',
        type=str,
        default='./hf_weights',
        help='root directory containing locally downloaded Hugging Face model snapshots',
    )
    parser.add_argument(
        '--freeze_encoder_layers',
        type=int,
        default=0,
        help='freeze first K transformer blocks (0 = train all)',
    )

    args = parser.parse_args()

    NUM_CHANNEL = args.num_image_channels
    num_classes = args.num_classes

    print(f'Using GPU: {torch.cuda.is_available()}')

    for syn_weight in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        print("\n" + "=" * 60)
        print(f"RUNNING EXPERIMENT WITH SYNTHETIC WEIGHT = {syn_weight}")
        print("=" * 60)

        synthetic_weight = syn_weight

        # experiment folders + logging
        # exp_name = args.exp_name or get_experiment_name("vit")
        exp_name = f'vit_syn_weight_{synthetic_weight}'
        os.makedirs(args.exp_folder, exist_ok=True)
        exp_folder = os.path.join(args.exp_folder, exp_name)
        os.makedirs(exp_folder, exist_ok=False)

        # log
        log_path = os.path.join(exp_folder, 'output_log.txt')
        setup_logger(log_path)
        logging.info(f'synthetic_weight: {synthetic_weight}')
        logging.info(f'dropout_rate: {args.dropout_rate}')
        logging.info(f'learning_rate: {args.learning_rate}')
        logging.info(f'patience: {args.patience}')
        logging.info(f'hf_model_name: {args.hf_model_name}')
        logging.info(f'hf_weights_root: {os.path.abspath(args.hf_weights_root)}')

        FIGURE_DIR = os.path.join(exp_folder, 'saved_figures')
        os.makedirs(FIGURE_DIR, exist_ok=True)
        MODEL_DIR = os.path.join(exp_folder, 'saved_models')
        os.makedirs(MODEL_DIR, exist_ok=True)

        # seed
        SEED = 42
        seed_everything(SEED)

        # 5-fold CV loop (mirrors your ResNet script)
        cv_test_f1_scores = []
        external_test_f1_scores = []
        cv_test_avg_f1_rows = []
        external_test_avg_f1_rows = []

        for fold_index in range(5):
            seed_everything(SEED + fold_index)

            real_root = args.path_to_real_images
            syn_root = args.path_to_syn_images
            real_train = glob(os.path.join(real_root, f'fold{fold_index}', 'train', '*', '*.png'))
            syn_train = glob(os.path.join(syn_root, f'fold{fold_index}', "train", "*", "*.png")) if syn_root else []
            train_images = sorted(set(real_train) | set(syn_train))
            valid_images = glob(os.path.join(real_root, f'fold{fold_index}', 'val', '*', '*.png'))
            test_images = glob(os.path.join(real_root, f'fold{fold_index}', 'test', '*', '*.png'))

            trainer = ViTTrainer(
                num_classes=num_classes,
                num_epochs=args.num_epochs,
                batch_size=args.batches,
                learning_rate=args.learning_rate,
                fold_index=fold_index,
                syn_root=syn_root,
                synth_loss_weight=synthetic_weight,
                hf_model_name=args.hf_model_name,
                hf_weights_root=args.hf_weights_root,
                freeze_encoder_layers=args.freeze_encoder_layers,
                dropout_rate=args.dropout_rate,
                num_image_channels=NUM_CHANNEL,
                patience=args.patience,
                loss='bce',
            )
            trainer.start_training(train_images, valid_images)

            cv_test_f1 = trainer.run_test(test_images, test_name='cv_test')
            cv_test_f1_scores.append(cv_test_f1)
            cv_test_avg_f1_rows.append({"fold": fold_index, **trainer.last_avg_f1})

            if args.path_to_external_test_images is not None:
                external_images = glob(os.path.join(args.path_to_external_test_images, '*', '*.png'))
                ext_f1 = trainer.run_test(external_images, test_name='external_validation')
                external_test_f1_scores.append(ext_f1)
                external_test_avg_f1_rows.append({"fold": fold_index, **trainer.last_avg_f1})

        logging.info("=" * 40)
        logging.info("**Final Summary**")
        logging.info("=" * 40)

        # Save CV results
        cv_test_f1_scores = np.asarray(cv_test_f1_scores, dtype=float)
        logging.info("Per-class F1 scores for each fold (CV test):")
        for fold_idx, scores in enumerate(cv_test_f1_scores):
            readable = {CLASS_LABELS[i]: round(scores[i], 3) for i in range(len(scores))}
            logging.info(f"  Fold {fold_idx}: {readable}")
            avg_row = cv_test_avg_f1_rows[fold_idx]
            logging.info(f"  Fold {fold_idx} (avg) weighted: {avg_row['weighted']:.3f}, macro: {avg_row['macro']:.3f}")

        df_cv = pd.DataFrame(cv_test_f1_scores, columns=[CLASS_LABELS[i] for i in range(cv_test_f1_scores.shape[1])])
        df_cv['weighted'] = [r['weighted'] for r in cv_test_avg_f1_rows]
        df_cv['macro'] = [r['macro'] for r in cv_test_avg_f1_rows]
        df_cv.index.name = "fold"
        df_cv.to_csv(os.path.join(exp_folder, 'cv_test_f1_scores.csv'))
        with np.errstate(invalid='ignore', divide='ignore'):
            per_class_mean = np.nanmean(cv_test_f1_scores, axis=0)
            per_class_std = np.nanstd(cv_test_f1_scores, axis=0)
        logging.info("Per-class F1 mean ± std across folds (CV test):")
        for i in range(len(CLASS_LABELS)):
            logging.info(f"  {CLASS_LABELS[i]}: {per_class_mean[i]:.3f} ({per_class_std[i]:.3f})")
        cv_w = np.array([r['weighted'] for r in cv_test_avg_f1_rows], dtype=float)
        cv_m = np.array([r['macro'] for r in cv_test_avg_f1_rows], dtype=float)
        logging.info(
            f"Average F1 across folds (CV test) — weighted: {np.nanmean(cv_w):.3f} ({np.nanstd(cv_w):.3f}), "
            f"macro: {np.nanmean(cv_m):.3f} ({np.nanstd(cv_m):.3f})"
        )

        # Save External results (if provided)
        if len(external_test_avg_f1_rows) > 0:
            external_test_f1_scores = np.asarray(external_test_f1_scores, dtype=float)
            logging.info("Per-class F1 scores for each fold (External Test):")
            for fold_idx, scores in enumerate(external_test_f1_scores):
                readable = {CLASS_LABELS[i]: round(scores[i], 3) for i in range(len(scores))}
                logging.info(f"  Fold {fold_idx}: {readable}")
                avg_row = external_test_avg_f1_rows[fold_idx]
                logging.info(f"  Fold {fold_idx} (avg) weighted: {avg_row['weighted']:.3f}, macro: {avg_row['macro']:.3f}")

            df_ext = pd.DataFrame(external_test_f1_scores, columns=[CLASS_LABELS[i] for i in range(external_test_f1_scores.shape[1])])
            df_ext['weighted'] = [r['weighted'] for r in external_test_avg_f1_rows]
            df_ext['macro'] = [r['macro'] for r in external_test_avg_f1_rows]
            df_ext.index.name = "fold"
            df_ext.to_csv(os.path.join(exp_folder, 'external_test_f1_scores.csv'))

            with np.errstate(invalid='ignore', divide='ignore'):
                per_class_mean_ext = np.nanmean(external_test_f1_scores, axis=0)
                per_class_std_ext = np.nanstd(external_test_f1_scores, axis=0)
            logging.info("Per-class F1 mean ± std across folds (External Test):")
            for i in range(len(CLASS_LABELS)):
                logging.info(f"  {CLASS_LABELS[i]}: {per_class_mean_ext[i]:.3f} ({per_class_std_ext[i]:.3f})")
            ext_w = np.array([r['weighted'] for r in external_test_avg_f1_rows], dtype=float)
            ext_m = np.array([r['macro'] for r in external_test_avg_f1_rows], dtype=float)
            logging.info(
                f"Average F1 across folds (External test) — weighted: {np.nanmean(ext_w):.3f} ({np.nanstd(ext_w):.3f}), "
                f"macro: {np.nanmean(ext_m):.3f} ({np.nanstd(ext_m):.3f})"
            )
