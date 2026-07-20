"""
mimic_cxr_eval.py
=================
Zero-shot evaluation of the Aura-X ConvNeXtV2-Tiny student model
on the MIMIC-CXR-JPG dataset (external validation for Table 7).

Requirements:
  - MIMIC-CXR-JPG access via PhysioNet (https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
  - Trained Aura-X student checkpoint (.pth file)
  - pip install torch torchvision timm scikit-learn pandas tqdm opencv-python

Usage:
  python mimic_cxr_eval.py \
    --mimic_root "D:/mimic-cxr-jpg" \
    --checkpoint "models/student_ema_best.pth" \
    --output    "docs/mimic_results.csv"
"""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ── Label Mapping ─────────────────────────────────────────────────────────────
# NIH class index order (must match your student model's output order)
NIH_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# MIMIC-CXR / CheXpert uses slightly different column names
MIMIC_TO_NIH = {
    'Atelectasis':    'Atelectasis',
    'Cardiomegaly':   'Cardiomegaly',
    'Pleural Effusion': 'Effusion',
    'Consolidation':  'Consolidation',
    'Edema':          'Edema',
    'Pneumonia':      'Pneumonia',
    'Pneumothorax':   'Pneumothorax',
}
# Only these 7 classes are evaluable (others have no MIMIC equivalent)
EVAL_CLASSES = list(MIMIC_TO_NIH.values())
EVAL_INDICES = [NIH_CLASSES.index(c) for c in EVAL_CLASSES]


# ── Dataset ───────────────────────────────────────────────────────────────────
class MimicCXRDataset(Dataset):
    def __init__(self, mimic_root, split='test', img_size=384):
        """
        Loads MIMIC-CXR-JPG test split.
        Expected directory layout:
          mimic_root/
            mimic-cxr-2.0.0-chexpert.csv.gz   (labels)
            files/
              p10/p10000032/s50414267/*.jpg    (images)
        """
        self.mimic_root = mimic_root
        self.img_size   = img_size

        # Load CheXpert labels
        label_file = os.path.join(mimic_root, 'mimic-cxr-2.0.0-chexpert.csv.gz')
        if not os.path.exists(label_file):
            raise FileNotFoundError(
                f"Label file not found: {label_file}\n"
                "Download from PhysioNet: https://physionet.org/content/mimic-cxr-jpg/2.0.0/"
            )
        df_labels = pd.read_csv(label_file)

        # Load split metadata
        split_file = os.path.join(mimic_root, 'mimic-cxr-2.0.0-split.csv.gz')
        df_split   = pd.read_csv(split_file)
        df_split   = df_split[df_split['split'] == split]

        # Merge
        df = df_split.merge(df_labels, on=['subject_id', 'study_id'], how='inner')

        # Drop rows where ALL mapped labels are NaN
        mimic_cols = list(MIMIC_TO_NIH.keys())
        df = df.dropna(subset=mimic_cols, how='all')

        # Fill NaN with 0 (uncertain treated as negative — conservative)
        for col in mimic_cols:
            df[col] = df[col].fillna(0).clip(0, 1)

        self.df = df.reset_index(drop=True)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])

        print(f"[MIMIC] Loaded {len(self.df)} scans from {split} split.")

    def _build_img_path(self, row):
        """Reconstruct JPG file path from MIMIC directory structure."""
        p = f"p{str(row['subject_id'])[:2]}"
        subj = f"p{row['subject_id']}"
        study = f"s{row['study_id']}"
        # Use the first DICOM in that study (frontal PA preferred)
        study_dir = os.path.join(self.mimic_root, 'files', p, subj, study)
        if not os.path.isdir(study_dir):
            return None
        jpgs = [f for f in os.listdir(study_dir) if f.endswith('.jpg')]
        if not jpgs:
            return None
        # Prefer PA view
        pa = [f for f in jpgs if '_PA' in f or '-PA' in f]
        return os.path.join(study_dir, pa[0] if pa else jpgs[0])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._build_img_path(row)

        if img_path is None or not os.path.exists(img_path):
            # Return zeros on missing file (will be masked later)
            img_tensor = torch.zeros(3, self.img_size, self.img_size)
            valid = False
        else:
            img = cv2.imread(img_path)
            if img is None:
                img_tensor = torch.zeros(3, self.img_size, self.img_size)
                valid = False
            else:
                # Apply same CLAHE preprocessing as training
                gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                eq    = clahe.apply(gray)
                rgb   = cv2.cvtColor(eq, cv2.COLOR_GRAY2RGB)
                img_tensor = self.transform(rgb)
                valid = True

        # Build label vector (7 evaluable classes only)
        labels = torch.zeros(len(EVAL_CLASSES))
        for i, mimic_col in enumerate(MIMIC_TO_NIH.keys()):
            labels[i] = float(row.get(mimic_col, 0.0))

        return img_tensor, labels, valid


# ── Student Model ─────────────────────────────────────────────────────────────
class AuraXStudent(nn.Module):
    def __init__(self, num_classes=14, dropout_rates=None):
        super().__init__()
        if dropout_rates is None:
            dropout_rates = [0.20, 0.25, 0.30, 0.35, 0.40]
        self.backbone = timm.create_model(
            'convnextv2_tiny.fcmae_ft_in22k_in1k_384',
            pretrained=False,
            num_classes=0,   # remove head
        )
        feat_dim = self.backbone.num_features  # 768
        self.dropouts = nn.ModuleList([nn.Dropout(p=r) for r in dropout_rates])
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        out  = torch.stack([self.fc(d(feat)) for d in self.dropouts]).mean(0)
        return out


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # Load model
    model = AuraXStudent(num_classes=14)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    # Support both raw state_dict and EMA dict wrappers
    state = ckpt.get('ema_state_dict', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f"[Model] Checkpoint loaded: {args.checkpoint}")

    # Dataset
    dataset    = MimicCXRDataset(args.mimic_root, split='test', img_size=384)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True, shuffle=False)

    all_preds  = []
    all_labels = []
    skipped    = 0

    with torch.no_grad():
        for imgs, labels, valids in tqdm(dataloader, desc="Evaluating MIMIC-CXR"):
            valid_mask = valids.bool()
            if valid_mask.sum() == 0:
                skipped += len(imgs)
                continue

            imgs_valid   = imgs[valid_mask].to(device)
            labels_valid = labels[valid_mask]

            logits = model(imgs_valid)
            probs  = torch.sigmoid(logits)[:, EVAL_INDICES].cpu().numpy()
            lbls   = labels_valid.numpy()

            all_preds.append(probs)
            all_labels.append(lbls)
            skipped += (~valid_mask).sum().item()

    if skipped > 0:
        print(f"[Warning] {skipped} scans skipped (missing image files).")

    all_preds  = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Per-class AUC (only classes with at least one positive)
    results = {}
    valid_aucs = []
    for i, cls_name in enumerate(EVAL_CLASSES):
        pos = all_labels[:, i].sum()
        if pos < 5:
            print(f"  [Skip] {cls_name}: only {int(pos)} positives — too few for AUC")
            continue
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        results[cls_name] = round(auc, 4)
        valid_aucs.append(auc)
        print(f"  {cls_name:25s}  AUC = {auc:.4f}")

    macro_auc = float(np.mean(valid_aucs))
    results['Macro-AUC (MIMIC, 7-class)'] = round(macro_auc, 4)
    nih_auc   = 0.9379  # Internal NIH result
    delta     = round((macro_auc - nih_auc) * 100, 2)
    results['Degradation vs NIH (%)'] = f"{delta:+.2f}%"

    print(f"\n{'='*55}")
    print(f"  Macro-AUC on MIMIC-CXR (7 overlapping classes): {macro_auc:.4f}")
    print(f"  Degradation vs NIH internal test:               {delta:+.2f}%")
    print(f"{'='*55}\n")

    # Save
    df_out = pd.DataFrame([results])
    df_out.to_csv(args.output, index=False)
    print(f"[Saved] Results -> {args.output}")

    # Update Table 7 in paper instruction
    print("\n📄 UPDATE TABLE 7 IN YOUR PAPER:")
    print(f"   MIMIC-CXR | Prospective External | {macro_auc:.4f} | {delta:+.2f}% | Completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aura-X MIMIC-CXR Zero-Shot Evaluation')
    parser.add_argument('--mimic_root',  type=str, required=True,
                        help='Root directory of MIMIC-CXR-JPG dataset')
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='Path to Aura-X student .pth checkpoint')
    parser.add_argument('--output',      type=str,
                        default='docs/mimic_results.csv',
                        help='Output CSV path for results')
    parser.add_argument('--batch_size',  type=int, default=16)
    args = parser.parse_args()
    evaluate(args)
