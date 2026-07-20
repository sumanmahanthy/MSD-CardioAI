# ================================================================
# CARDIOAI V7: STRONG TEACHER — EfficientNetV2-M
# Overcomes V5/V6 paper cons:
#   [CON-1] DenseNet121 teacher too weak (0.83 AUC) → EfficientNetV2-M targets 0.88+
#   [CON-2] RandomResizedCrop destroys heart/lung edges → RandomAffine
#   [CON-3] No gradient clipping → added clip_grad_norm_(1.0)
#   [CON-4] No early stopping → patience=6 early stop
#   [CON-5] Cold LR start → 2-epoch linear warmup + cosine decay
#   [CON-6] Single LR for all layers → differential LR (backbone vs head)
#   [CON-7] No TTA during validation → 3-view TTA
#   [CON-8] Best single checkpoint used → Checkpoint soup top-3 epochs
#   [CON-9] 15 epochs fixed → up to 30 with early stopping
# Target: 0.88–0.91 teacher AUC to provide strong distillation signal
# ================================================================

import os, gc, glob, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import timm

# ────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ────────────────────────────────────────────────────────────────
DATA_DIR        = "/kaggle/input/datasets/organizations/nih-chest-xrays/data/"
META_PATH       = DATA_DIR + "Data_Entry_2017.csv"
CKPT_DIR        = "/kaggle/working/"

num_gpus        = torch.cuda.device_count()
BATCH_SIZE      = 32 * max(1, num_gpus)   # EfficientNetV2-M needs slightly smaller batch
EPOCHS          = 30
WARMUP_EPOCHS   = 3
EARLY_STOP_PAT  = 6
LR_BACKBONE     = 3e-5
LR_HEAD         = 1e-4
SOUP_TOP_K      = 3
RESUME_EPOCH    = 0
RESUME_BEST_AUC = 0.0
NUM_WORKERS     = 4
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
    "Fibrosis","Pleural_Thickening","Hernia"
]
NUM_CLASSES = len(DISEASES)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
print(f"V7 TEACHER | {DEVICE} | GPUs:{num_gpus} | Batch:{BATCH_SIZE}")

# ────────────────────────────────────────────────────────────────
# 2. DATA
# ────────────────────────────────────────────────────────────────
metadata    = pd.read_csv(META_PATH)
image_paths = glob.glob(DATA_DIR + "images_*/images/*.png")
path_dict   = {os.path.basename(p): p for p in image_paths}
metadata    = metadata[metadata["Image Index"].isin(path_dict)].reset_index(drop=True)
metadata["image_path"] = metadata["Image Index"].map(path_dict)

def get_hard_labels(s):
    v = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, d in enumerate(DISEASES):
        if d in s.split("|"): v[i] = 1.0
    return v

metadata["hard_labels"] = list(np.stack(metadata["Finding Labels"].apply(get_hard_labels).values))

# Same fixed split used by ALL student notebooks (random_state=42 is the contract)
train_pat, temp_pat = train_test_split(metadata["Patient ID"].unique(), test_size=0.20, random_state=42)
val_pat,   test_pat = train_test_split(temp_pat, test_size=0.50, random_state=42)
train_df = metadata[metadata["Patient ID"].isin(train_pat)].reset_index(drop=True)
val_df   = metadata[metadata["Patient ID"].isin(val_pat)].reset_index(drop=True)
test_df  = metadata[metadata["Patient ID"].isin(test_pat)].reset_index(drop=True)
print(f"Split → Train:{len(train_df):,}  Val:{len(val_df):,}  Test:{len(test_df):,}")


class ChestXrayDataset(Dataset):
    def __init__(self, df, augment=False, img_size=224):
        self.df        = df
        self.augment   = augment
        self.img_size  = img_size
        aug_list = [
            T.Resize((img_size, img_size), antialias=True),
            T.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            T.RandomHorizontalFlip(p=0.5),
            # Mild intensity jitter — chest X-rays can vary in contrast/brightness
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.RandomErasing(p=0.1, scale=(0.02, 0.08)),   # randomly mask small regions
        ]
        val_list = [T.Resize((img_size, img_size), antialias=True)]
        self.transform = T.Compose([
            *(aug_list if augment else val_list),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=(0.48145, 0.45782, 0.40821), std=(0.26862, 0.26130, 0.27577)),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, torch.tensor(np.asarray(row["hard_labels"]), dtype=torch.float32)


def make_loader(df, augment, batch_size=BATCH_SIZE, img_size=224):
    return DataLoader(
        ChestXrayDataset(df, augment, img_size),
        batch_size=batch_size, shuffle=augment,
        num_workers=NUM_WORKERS, pin_memory=True,
        prefetch_factor=2, persistent_workers=True
    )

train_loader = make_loader(train_df, True)
val_loader   = make_loader(val_df,   False)

# ────────────────────────────────────────────────────────────────
# 3. MODEL — EfficientNetV2-M (stronger than DenseNet121)
# ────────────────────────────────────────────────────────────────
class StrongTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        # EfficientNetV2-M: superior to DenseNet121 on medical imaging benchmarks
        base     = timm.create_model("tf_efficientnetv2_m", pretrained=True, num_classes=0, drop_rate=0.3)
        self.backbone = base
        feat_dim      = base.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        return self.head(self.backbone(x))


model = StrongTeacher().to(DEVICE)
if num_gpus > 1:
    model = nn.DataParallel(model)


def asymmetric_focal_loss(logits, targets, gamma_neg=2, gamma_pos=1, clip=0.05):
    p     = torch.sigmoid(logits)
    p_neg = torch.clamp(p + clip, max=1.0)
    l_pos = -(1 - p) ** gamma_pos * targets       * torch.log(torch.clamp(p,     min=1e-6))
    l_neg = -(p_neg) ** gamma_neg * (1 - targets) * torch.log(torch.clamp(1 - p, min=1e-6))
    return (l_pos + l_neg).mean()


# ────────────────────────────────────────────────────────────────
# 4. OPTIMIZER + SCHEDULER
# ────────────────────────────────────────────────────────────────
model_ref = model.module if hasattr(model, "module") else model
optimizer = optim.AdamW([
    {"params": model_ref.backbone.parameters(), "lr": LR_BACKBONE},
    {"params": model_ref.head.parameters(),     "lr": LR_HEAD},
], weight_decay=1e-3)

warmup  = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
cosine  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=5e-7)
sched   = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])
scaler  = torch.amp.GradScaler("cuda")

best_auc     = RESUME_BEST_AUC
START_EPOCH  = RESUME_EPOCH
no_improve   = 0
top_ckpts    = []   # (auc, state_dict) for soup

if START_EPOCH > 0:
    p = f"{CKPT_DIR}v7_teacher_ep{START_EPOCH:02d}.pth"
    if os.path.exists(p):
        c = torch.load(p, map_location=DEVICE, weights_only=False)
        model.load_state_dict(c["model"]); optimizer.load_state_dict(c["optimizer"])
        sched.load_state_dict(c["scheduler"])
        best_auc = c.get("best_auc", RESUME_BEST_AUC) if RESUME_BEST_AUC == 0.0 else RESUME_BEST_AUC
        print(f"Resumed Epoch {START_EPOCH} | best_auc={best_auc:.4f}")


# ────────────────────────────────────────────────────────────────
# 5. HELPERS
# ────────────────────────────────────────────────────────────────
def validate_tta(loader):
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                p0 = torch.sigmoid(model(imgs))
                p1 = torch.sigmoid(model(torch.flip(imgs, [3])))
                p2 = torch.sigmoid(model(F.interpolate(imgs, (256, 256), mode="bilinear", align_corners=False)))
            preds.append(((p0 + p1 + p2) / 3).cpu().numpy())
            targs.append(lbls.numpy())
    return float(roc_auc_score(np.concatenate(targs), np.concatenate(preds), average="macro"))


# ────────────────────────────────────────────────────────────────
# 6. TRAINING LOOP
# ────────────────────────────────────────────────────────────────
for epoch in range(START_EPOCH, EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
    for imgs, lbls in pbar:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
            loss = asymmetric_focal_loss(model(imgs), lbls)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    sched.step()

    auc = validate_tta(val_loader)
    print(f"Epoch {epoch+1}  TTA Val AUC: {auc:.4f}  LR: {sched.get_last_lr()}")

    core = copy.deepcopy(model.module.state_dict() if hasattr(model, "module") else model.state_dict())
    top_ckpts.append((auc, core))
    top_ckpts.sort(key=lambda x: x[0], reverse=True)
    top_ckpts = top_ckpts[:SOUP_TOP_K]

    if auc > best_auc:
        best_auc, no_improve = auc, 0
        torch.save(core, f"{CKPT_DIR}v7_best_teacher.pth")
        print(f"   NEW BEST: {best_auc:.4f} → v7_best_teacher.pth")
    else:
        no_improve += 1
        print(f"   No improve ({no_improve}/{EARLY_STOP_PAT})")

    torch.save({"epoch": epoch+1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": sched.state_dict(), "best_auc": best_auc, "auc": auc},
               f"{CKPT_DIR}v7_teacher_ep{epoch+1:02d}.pth")

    if no_improve >= EARLY_STOP_PAT:
        print(f"Early stop at epoch {epoch+1}. Best AUC={best_auc:.4f}")
        break
    gc.collect()
    torch.cuda.empty_cache()   # FIX-3: flush CUDA allocator after each epoch


# ────────────────────────────────────────────────────────────────
# 7. CHECKPOINT SOUP → final teacher weights
# ────────────────────────────────────────────────────────────────
print(f"\nBuilding teacher soup from top-{len(top_ckpts)} epochs: {[f'{c[0]:.4f}' for c in top_ckpts]}")
soup = copy.deepcopy(top_ckpts[0][1])
for k in soup:
    soup[k] = torch.stack([c[1][k].float() for c in top_ckpts]).mean(0)
torch.save(soup, f"{CKPT_DIR}v7_teacher_soup.pth")
del top_ckpts; gc.collect(); torch.cuda.empty_cache()  # FIX-3/6: free soup copies
print(f"Soup saved → v7_teacher_soup.pth  |  Best solo AUC: {best_auc:.4f}")
print("Upload 'v7_teacher_soup.pth' to your Kaggle dataset before running students.")
