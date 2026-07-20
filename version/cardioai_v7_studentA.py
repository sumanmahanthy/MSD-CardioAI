# ================================================================
# CARDIOAI V7 — STUDENT A: BiomedCLIP ViT-B/16 + CBAM
# Role in ensemble: Biomedical domain knowledge (PubMed pretrained)
# Overcomes V5 cons:
#   [CON-1] DISTILL_ALPHA=0.40 too high → 0.20 (BiomedCLIP dominates)
#   [CON-2] No warmup → 3-epoch linear warmup
#   [CON-3] DataParallel before .to(DEVICE) → fixed ordering
#   [CON-4] No prefetch_factor → prefetch_factor=2
#   [CON-5] Best single epoch → checkpoint soup top-3
#   [CON-6] Hard mining 2x multiplier unstable → capped at 1.5x
#   [CON-7] 0.93 booster unrealistic → 0.87 threshold
#   [CON-8] Teacher soup from V7 teacher → much stronger signal
# Output: saves val/test TTA predictions as .npy for ensemble
# ================================================================

import os, gc, glob, copy, json
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
from sklearn.metrics import roc_auc_score, f1_score
import open_clip, timm

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF auth OK.")
except Exception:
    print("No HF_TOKEN — anonymous download.")

# ────────────────────────────────────────────────────────────────
# 1. CONFIG
# ────────────────────────────────────────────────────────────────
DATA_DIR        = "/kaggle/input/datasets/organizations/nih-chest-xrays/data/"
META_PATH       = DATA_DIR + "Data_Entry_2017.csv"
CKPT_DIR        = "/kaggle/working/"
TEACHER_WEIGHTS = "/kaggle/input/mymodels/v7_teacher_soup.pth"   # Upload from teacher notebook

num_gpus        = torch.cuda.device_count()
BATCH_SIZE      = 32 * max(1, num_gpus)
EPOCHS          = 15
WARMUP_EPOCHS   = 3
LR_BACKBONE     = 1e-5
LR_HEAD         = 2e-4
DISTILL_ALPHA   = 0.20
T_TEMP          = 2.0
SOUP_TOP_K      = 3
RESUME_EPOCH    = 0
RESUME_BEST_AUC = 0.0
NUM_WORKERS     = 4
MODEL_TAG       = "A"        # used in output filenames
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DISEASES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
    "Fibrosis","Pleural_Thickening","Hernia"
]
NUM_CLASSES = len(DISEASES)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
print(f"STUDENT-{MODEL_TAG} | {DEVICE} | GPUs:{num_gpus} | Batch:{BATCH_SIZE}")

# ────────────────────────────────────────────────────────────────
# 2. DATA  (identical split contract: random_state=42)
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
train_pat, temp_pat = train_test_split(metadata["Patient ID"].unique(), test_size=0.20, random_state=42)
val_pat,   test_pat = train_test_split(temp_pat, test_size=0.50, random_state=42)
train_df = metadata[metadata["Patient ID"].isin(train_pat)].reset_index(drop=True)
val_df   = metadata[metadata["Patient ID"].isin(val_pat)].reset_index(drop=True)
test_df  = metadata[metadata["Patient ID"].isin(test_pat)].reset_index(drop=True)


class ChestXrayDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df
        aug = [
            T.Resize((224, 224), antialias=True),
            T.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.97, 1.03)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.RandomErasing(p=0.1, scale=(0.02, 0.08)),
        ]
        val = [T.Resize((224, 224), antialias=True)]
        self.transform = T.Compose([
            *(aug if augment else val),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.48145, 0.45782, 0.40821), (0.26862, 0.26130, 0.27577)),
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as img:
            t = self.transform(img.convert("RGB"))
        return t, torch.tensor(np.asarray(row["hard_labels"]), dtype=torch.float32)


def make_loader(df, aug, bs=BATCH_SIZE, persistent=True):
    # FIX-2/4: persistent=False for one-shot eval loaders to avoid worker leaks
    return DataLoader(ChestXrayDataset(df, aug), batch_size=bs, shuffle=aug,
                      num_workers=NUM_WORKERS, pin_memory=True,
                      prefetch_factor=2, persistent_workers=persistent)

train_loader = make_loader(train_df, True)
val_loader   = make_loader(val_df,   False)
# FIX-4: test_loader removed — tta_predict() creates its own one-shot loader


# ────────────────────────────────────────────────────────────────
# 3. STUDENT MODEL — BiomedCLIP ViT-B/16 + CBAM
# ────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(nn.Conv2d(ch, ch//r, 1, bias=False), nn.ReLU(),
                                  nn.Conv2d(ch//r, ch, 1, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()
    def forward(self, x):
        return x * self.sig(self.conv(torch.cat(
            [torch.mean(x,1,keepdim=True), torch.max(x,1,keepdim=True)[0]], 1)))

class CBAM(nn.Module):
    def __init__(self, ch): super().__init__(); self.ca=ChannelAttention(ch); self.sa=SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class ViTCBAMStudent(nn.Module):
    def __init__(self):
        super().__init__()
        clip, _, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        self.vit = clip.visual
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            f = self.vit.forward_features(dummy) if hasattr(self.vit, "forward_features") else self.vit(dummy)
        dim = f.shape[-1]
        print(f"   BiomedCLIP dim={dim}")
        self.cbam = CBAM(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.4), nn.Linear(dim, NUM_CLASSES))

    def forward(self, x):
        x = self.vit.forward_features(x) if hasattr(self.vit, "forward_features") else self.vit(x)
        if x.dim() == 2:
            return self.head(x)
        B, N, C = x[:, 1:].shape
        H = W = int(N**0.5)
        sp   = x[:, 1:].transpose(1,2).reshape(B, C, H, W)
        feat = F.adaptive_avg_pool2d(self.cbam(sp), 1).flatten(1)
        return self.head(feat)


# FIX: .to(DEVICE) BEFORE DataParallel
model = ViTCBAMStudent().to(DEVICE)
if num_gpus > 1:
    model = nn.DataParallel(model)


# ────────────────────────────────────────────────────────────────
# 4. TEACHER
# ────────────────────────────────────────────────────────────────
class TeacherEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = timm.create_model("tf_efficientnetv2_m", pretrained=False, num_classes=0)
        self.backbone = base
        self.head     = nn.Sequential(
            nn.BatchNorm1d(base.num_features), nn.Dropout(0.4),
            nn.Linear(base.num_features, 512), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(512, NUM_CLASSES)
        )
    def forward(self, x): return self.head(self.backbone(x))

# FIX-1: Teacher also wrapped in DataParallel — prevents GPU 0 OOM on multi-GPU
teacher     = TeacherEfficientNet().to(DEVICE)
if num_gpus > 1:
    teacher = nn.DataParallel(teacher)
HAS_TEACHER = True
try:
    sd  = torch.load(TEACHER_WEIGHTS, map_location=DEVICE, weights_only=False)
    sd  = {k.replace("module.", ""): v for k, v in sd.items()}
    teacher.load_state_dict(sd, strict=True)
    print(f"Teacher loaded from {TEACHER_WEIGHTS}")
except Exception as e:
    print(f"Teacher load failed: {e}  → pure student training.")
    HAS_TEACHER = False
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False


# ────────────────────────────────────────────────────────────────
# 5. LOSS
# ────────────────────────────────────────────────────────────────
def afl(logits, targets, gn=2, gp=1, clip=0.05, reduce=True):
    p     = torch.sigmoid(logits)
    p_neg = torch.clamp(p + clip, max=1.0)
    lp    = -(1-p)**gp * targets       * torch.log(torch.clamp(p,   min=1e-6))
    ln    = -(p_neg)**gn * (1-targets) * torch.log(torch.clamp(1-p, min=1e-6))
    loss  = lp + ln
    return loss.mean() if reduce else loss.mean(dim=1)


# ────────────────────────────────────────────────────────────────
# 6. OPTIMIZER + SCHEDULER
# ────────────────────────────────────────────────────────────────
mr = model.module if hasattr(model, "module") else model
optimizer = optim.AdamW([
    {"params": mr.vit.parameters(),  "lr": LR_BACKBONE},
    {"params": mr.cbam.parameters(), "lr": LR_HEAD},
    {"params": mr.head.parameters(), "lr": LR_HEAD},
], weight_decay=1e-3)

warmup = optim.lr_scheduler.LinearLR(optimizer, 0.1, 1.0, WARMUP_EPOCHS)  # FIX-6: 10% start
cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS - WARMUP_EPOCHS, 1e-6)
sched  = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [WARMUP_EPOCHS])
scaler = torch.amp.GradScaler("cuda")

best_auc    = RESUME_BEST_AUC
START_EPOCH = RESUME_EPOCH
top_ckpts   = []

if START_EPOCH > 0:
    c = torch.load(f"{CKPT_DIR}v7_student{MODEL_TAG}_ep{START_EPOCH:02d}.pth",
                   map_location=DEVICE, weights_only=False)
    model.load_state_dict(c["model"]); optimizer.load_state_dict(c["optimizer"])
    sched.load_state_dict(c["scheduler"])
    best_auc = c.get("best_auc", RESUME_BEST_AUC) if RESUME_BEST_AUC == 0.0 else RESUME_BEST_AUC
    print(f"Resumed ep{START_EPOCH} | best={best_auc:.4f}")


# ────────────────────────────────────────────────────────────────
# 7. TRAINING
# ────────────────────────────────────────────────────────────────
for epoch in range(START_EPOCH, EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS}")
    for imgs, lbls in pbar:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
            s_logits = model(imgs)
            t_soft   = torch.sigmoid(teacher(imgs) / T_TEMP).detach() if HAS_TEACHER else lbls

            # Hard sample mining — capped at 1.5x (stable)
            base  = afl(s_logits, lbls, reduce=False)
            h_loss = (base * (1.0 + 0.5*(base > base.mean()).float())).mean()

            # Temperature-scaled distillation
            s_loss = F.binary_cross_entropy_with_logits(s_logits / T_TEMP, t_soft)
            loss   = (1 - DISTILL_ALPHA) * h_loss + DISTILL_ALPHA * s_loss * (T_TEMP**2)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    sched.step()

    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
                vp.append(torch.sigmoid(model(imgs)).cpu().numpy())
            vt.append(lbls.numpy())
    auc = float(roc_auc_score(np.concatenate(vt), np.concatenate(vp), average="macro"))
    del vp, vt   # FIX-5: explicitly free val accumulation lists
    print(f"Ep {epoch+1}  Val AUC: {auc:.4f}")

    core = copy.deepcopy(model.module.state_dict() if hasattr(model,"module") else model.state_dict())
    top_ckpts.append((auc, core))
    top_ckpts.sort(key=lambda x:x[0], reverse=True)
    top_ckpts = top_ckpts[:SOUP_TOP_K]

    if auc > best_auc:
        best_auc = auc
        torch.save({"model": model.state_dict(), "auc": auc},
                   f"{CKPT_DIR}v7_student{MODEL_TAG}_best.pth")
        print(f"   BEST: {best_auc:.4f}")

    torch.save({"epoch": epoch+1, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": sched.state_dict(), "best_auc": best_auc, "auc": auc},
               f"{CKPT_DIR}v7_student{MODEL_TAG}_ep{epoch+1:02d}.pth")
    gc.collect()
    torch.cuda.empty_cache()   # FIX-3: flush CUDA allocator after each epoch


# ────────────────────────────────────────────────────────────────
# 8. CHECKPOINT SOUP
# ────────────────────────────────────────────────────────────────
print(f"\nSoup from top-{len(top_ckpts)}: {[f'{c[0]:.4f}' for c in top_ckpts]}")
soup = copy.deepcopy(top_ckpts[0][1])
for k in soup:
    soup[k] = torch.stack([c[1][k].float() for c in top_ckpts]).mean(0)
if hasattr(model, "module"): model.module.load_state_dict(soup)
else: model.load_state_dict(soup)
del top_ckpts; gc.collect(); torch.cuda.empty_cache()  # FIX-6: free 3x model copies
print("Soup model loaded.")


# ────────────────────────────────────────────────────────────────
# 9. GENERATE TTA PREDICTIONS → .npy for ensemble
# ────────────────────────────────────────────────────────────────
def tta_predict(df):
    loader = make_loader(df, False, persistent=False)  # FIX-2: no persistent workers for one-shot
    preds, targs = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="TTA predict"):
            imgs = imgs.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
                p0 = torch.sigmoid(model(imgs))
                p1 = torch.sigmoid(model(torch.flip(imgs, [3])))
                p2 = torch.sigmoid(model(F.interpolate(imgs, (256,256), mode="bilinear", align_corners=False)))
                p3 = torch.sigmoid(model(F.interpolate(imgs, (192,192), mode="bilinear", align_corners=False)))
            avg = ((p0+p1+p2+p3)/4).cpu().numpy()
            preds.append(avg); targs.append(lbls.numpy())
    return np.concatenate(preds), np.concatenate(targs)

print("\nGenerating val predictions...")
val_preds, Y_val = tta_predict(val_df)
np.save(f"{CKPT_DIR}preds_{MODEL_TAG}_val.npy",  val_preds)
np.save(f"{CKPT_DIR}Y_val.npy",                  Y_val)

print("Generating test predictions...")
test_preds, Y_test = tta_predict(test_df)
np.save(f"{CKPT_DIR}preds_{MODEL_TAG}_test.npy", test_preds)
np.save(f"{CKPT_DIR}Y_test.npy",                 Y_test)

solo_val  = float(roc_auc_score(Y_val,  val_preds,  average="macro"))
solo_test = float(roc_auc_score(Y_test, test_preds, average="macro"))
print(f"\nStudent-{MODEL_TAG} | Val AUC: {solo_val:.4f} | Test AUC: {solo_test:.4f}")
print(f"Saved: preds_{MODEL_TAG}_val.npy  preds_{MODEL_TAG}_test.npy  Y_val.npy  Y_test.npy")
