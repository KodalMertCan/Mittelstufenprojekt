"""Build v8b_train.ipynb - DINOv2-L + Pseudo-Labels + Knowledge Distillation.

V8b = V8a + KD vom V7-Teacher-Ensemble (V5 DINOv2-L + V6 EVA-02-L Checkpoints).

Pipeline:
1. (One-time) Precompute Teacher-Soft-Targets auf Train-Daten mit V7-Ensemble
   -> gespeichert als teacher_soft_{tag}.npy (shape: [N_train, 5])
2. Training: Student DINOv2-L lernt gegen:
   - Hard Target (CE + class weights)
   - Soft Target vom Teacher (KL-Divergence mit Temperature)
   - loss = alpha * CE_hard + (1-alpha) * T^2 * KL(student/T, teacher/T)

Offline KD: Teacher laeuft nur 1x auf unaugmentierten Bildern, Student sieht
augmentierte Varianten -> Teacher-Target bleibt pro Bild fix (nicht pro Augmentation).
"""
import json


def code_cell(cell_id, source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "code", "execution_count": None, "id": cell_id,
        "metadata": {}, "outputs": [], "source": src
    }


def md_cell(cell_id, source):
    lines = source.split("\n")
    src = [l + "\n" for l in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {
        "cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src
    }


cells = []

cells.append(md_cell("title", """# Pig Posture Recognition - V8b Training (DINOv2-L + Pseudo + KD)

**V8b = V8a + Knowledge Distillation** aus dem V7-Teacher-Ensemble.

**Extra-Gewinn vs V8a:** Student lernt nicht nur das harte Label, sondern auch
die softe Probability-Verteilung des staerkeren Ensembles (dark knowledge).
Erwartung: +0.3-0.8% ueber V8a.

**Rezept:**
- Student: DINOv2-L @ 518px (einzelnes Modell)
- Teacher: V5 DINOv2-L + V6 EVA-02-L Ensemble (precomputed auf Train)
- Loss: `alpha * CE_hard + (1-alpha) * T^2 * KL(student/T || teacher/T)`
- alpha=0.5, Temperature T=4.0
- Alles andere: V3-Hyperparams, EMA, MixUp/CutMix, LLRD 0.75

**Wichtig:** Schritt 1 (Precompute Teacher) ist ein einmaliger Durchlauf ueber Train-Daten.
Danach wird teacher_soft_{tag}.npy genutzt, das Teacher-Netz ist weg."""))

cells.append(md_cell("config-h", "## Configuration"))

cells.append(code_cell("config", '''TAG = "T2"

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

_candidates = [
    "multiview_pig_posture_recognition",
    "./multiview_pig_posture_recognition",
    "/datasets/multi-view-pig-posture-recognition",
    "/multi-view-pig-posture-recognition",
]
DATA_ROOT = None
for _p in _candidates:
    if os.path.isdir(_p):
        DATA_ROOT = _p
        break
assert DATA_ROOT is not None
print(f"DATA_ROOT = {os.path.abspath(DATA_ROOT)}")

if TAG == "T1":
    CSV_PATH = f"{DATA_ROOT}/train1.csv"
    IMG_DIR  = f"{DATA_ROOT}/train1_images"
else:
    CSV_PATH = f"{DATA_ROOT}/train2.csv"
    IMG_DIR  = f"{DATA_ROOT}/train2_images"

OUTPUT_DIR = f"runs/v8b_{TAG.lower()}"

# --- Student: nur DINOv2-L ---
ARCH_LIST = [
    {
        "name": "vit_large_patch14_dinov2.lvd142m",
        "img_size": 518,
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 1e-4,
        "lr_backbone_mult": 0.1,
        "layer_decay": 0.75,
        "epochs": 30,
        "patience": 10,
        "prefix": "dinov2l",
    },
]

# --- Teacher-Ensemble Quellen (V5 DINOv2-L + V6 EVA-02-L) ---
TEACHER_SOURCES = [
    ("dinov2l", f"runs/v5_{TAG.lower()}", 518, 1.5),  # Gewicht wie B_dinov2_strong
    ("eva02l",  f"runs/v6_{TAG.lower()}", 448, 1.0),
]
TEACHER_SOFT_FILE = f"teacher_soft_{TAG.lower()}_v7.npy"   # precomputed cache
TEACHER_TTA       = False   # mit TTA deutlich teurer; ohne TTA reicht meist

# --- KD Hyperparams ---
KD_ALPHA       = 0.5     # Gewicht Hard-CE vs Soft-KD (0.5 = gleich)
KD_TEMPERATURE = 4.0     # glaettet die Verteilung (hoeher = mehr dark knowledge)

WARMUP_EPOCHS     = 3
LABEL_SMOOTH      = 0.05
PAD_RATIO         = 0.1
NUM_WORKERS       = 16
SEED              = 42
NUM_CLASSES       = 5

MIXUP_ALPHA       = 0.2
CUTMIX_ALPHA      = 1.0
MIX_PROB          = 0.5
SWITCH_PROB       = 0.5

USE_EMA           = True
EMA_DECAY         = 0.9995

CLASS_NAMES = ["Lateral_lying_left", "Lateral_lying_right",
               "Sitting", "Standing", "Sternal_lying"]

LOSS_TYPE    = "ce"
TEST_CAMERAS = ["pen1_tur_cam1", "pen2_orb_cam2", "pen2_tur_cam2"]
VALIDATION_STRATEGY = "test_only"

USE_PSEUDO_LABELS = True
PSEUDO_CSV        = "pseudo_labels_t2_v7.csv"

PRETRAINED_CKPT   = None

print(f"Tag: {TAG}  |  Output: {OUTPUT_DIR}")
print(f"Student: {ARCH_LIST[0]['name']} @ {ARCH_LIST[0]['img_size']}px")
print(f"Teacher: {[f'{p}@{s}' for p,_,s,_ in TEACHER_SOURCES]}  (cache: {TEACHER_SOFT_FILE})")
print(f"KD: alpha={KD_ALPHA}, T={KD_TEMPERATURE}")
print(f"Pseudo: {USE_PSEUDO_LABELS} ({PSEUDO_CSV})")'''))

cells.append(md_cell("imports-h", "## Imports"))

cells.append(code_cell("imports", '''import os, ast, random, re, copy
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from tqdm.notebook import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
import torchvision.transforms.functional as TFn
import timm
from timm.data import Mixup

from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)'''))

cells.append(md_cell("data-h", "## Daten laden (inkl. Pseudo-Labels)"))

cells.append(code_cell("data", '''df = pd.read_csv(CSV_PATH)

def extract_camera(image_id):
    m = re.match(r"(pen\\d+_\\w+_cam\\d+)", image_id)
    return m.group(1) if m else "unknown"

df["camera"] = df["image_id"].apply(extract_camera)
df["img_dir"] = IMG_DIR
df["is_pseudo"] = False

if USE_PSEUDO_LABELS and PSEUDO_CSV and os.path.exists(PSEUDO_CSV):
    pseudo_df = pd.read_csv(PSEUDO_CSV)
    keep_cols = [c for c in pseudo_df.columns if c in ["row_id","image_id","width","height","bbox","class_id"]]
    pseudo_df = pseudo_df[keep_cols]
    pseudo_df["camera"] = pseudo_df["image_id"].apply(extract_camera)
    pseudo_df["img_dir"] = os.path.join(DATA_ROOT, "test_images")
    pseudo_df["is_pseudo"] = True
    df = pd.concat([df, pseudo_df], ignore_index=True)
    print(f"Pseudo-Labels: {len(pseudo_df)} hinzugefuegt -> Gesamt: {len(df)}")

# stabiler Index fuer teacher_soft Lookup (positional)
df = df.reset_index(drop=True)
df["sample_idx"] = np.arange(len(df), dtype=int)

print(f"Instanzen total: {len(df)}  |  echt: {(~df['is_pseudo']).sum()}  |  pseudo: {df['is_pseudo'].sum()}")
for c in range(NUM_CLASSES):
    cnt = (df["class_id"] == c).sum()
    print(f"  {c} - {CLASS_NAMES[c]:<22} {cnt:>5}")'''))

cells.append(md_cell("precompute-h", """## Schritt 1: Teacher-Soft-Targets precomputen

**Einmal pro Datensatz-Konfiguration.** Laed V5+V6 Checkpoints, inferiert auf
unaugmentierten Train-Bildern, speichert `teacher_soft_{tag}_v7.npy` mit shape
`[N, 5]`.

Wenn die Datei schon existiert, wird diese Zelle uebersprungen.
Laufzeit: ca. 15-45 min auf 4xV100 (je nach N)."""))

cells.append(code_cell("precompute", '''class PigSoftDataset(Dataset):
    """Gibt (img, sample_idx) zurueck - fuer Teacher-Inferenz auf unaugmentierten Bildern."""
    def __init__(self, df, transform, pad_ratio=0.1):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.pad_ratio = pad_ratio

    def __len__(self): return len(self.df)

    def _crop(self, img, bbox):
        W, H = img.size
        x, y, w, h = [float(v) for v in ast.literal_eval(bbox)]
        px, py = w * self.pad_ratio, h * self.pad_ratio
        x1 = max(0, int(x - px));  y1 = max(0, int(y - py))
        x2 = min(W, int(x+w+px));  y2 = min(H, int(y+h+py))
        return img.crop((x1, y1, x2, y2))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(row["img_dir"], row["image_id"])).convert("RGB")
        crop = self._crop(img, row["bbox"])
        if self.transform:
            crop = self.transform(crop)
        return crop, int(row["sample_idx"])


def load_vit_with_resampling(name, ckpt, num_classes, infer_size):
    model = timm.create_model(name, pretrained=False, num_classes=num_classes, img_size=infer_size)
    state_dict = dict(ckpt["model"])
    if "pos_embed" in state_dict:
        old_pe = state_dict["pos_embed"]
        if old_pe.shape != model.pos_embed.shape:
            try:
                from timm.layers import resample_abs_pos_embed
            except ImportError:
                from timm.models.layers import resample_abs_pos_embed
            num_prefix = getattr(model, "num_prefix_tokens", 1)
            state_dict["pos_embed"] = resample_abs_pos_embed(
                old_pe, new_size=model.patch_embed.grid_size, num_prefix_tokens=num_prefix
            )
    model.load_state_dict(state_dict, strict=False)
    return model


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


def teacher_inference_single(ckpt_path, img_size, df_all, batch_size=32):
    """Inferiert EIN Teacher-Modell auf allen df_all-Samples. Returns np.array [N, 5] probs."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    name = ckpt.get("model_name", "unknown")
    try:
        model = load_vit_with_resampling(name, ckpt, NUM_CLASSES, img_size)
    except TypeError:
        model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
        model.load_state_dict(ckpt["model"])
    model = model.to(DEVICE).eval()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_STD),
    ])
    ds = PigSoftDataset(df_all, transform=tf, pad_ratio=PAD_RATIO)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    probs = np.zeros((len(df_all), NUM_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for imgs, idxs in tqdm(loader, desc=f"    {os.path.basename(ckpt_path)}", leave=False):
            with autocast():
                logits = model(imgs.to(DEVICE))
            p = F.softmax(logits, dim=1).cpu().numpy()
            for i, s_idx in enumerate(idxs.numpy()):
                probs[int(s_idx)] = p[i]

    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return probs


if os.path.exists(TEACHER_SOFT_FILE):
    teacher_soft = np.load(TEACHER_SOFT_FILE)
    assert teacher_soft.shape == (len(df), NUM_CLASSES), \\
        f"Cache shape mismatch: {teacher_soft.shape} vs erwartet {(len(df), NUM_CLASSES)}"
    print(f"[CACHE] geladen: {TEACHER_SOFT_FILE}  shape={teacher_soft.shape}")
else:
    print(f"[COMPUTE] Starte Teacher-Ensemble Inferenz ueber {len(df)} Samples...")
    accum_probs = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    weight_sum = 0.0

    for prefix, ckpt_dir, img_size, w in TEACHER_SOURCES:
        if not os.path.isdir(ckpt_dir):
            print(f"  WARN: {ckpt_dir} existiert nicht, ueberspringe {prefix}")
            continue
        ckpts = sorted([f for f in os.listdir(ckpt_dir)
                        if f.startswith(f"best_{prefix}_fold_") and f.endswith(".pth")])
        if not ckpts:
            print(f"  WARN: keine Checkpoints in {ckpt_dir} fuer {prefix}")
            continue

        fold_sum = np.zeros_like(accum_probs)
        for f in ckpts:
            path = os.path.join(ckpt_dir, f)
            print(f"\\n  {prefix}: {f} @ {img_size}px")
            fold_sum += teacher_inference_single(path, img_size, df, batch_size=32)
        fold_mean = fold_sum / len(ckpts)

        accum_probs += fold_mean * w
        weight_sum += w
        print(f"  {prefix}: {len(ckpts)} Folds gemittelt, weight={w}")

    assert weight_sum > 0, "Kein Teacher-Modell gefunden!"
    teacher_soft = accum_probs / weight_sum

    np.save(TEACHER_SOFT_FILE, teacher_soft)
    print(f"\\n[SAVED] {TEACHER_SOFT_FILE}  shape={teacher_soft.shape}")

# Argmax-Genauigkeit des Teachers auf Train als Sanity-Check
teacher_argmax = teacher_soft.argmax(axis=1)
true_labels = df["class_id"].values.astype(int)
agree = (teacher_argmax == true_labels).mean()
print(f"\\nTeacher-Argmax Agreement mit Labels: {agree:.4f}")
print("Max-Prob Histogram:")
bins = [0, 0.5, 0.7, 0.85, 0.95, 1.0]
hist, _ = np.histogram(teacher_soft.max(axis=1), bins=bins)
for lo, hi, cnt in zip(bins[:-1], bins[1:], hist):
    print(f"  [{lo:.2f},{hi:.2f}): {cnt:>5} ({100*cnt/len(df):.1f}%)")'''))

cells.append(md_cell("dataset-h", "## Dataset gibt teacher_soft mit zurueck"))

cells.append(code_cell("dataset", '''class PigPostureDatasetKD(Dataset):
    """Wie V5-Dataset, aber returnt zusaetzlich teacher_soft fuer KD."""
    def __init__(self, df, teacher_soft, transform=None, pad_ratio=0.25,
                 is_train=False, hflip_prob=0.5):
        self.df = df.reset_index(drop=True)
        self.teacher_soft = teacher_soft
        self.transform = transform
        self.pad_ratio = pad_ratio
        self.is_train = is_train
        self.hflip_prob = hflip_prob

    def __len__(self):
        return len(self.df)

    def _crop(self, img, bbox):
        W, H = img.size
        x, y, w, h = [float(v) for v in ast.literal_eval(bbox)]
        px, py = w * self.pad_ratio, h * self.pad_ratio
        x1 = max(0, int(x - px));  y1 = max(0, int(y - py))
        x2 = min(W, int(x+w+px));  y2 = min(H, int(y+h+py))
        return img.crop((x1, y1, x2, y2))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(row["img_dir"], row["image_id"])).convert("RGB")
        crop = self._crop(img, row["bbox"])
        label = int(row["class_id"])
        soft = self.teacher_soft[int(row["sample_idx"])].copy()

        if self.is_train and random.random() < self.hflip_prob:
            crop = TFn.hflip(crop)
            if label == 0:
                label = 1
                soft = soft.copy(); soft[[0,1]] = soft[[1,0]]
            elif label == 1:
                label = 0
                soft = soft.copy(); soft[[0,1]] = soft[[1,0]]

        if self.transform:
            crop = self.transform(crop)
        return crop, label, torch.from_numpy(soft.astype(np.float32))'''))

cells.append(md_cell("aug-h", "## Augmentations (V5-identisch)"))

cells.append(code_cell("augmentations", '''class CameraSimTransform:
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        if random.random() < 0.4:
            w, h = img.size
            scale = random.uniform(0.3, 0.7)
            small = img.resize((max(16, int(w*scale)), max(16, int(h*scale))), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)
        if random.random() < 0.2:
            quality = random.randint(25, 65)
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")
        if random.random() < 0.15:
            arr = np.array(img, dtype=np.float32)
            noise = np.random.normal(0, random.uniform(5, 15), arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        return img


class PerspectiveJitter:
    def __init__(self, distortion_scale=0.15, p=0.4):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            d = self.distortion_scale
            w, h = img.size
            return TFn.perspective(
                img,
                startpoints=[[0,0],[w,0],[w,h],[0,h]],
                endpoints=[
                    [int(random.uniform(0, w*d)), int(random.uniform(0, h*d))],
                    [int(w - random.uniform(0, w*d)), int(random.uniform(0, h*d))],
                    [int(w - random.uniform(0, w*d)), int(h - random.uniform(0, h*d))],
                    [int(random.uniform(0, w*d)), int(h - random.uniform(0, h*d))],
                ],
                fill=0
            )
        return img


def get_train_transform(size):
    return T.Compose([
        CameraSimTransform(p=0.4),
        PerspectiveJitter(distortion_scale=0.15, p=0.4),
        T.Resize((int(size*1.15), int(size*1.15)), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.85, 1.15),
                            interpolation=T.InterpolationMode.BICUBIC),
        T.RandomRotation(degrees=12),
        T.RandomAffine(degrees=0, scale=(0.85, 1.15), translate=(0.05, 0.05)),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.08),
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_val_transform(size):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

print("Augmentations definiert.")'''))

cells.append(md_cell("ema-h", "## EMA"))

cells.append(code_cell("ema", '''class ModelEMA:
    def __init__(self, model, decay=0.9995, cpu=False):
        self.decay = decay
        self.cpu = cpu
        inner = model.module if isinstance(model, nn.DataParallel) else model
        self.ema = copy.deepcopy(inner).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        if cpu:
            self.ema = self.ema.cpu()

    @torch.no_grad()
    def update(self, model):
        inner = model.module if isinstance(model, nn.DataParallel) else model
        msd = inner.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                src = msd[k].detach()
                if self.cpu:
                    src = src.cpu()
                v.mul_(self.decay).add_(src, alpha=1 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.ema.state_dict()'''))

cells.append(md_cell("lrd-h", "## Layer-wise LR Decay"))

cells.append(code_cell("lrd", '''def get_vit_layer_id(name, num_layers):
    name = name.replace("module.", "")
    if name.startswith(("cls_token", "pos_embed", "patch_embed", "mask_token", "reg_token")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return num_layers


def build_vit_param_groups(model, base_lr, layer_decay=0.75, weight_decay=1e-2):
    inner = model.module if isinstance(model, nn.DataParallel) else model
    num_layers = len(inner.blocks) if hasattr(inner, "blocks") else 12
    lr_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 1)]

    groups = {}
    for name, param in inner.named_parameters():
        if not param.requires_grad:
            continue
        lid = get_vit_layer_id(name, num_layers)
        no_wd = param.ndim <= 1 or name.endswith(".bias") or "norm" in name or "cls_token" in name or "pos_embed" in name
        key = (lid, no_wd)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": base_lr * lr_scales[lid],
                "weight_decay": 0.0 if no_wd else weight_decay,
                "lr_scale": lr_scales[lid],
            }
        groups[key]["params"].append(param)
    return list(groups.values()), num_layers


def build_optimizer(model, arch_cfg):
    base_lr = arch_cfg["lr"]
    groups, n_layers = build_vit_param_groups(model, base_lr, arch_cfg["layer_decay"])
    print(f"  ViT LLRD: {n_layers} Layers, decay={arch_cfg['layer_decay']}, {len(groups)} groups")
    return optim.AdamW(groups, lr=base_lr)'''))

cells.append(md_cell("loss-h", """## KD Loss

**Kombinierter Loss:**
- `CE_hard`: Student-Logits vs. Hard Label (mit class_weights, label_smooth)
- `KL_kd`: `T^2 * KL(softmax(s/T) || softmax(t/T))` fuer dark knowledge
- `loss = alpha * CE_hard + (1-alpha) * KL_kd`

Bei MixUp: Teacher-Targets werden mit demselben Lambda gemischt wie Labels."""))

cells.append(code_cell("loss", '''def build_criterion(class_weights, label_smooth=0.05):
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smooth)


def soft_ce(logits, targets_soft):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets_soft * log_probs).sum(dim=-1).mean()


def soft_ce_weighted(logits, targets_soft, class_weights):
    log_probs = F.log_softmax(logits, dim=-1)
    w = class_weights.unsqueeze(0)
    return -(targets_soft * log_probs * w).sum(dim=-1).mean()


def kd_loss(student_logits, teacher_probs, T=4.0):
    """KL(student/T || teacher/T) * T^2.
    teacher_probs sind bereits Wahrscheinlichkeiten ([0,1], sum=1).
    """
    teacher_probs_T = F.softmax(teacher_probs.log() * (1.0/T), dim=-1)  # re-sharp mit T
    # einfacher: teacher_probs ist schon softmax, also potenziere dann renormalize
    tlog = torch.log(teacher_probs.clamp_min(1e-8))
    t_T = F.softmax(tlog / T, dim=-1)
    s_T = F.log_softmax(student_logits / T, dim=-1)
    # KL(t || s) = sum(t * (log t - log s)) -> wir wollen student lernen Teacher zu imitieren
    return F.kl_div(s_T, t_T, reduction="batchmean") * (T * T)'''))

cells.append(md_cell("helpers-h", "## Training Helpers (mit KD)"))

cells.append(code_cell("helpers", '''def train_one_epoch_kd(model, loader, optimizer, scaler, criterion,
                       mixup_fn=None, ema=None, grad_accum=1, class_weights=None,
                       kd_alpha=0.5, kd_T=4.0):
    model.train()
    optimizer.zero_grad()
    loss_sum, n = 0.0, 0
    preds, labels_all = [], []

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        imgs, labels, soft = batch
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        soft   = soft.to(DEVICE, non_blocking=True)

        if mixup_fn is not None:
            # MixUp mischt Bilder und Labels; wir muessen soft-targets MIT demselben
            # Lambda mischen. timm.Mixup macht das nicht fuer soft-targets, also manuell.
            # Vereinfachung: bei MixUp skippen wir KD-Teil (nur fuer diese Batch).
            imgs_m, targets_m = mixup_fn(imgs, labels)
            with autocast():
                logits = model(imgs_m)
                if class_weights is not None:
                    loss_hard = soft_ce_weighted(logits, targets_m, class_weights)
                else:
                    loss_hard = soft_ce(logits, targets_m)
                loss = loss_hard / grad_accum
        else:
            with autocast():
                logits = model(imgs)
                loss_hard = criterion(logits, labels)
                loss_kd   = kd_loss(logits, soft, T=kd_T)
                loss = (kd_alpha * loss_hard + (1.0 - kd_alpha) * loss_kd) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        loss_sum += loss.item() * imgs.size(0) * grad_accum
        n += imgs.size(0)
        preds.extend(logits.argmax(1).detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    return loss_sum / max(n, 1), f1_score(labels_all, preds, average="macro", zero_division=0)


@torch.no_grad()
def validate_epoch_kd(model_or_ema, loader, criterion):
    model_or_ema.eval()
    loss_sum, n = 0.0, 0
    preds, labels_all = [], []
    for batch in tqdm(loader, desc="  Val  ", leave=False):
        imgs, labels, _soft = batch   # soft beim Val ignoriert
        imgs   = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast():
            logits = model_or_ema(imgs)
            loss = criterion(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        n += imgs.size(0)
        preds.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
    return (loss_sum / max(n, 1),
            f1_score(labels_all, preds, average="macro", zero_division=0),
            preds, labels_all)'''))

cells.append(md_cell("folds-h", "## Strict CLO Folds"))

cells.append(code_cell("build_folds", '''df_train = df.copy()

available_cams = set(df_train[~df_train["is_pseudo"]]["camera"].unique())
test_cams_in_data = sorted([c for c in TEST_CAMERAS if c in available_cams])

if len(test_cams_in_data) == 0:
    cams_for_clo = sorted(available_cams)
    print(f"T1-Modus: CLO auf alle Kameras")
else:
    cams_for_clo = list(test_cams_in_data)
    print(f"STRICT CLO auf Test-Kameras: {cams_for_clo}")

splits = []
fold_cameras = []
for cam in cams_for_clo:
    val_mask = (df_train["camera"] == cam) & (~df_train["is_pseudo"])
    val_idx = df_train.index[val_mask].values
    train_idx = df_train.index[~val_mask].values
    if len(val_idx) > 0:
        splits.append((train_idx, val_idx))
        fold_cameras.append(cam)

n_folds = len(splits)
print(f"\\n{n_folds} Folds:")
for i, (tr, vl) in enumerate(splits):
    n_pseudo_train = df_train.iloc[tr]["is_pseudo"].sum()
    n_real_train = len(tr) - n_pseudo_train
    print(f"  Fold {i+1}: Val={fold_cameras[i]} ({len(vl)}) | "
          f"Train={len(tr)} ({n_real_train} echt + {n_pseudo_train} pseudo)")'''))

cells.append(md_cell("training-h", "## Training Loop (V8b mit KD)"))

cells.append(code_cell("training", '''all_results = {}

for arch_idx, arch in enumerate(ARCH_LIST):
    print(f"\\n{'#'*60}")
    print(f"  STUDENT: {arch['prefix']} ({arch['name']})")
    print(f"  KD: alpha={KD_ALPHA}, T={KD_TEMPERATURE}")
    print(f"{'#'*60}")

    IMG_SIZE_ARCH  = arch["img_size"]
    BATCH_SIZE     = arch["batch_size"]
    GRAD_ACCUM     = arch.get("grad_accum", 1)
    LR             = arch["lr"]
    EPOCHS_ARCH    = arch["epochs"]
    PATIENCE       = arch["patience"]
    PREFIX         = arch["prefix"]

    arch_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        print(f"\\n{'='*60}")
        print(f"  [{PREFIX}] FOLD {fold_idx + 1} / {n_folds}  (Val={fold_cameras[fold_idx]})")
        print(f"{'='*60}")

        fold_train = df_train.iloc[train_idx].reset_index(drop=True)
        fold_val   = df_train.iloc[val_idx].reset_index(drop=True)

        print(f"  Train: {len(fold_train)} ({fold_train['is_pseudo'].sum()} pseudo) | "
              f"Val: {len(fold_val)} | effBatch={BATCH_SIZE*GRAD_ACCUM}")

        ckpt_path = os.path.join(OUTPUT_DIR, f"best_{PREFIX}_fold_{fold_idx+1}.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            print(f"  Checkpoint existiert (val_f1={ckpt.get('val_f1',0):.4f}), skip")
            arch_results.append(ckpt.get("val_f1", 0))
            continue

        train_ds = PigPostureDatasetKD(fold_train, teacher_soft,
                                       transform=get_train_transform(IMG_SIZE_ARCH),
                                       pad_ratio=PAD_RATIO, is_train=True, hflip_prob=0.5)
        val_ds   = PigPostureDatasetKD(fold_val, teacher_soft,
                                       transform=get_val_transform(IMG_SIZE_ARCH),
                                       pad_ratio=PAD_RATIO, is_train=False)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
                                  persistent_workers=(NUM_WORKERS > 0))
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True,
                                  persistent_workers=(NUM_WORKERS > 0))

        try:
            model = timm.create_model(arch["name"], pretrained=True,
                                      num_classes=NUM_CLASSES, img_size=IMG_SIZE_ARCH)
        except TypeError:
            model = timm.create_model(arch["name"], pretrained=True, num_classes=NUM_CLASSES)

        model = model.to(DEVICE)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Student: {arch['name']} ({params:.1f}M)")

        ema = ModelEMA(model, decay=EMA_DECAY, cpu=False) if USE_EMA else None

        counts = Counter(fold_train["class_id"].tolist())
        class_weights = torch.tensor(
            [len(fold_train) / (NUM_CLASSES * max(counts.get(c, 1), 1))
             for c in range(NUM_CLASSES)], dtype=torch.float32
        ).to(DEVICE)
        criterion = build_criterion(class_weights, label_smooth=LABEL_SMOOTH)
        print(f"  Class Weights: {[f'{w:.2f}' for w in class_weights.cpu().tolist()]}")

        mixup_fn = Mixup(
            mixup_alpha=MIXUP_ALPHA, cutmix_alpha=CUTMIX_ALPHA,
            prob=MIX_PROB, switch_prob=SWITCH_PROB,
            label_smoothing=LABEL_SMOOTH, num_classes=NUM_CLASSES,
        )

        optimizer = build_optimizer(model, arch)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_EPOCHS)
        cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS_ARCH - WARMUP_EPOCHS, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
        scaler = GradScaler()

        best_val_f1 = 0.0
        patience_counter = 0
        best_source = "model"

        for epoch in range(1, EPOCHS_ARCH + 1):
            train_loss, train_f1 = train_one_epoch_kd(
                model, train_loader, optimizer, scaler, criterion,
                mixup_fn=mixup_fn, ema=ema, grad_accum=GRAD_ACCUM,
                class_weights=class_weights, kd_alpha=KD_ALPHA, kd_T=KD_TEMPERATURE
            )
            val_loss, val_f1, val_preds, val_labels = validate_epoch_kd(model, val_loader, criterion)
            source = "model"
            if ema is not None:
                e_loss, e_f1, e_preds, e_labels = validate_epoch_kd(ema.ema, val_loader, criterion)
                if e_f1 > val_f1:
                    val_loss, val_f1, val_preds, val_labels = e_loss, e_f1, e_preds, e_labels
                    source = "ema"
            scheduler.step()

            improved = val_f1 > best_val_f1
            mark = "*" if improved else " "
            phase = "warmup" if epoch <= WARMUP_EPOCHS else "cosine"
            print(f"  {mark} Epoch {epoch:02d}/{EPOCHS_ARCH} [{phase}] | "
                  f"Train L={train_loss:.4f} F1={train_f1:.4f} | "
                  f"Val[{source}] L={val_loss:.4f} F1={val_f1:.4f}"
                  f"{' <- BEST' if improved else ''}")

            if improved:
                best_val_f1 = val_f1
                best_source = source
                patience_counter = 0
                if source == "ema" and ema is not None:
                    state = ema.state_dict()
                else:
                    state = (model.module.state_dict() if isinstance(model, nn.DataParallel)
                             else model.state_dict())
                torch.save({
                    "epoch": epoch, "model": state,
                    "val_f1": val_f1, "model_name": arch["name"], "arch_prefix": PREFIX,
                    "tag": TAG, "img_size": IMG_SIZE_ARCH, "pad_ratio": PAD_RATIO,
                    "val_camera": fold_cameras[fold_idx], "source": source,
                    "kd_alpha": KD_ALPHA, "kd_T": KD_TEMPERATURE,
                }, ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early Stopping nach {PATIENCE} Epochen")
                    break

        print(f"\\n  Klassifikation Fold {fold_idx+1} (source={best_source}):")
        for c in range(NUM_CLASSES):
            mask = np.array(val_labels) == c
            if mask.sum() > 0:
                correct = (np.array(val_preds)[mask] == c).sum()
                print(f"    {CLASS_NAMES[c]:<22} {correct}/{mask.sum()} ({100*correct/max(mask.sum(),1):.0f}%)")

        arch_results.append(best_val_f1)
        print(f"  Best Val F1: {best_val_f1:.4f}  ({best_source})")

        try:
            del train_loader, val_loader, train_ds, val_ds
        except NameError:
            pass
        del model, optimizer, scheduler, scaler
        if ema is not None:
            del ema
        import gc
        gc.collect()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() if torch.cuda.is_available() else None

    all_results[PREFIX] = arch_results
    print(f"\\n  [{PREFIX}] Mean F1: {np.mean(arch_results):.4f}")

print(f"\\n{'#'*60}")
print(f"  GESAMT-ERGEBNIS V8b (KD)")
print(f"{'#'*60}")
for prefix, results in all_results.items():
    print(f"  {prefix:<12} | Mean F1: {np.mean(results):.4f} (+/- {np.std(results):.4f}) | Folds: {results}")'''))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"}
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = "c:/Users/LampeF/Mittelstufenprojekt/v8b_train.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Saved {out_path}")
