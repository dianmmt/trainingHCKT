#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train model với dataset thực tế của bạn.

Chạy sau khi đã chạy prepare_your_dataset.py:

    # Binary (detect UAV hay không):
    python train_your_model.py --data E:\bb60c\dataset\binary --mode binary --epochs 50

    # Multiclass (phân loại từng loại):
    python train_your_model.py --data E:\bb60c\dataset\multiclass --mode multiclass --epochs 100
"""

import argparse
import json
import time
from pathlib import Path


def check_env():
    """Kiểm tra môi trường trước khi train."""
    errors = []
    try:
        import torch
        gpu = torch.cuda.is_available()
        if gpu:
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU: {name} ({mem:.1f} GB)")
        else:
            print("  ⚠ Không có GPU — train bằng CPU (chậm hơn ~20x)")
    except ImportError:
        errors.append("torch  →  pip install torch torchvision "
                      "--index-url https://download.pytorch.org/whl/cu121")
    try:
        import timm
        print(f"  ✓ timm {timm.__version__}")
    except ImportError:
        errors.append("timm   →  pip install timm")
    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        errors.append("scikit-learn  →  pip install scikit-learn")

    if errors:
        print("\n  [!] Thiếu thư viện:")
        for e in errors:
            print(f"      pip install {e.split('→')[1].strip()}")
        raise SystemExit(1)


def get_class_info(data_dir: Path) -> tuple[list[str], dict[str, int]]:
    """Đọc các class từ thư mục train/."""
    train_dir = data_dir / "train"
    classes   = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    counts    = {c: len(list((train_dir / c).glob("*.png"))) for c in classes}
    return classes, counts


def train(args):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import timm
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes, counts = get_class_info(data_dir)
    num_classes     = len(classes)

    print(f"\n  {'─'*55}")
    print(f"  Lớp phân loại ({num_classes} lớp):")
    for i, c in enumerate(classes):
        n = counts[c]
        bar = "█" * min(30, n // 10)
        print(f"    [{i}] {c:30s} {n:5d} ảnh  {bar}")

    # ── Transforms ────────────────────────────────────────────────────────────
    IMG_SIZE = 224
    # Augmentation riêng cho spectrogram:
    # - KHÔNG flip (mất nghĩa vật lý tần số)
    # - Shift nhỏ mô phỏng lệch tần
    # - Brightness/contrast mô phỏng thay đổi noise floor
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),              # shift ngẫu nhiên ±8px
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=val_tf)

    # Tính class weights để xử lý imbalanced (có lớp ít ảnh hơn)
    sample_counts = np.array([counts[c] for c in classes], dtype=np.float32)
    weights       = 1.0 / sample_counts
    weights       = weights / weights.sum() * num_classes
    class_weights = torch.FloatTensor(weights).to(device)

    num_workers = 0 if device.type == "cpu" else 4
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=num_workers)

    print(f"\n  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ────────────────────────────────────────────────────────────────
    CANDIDATE_MODELS = [
        "tf_efficientnetv2_s.in21ft1k",   # timm >= 0.9
        "tf_efficientnetv2_s.in1k",
        "efficientnetv2_s",                # timm < 0.9
        "tf_efficientnetv2_s",
    ]
    available = set(timm.list_models(pretrained=True))
    model_name = None
    for c in CANDIDATE_MODELS:
        if c in available:
            model_name = c
            break
    if model_name is None:
        matches = [m for m in timm.list_models("*efficientnetv2_s*", pretrained=True)]
        model_name = sorted(matches)[0] if matches else "efficientnet_b3.ra2_in1k"
        print(f"  [!] Dùng fallback: {model_name}")

    print(f"\n  Khởi tạo {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)

    head_keywords = ("classifier", "head")

    # Phase 1: freeze backbone
    for name, param in model.named_parameters():
        if not any(k in name for k in head_keywords):
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    WARMUP = min(5, max(2, args.epochs // 10))

    # Phase 1 optimizer — chỉ train head params
    head_params = [p for n, p in model.named_parameters()
                   if any(k in n for k in head_keywords) and p.requires_grad]
    if not head_params:
        # Nếu không tìm được head, unfreeze toàn bộ
        print("  [!] Không tìm thấy head params — unfreeze toàn bộ ngay từ đầu")
        for p in model.parameters(): p.requires_grad = True
        head_params = list(model.parameters())
    opt = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)

    best_val_acc = 0.0
    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])

    print(f"  Epochs: {args.epochs}  |  Batch: {args.batch}  |  Device: {device}")
    print(f"  Phase 1 (head):   epoch 1–{WARMUP}")
    print(f"  Phase 2 (full):   epoch {WARMUP+1}–{args.epochs}")
    print(f"  {'─'*55}")

    for epoch in range(1, args.epochs + 1):

        # Chuyển phase 2: unfreeze toàn bộ backbone
        if epoch == WARMUP + 1:
            for p in model.parameters(): p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=args.epochs - WARMUP, eta_min=1e-7,
            )
            print(f"  [Epoch {epoch}] Unfreeze backbone — finetune toàn bộ")

        if epoch > WARMUP + 1:
            sched.step()

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_correct = tr_total = 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss    += loss.item() * imgs.size(0)
            tr_correct += (out.argmax(1) == labels).sum().item()
            tr_total   += imgs.size(0)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        va_loss = va_correct = va_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                va_loss    += loss.item() * imgs.size(0)
                va_correct += (out.argmax(1) == labels).sum().item()
                va_total   += imgs.size(0)

        tr_acc = tr_correct / tr_total * 100
        va_acc = va_correct / va_total * 100
        history["train_loss"].append(tr_loss / tr_total)
        history["val_loss"].append(va_loss / va_total)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        star = " ✓ best" if va_acc > best_val_acc else ""
        print(f"  Epoch [{epoch:3d}/{args.epochs}]  "
              f"train {tr_acc:5.1f}%  val {va_acc:5.1f}%  "
              f"({time.time()-t0:.1f}s){star}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_dir / "best.pt")

    # ── Test set evaluation ───────────────────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  ĐÁNH GIÁ TRÊN TEST SET")
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=classes, digits=3))

    # Confusion matrix
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(6, num_classes + 2), max(5, num_classes + 1)))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes,
                cmap="Blues", ax=ax, linewidths=0.5)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế")
    ax.set_title(f"Confusion Matrix — Test ({args.mode})")
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"  → {out_dir}/confusion_matrix.png")

    # Training curves
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch"); ax1.grid(True, alpha=0.3)
    ax2.plot(history["train_acc"], label="Train"); ax2.plot(history["val_acc"], label="Val")
    ax2.set_title("Accuracy (%)"); ax2.legend(); ax2.set_xlabel("Epoch"); ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "training_curves.png", dpi=150)
    print(f"  → {out_dir}/training_curves.png")

    # ── Export ONNX ───────────────────────────────────────────────────────────
    print(f"\n  Export ONNX...")
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    onnx_path = out_dir / f"classifier_{args.mode}.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"  ✓ ONNX → {onnx_path}")

    # Lưu class info
    info = {"classes": classes, "num_classes": num_classes,
            "best_val_acc": round(best_val_acc, 4),
            "mode": args.mode, "img_size": IMG_SIZE}
    with open(out_dir / "class_labels.json", "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"""
  {'='*55}
  ✓ TRAINING HOÀN TẤT

  Best val accuracy : {best_val_acc:.2f}%
  Files tại         : {out_dir}/
    best.pt                   ← PyTorch weights
    classifier_{args.mode}.onnx  ← ONNX (deploy)
    class_labels.json         ← Tên các lớp
    confusion_matrix.png      ← Ma trận nhầm lẫn
    training_curves.png       ← Đồ thị loss/accuracy

  → Tiếp theo:
    python step3_deploy.py \\
      --model   {onnx_path} \\
      --labels  {out_dir}/class_labels.json \\
      --backend ..\\backend
  {'='*55}
""")


def main():
    parser = argparse.ArgumentParser(description="Train model waterfall UAV")
    parser.add_argument("--data",   required=True,
                        help="Thư mục dataset (output của prepare_your_dataset.py)")
    parser.add_argument("--mode",   choices=["binary", "multiclass"],
                        default="binary",
                        help="binary: uav/no_uav | multiclass: từng loại riêng")
    parser.add_argument("--out",    default="",
                        help="Thư mục lưu model (mặc định: models_<mode>/)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch",  type=int, default=16,
                        help="Batch size (giảm nếu hết VRAM GPU)")
    args = parser.parse_args()

    if not args.out:
        args.out = f"models_{args.mode}"

    print(f"\n{'='*55}")
    print(f"  TRAIN MODEL UAV WATERFALL")
    print(f"{'='*55}")
    print(f"  Mode     : {args.mode}")
    print(f"  Data     : {args.data}")
    print(f"  Output   : {args.out}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch}")

    check_env()
    train(args)


if __name__ == "__main__":
    main()
