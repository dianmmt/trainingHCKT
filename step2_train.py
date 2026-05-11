#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BƯỚC 2 — TRAINING MODEL AI
============================
Hỗ trợ 2 pipeline:

  A) DETECTION (YOLO) — phát hiện vị trí tín hiệu UAV trong waterfall
     python step2_train.py --mode yolo --data dataset --epochs 100

  B) CLASSIFICATION (EfficientNetV2-S) — phân loại UAV/no_uav hoặc loại UAV
     python step2_train.py --mode classify --data dataset --epochs 50

YÊU CẦU GPU:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install ultralytics timm
    pip install albumentations matplotlib seaborn scikit-learn

Chạy kiểm tra GPU:
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


# ─── Kiểm tra GPU ────────────────────────────────────────────────────────────

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU: {name} ({mem:.1f} GB VRAM)")
            return "cuda"
        else:
            print("  [!] Không tìm thấy GPU → dùng CPU (rất chậm!)")
            return "cpu"
    except ImportError:
        print("  [!] PyTorch chưa cài. Chạy: pip install torch torchvision")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE A: CLASSIFICATION — EfficientNetV2-S (KHUYẾN NGHỊ)
# ═══════════════════════════════════════════════════════════════════════════════

def train_classifier(args):
    """
    Fine-tune EfficientNetV2-S trên dataset waterfall.

    EfficientNetV2-S được chọn vì:
    - Chính xác cao với dữ liệu spectrogram (CNN + attention)
    - ~21M params — nhỏ, nhanh, deploy trên CPU được
    - Đã train trên ImageNet → transfer learning tốt
    - Tốc độ inference ~8ms/ảnh trên CPU
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    import timm   # pip install timm

    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    device = torch.device(check_gpu())
    data_dir = Path(args.data)

    # ── Data transforms ──────────────────────────────────────────────────────
    IMG_SIZE = 224   # EfficientNetV2-S input

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Augmentation cho spectrogram — KHÔNG dùng flip ngang/dọc
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # shift nhỏ
        transforms.ColorJitter(brightness=0.2, contrast=0.2),        # thay đổi độ sáng
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),    # noise mô phỏng
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir / "val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(data_dir / "test",  transform=val_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"\n  Lớp phân loại: {class_names}")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Class weights để xử lý imbalanced dataset
    class_counts = np.bincount([y for _, y in train_ds.samples])
    class_weights = torch.FloatTensor(1.0 / class_counts).to(device)
    class_weights = class_weights / class_weights.sum() * num_classes

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model: EfficientNetV2-S ───────────────────────────────────────────────
    # timm >= 0.9 đổi tên: "efficientnetv2_s" → "tf_efficientnetv2_s.in21ft1k"
    # Script tự tìm tên đúng tương ứng với phiên bản timm đang cài
    CANDIDATE_MODELS = [
        "tf_efficientnetv2_s.in21ft1k",   # timm >= 0.9 (khuyến nghị, ImageNet-21k fine-tuned)
        "tf_efficientnetv2_s.in1k",        # timm >= 0.9 (ImageNet-1k)
        "efficientnetv2_s",                # timm < 0.9 (cũ)
        "tf_efficientnetv2_s",             # fallback
    ]
    available = set(timm.list_models(pretrained=True))
    model_name = None
    for candidate in CANDIDATE_MODELS:
        if candidate in available:
            model_name = candidate
            break
    if model_name is None:
        # Tìm bất kỳ efficientnetv2_s nào có pretrained
        matches = [m for m in timm.list_models("*efficientnetv2_s*", pretrained=True)]
        if matches:
            model_name = sorted(matches)[0]
        else:
            # Fallback sang EfficientNet-B3 nếu không có V2
            model_name = "efficientnet_b3.ra2_in1k"
            print(f"  [!] Không tìm thấy EfficientNetV2-S, dùng {model_name}")

    print(f"\n  Khởi tạo {model_name} (pretrained ImageNet)...")
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    # Xác định tên lớp head để freeze/unfreeze đúng
    # EfficientNetV2: head "classifier", một số biến thể: "head.fc", "head"
    head_keywords = ("classifier", "head")

    # Fine-tune: freeze backbone, chỉ train head (vài epoch đầu)
    for name, param in model.named_parameters():
        if not any(k in name for k in head_keywords):
            param.requires_grad = False

    model = model.to(device)

    # ── Loss, Optimizer, Scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1: chỉ train head — dùng head_keywords đã xác định ở trên
    head_params = [p for n, p in model.named_parameters()
                   if any(k in n for k in head_keywords) and p.requires_grad]
    if not head_params:
        # Nếu không tìm được head, unfreeze toàn bộ ngay
        print("  [!] Không tìm được head params, unfreeze toàn bộ từ đầu")
        for p in model.parameters():
            p.requires_grad = True
        head_params = list(model.parameters())

    optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    WARMUP_EPOCHS = min(5, args.epochs // 5)   # phase 1: train head
    UNFREEZE_AT   = WARMUP_EPOCHS              # phase 2: unfreeze toàn bộ

    print(f"\n  {'='*50}")
    print(f"  Bắt đầu training ({args.epochs} epochs, device={device})")
    print(f"  Phase 1 (head only): {WARMUP_EPOCHS} epochs")
    print(f"  Phase 2 (full finetune): {args.epochs - WARMUP_EPOCHS} epochs")
    print(f"  {'='*50}")

    for epoch in range(args.epochs):

        # Unfreeze toàn bộ backbone ở phase 2
        if epoch == UNFREEZE_AT:
            print(f"\n  [Epoch {epoch+1}] Unfreeze toàn bộ backbone → finetune")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=1e-4, weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - UNFREEZE_AT, eta_min=1e-6,
            )
        elif epoch > UNFREEZE_AT:
            scheduler.step()

        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        tr_loss = train_loss / train_total
        tr_acc  = train_correct / train_total * 100
        va_loss = val_loss / val_total
        va_acc  = val_correct / val_total * 100
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"  Epoch [{epoch+1:3d}/{args.epochs}] "
              f"train={tr_acc:.1f}%/{tr_loss:.4f}  "
              f"val={va_acc:.1f}%/{va_loss:.4f}  "
              f"({elapsed:.1f}s)")

        # Lưu best model
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_dir / "best_classifier.pt")
            print(f"         ✓ Best model saved (val_acc={va_acc:.2f}%)")

    # ── Đánh giá trên test set ───────────────────────────────────────────────
    print(f"\n  {'='*50}")
    print(f"  ĐÁNH GIÁ TRÊN TEST SET")
    print(f"  {'='*50}")
    model.load_state_dict(torch.load(out_dir / "best_classifier.pt"))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out  = model(imgs)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes-1)))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Test Set")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    print(f"\n  Confusion matrix → {out_dir}/confusion_matrix.png")

    # Training curves
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss"); ax1.legend(); ax1.set_xlabel("Epoch")
    ax2.plot(history["train_acc"], label="Train"); ax2.plot(history["val_acc"], label="Val")
    ax2.set_title("Accuracy (%)"); ax2.legend(); ax2.set_xlabel("Epoch")
    fig2.tight_layout(); fig2.savefig(out_dir / "training_curves.png", dpi=150)

    # ── Export ONNX ───────────────────────────────────────────────────────────
    print(f"\n  Export ONNX...")
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    onnx_path = out_dir / "classifier.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"  ✓ ONNX → {onnx_path}")

    # Lưu class labels
    labels_path = out_dir / "class_labels.json"
    with open(labels_path, "w") as f:
        json.dump({"classes": class_names, "num_classes": num_classes}, f, indent=2)

    print(f"\n  ✓ Training xong!")
    print(f"  Best val accuracy : {best_val_acc:.2f}%")
    print(f"  Files tại         : {out_dir}/")
    print(f"    best_classifier.pt  ← PyTorch weights")
    print(f"    classifier.onnx     ← ONNX (deploy không cần PyTorch)")
    print(f"    class_labels.json   ← Tên các lớp")
    print(f"\n  → Tiếp theo: python step3_deploy.py --model {out_dir}/classifier.onnx")


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE B: DETECTION — YOLOv8s / YOLOv11n
# ═══════════════════════════════════════════════════════════════════════════════

def train_yolo(args):
    """
    Train YOLOv8s để detect bounding box tín hiệu UAV trong waterfall.
    Cần chuẩn bị annotation theo format YOLO (xywh normalized).

    Cấu trúc dataset YOLO:
        yolo_dataset/
        ├── images/train/*.png
        ├── images/val/*.png
        ├── images/test/*.png
        ├── labels/train/*.txt   ← annotation YOLO format
        ├── labels/val/*.txt
        └── dataset.yaml
    """
    from ultralytics import YOLO

    check_gpu()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data) / "dataset.yaml"
    if not data_yaml.exists():
        # Tự tạo dataset.yaml nếu chưa có
        content = f"""
path: {Path(args.data).absolute()}
train: images/train
val:   images/val
test:  images/test

nc: 2
names: ['uav_signal', 'controller_signal']
"""
        data_yaml.write_text(content.strip())
        print(f"  Đã tạo: {data_yaml}")
        print(f"  [!] Cần annotation YOLO format trong labels/ trước khi train!")

    print(f"\n  Model: YOLOv8s (phát hiện tín hiệu UAV)")
    model = YOLO("yolov8s.pt")   # tự download pretrained

    results = model.train(
        data    = str(data_yaml),
        epochs  = args.epochs,
        imgsz   = 640,
        batch   = args.batch,
        device  = 0 if check_gpu() == "cuda" else "cpu",
        project = str(out_dir),
        name    = "yolo_uav",
        patience= 20,        # early stopping
        save    = True,
        plots   = True,
        # Augmentation tốt cho spectrogram
        fliplr  = 0.0,       # KHÔNG flip ngang (mất nghĩa vật lý)
        flipud  = 0.0,       # KHÔNG flip dọc
        mosaic  = 0.5,       # mosaic augmentation
        translate = 0.05,
        scale   = 0.1,
        hsv_h   = 0.0,       # KHÔNG thay đổi màu sắc (màu có nghĩa vật lý)
        hsv_s   = 0.1,
        hsv_v   = 0.2,
    )

    # Export best model
    best = out_dir / "yolo_uav" / "weights" / "best.pt"
    if best.exists():
        yolo_best = YOLO(str(best))
        yolo_best.export(format="onnx", imgsz=640, simplify=True)
        print(f"\n  ✓ ONNX → {best.parent}/best.onnx")

    print(f"\n  ✓ Training YOLO xong!")
    print(f"  → Tiếp theo: python step3_deploy.py --model {best}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train AI model UAV detection")
    parser.add_argument("--mode",   choices=["classify", "yolo"], default="classify",
                        help="Pipeline: classify (EfficientNetV2-S) hoặc yolo (YOLOv8s)")
    parser.add_argument("--data",   default="dataset",
                        help="Thư mục dataset (output của step1)")
    parser.add_argument("--out",    default="models_output",
                        help="Thư mục lưu model đã train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch",  type=int, default=16)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  BƯỚC 2 — TRAINING MODEL AI")
    print(f"{'='*55}")
    print(f"  Mode   : {args.mode}")
    print(f"  Data   : {args.data}")
    print(f"  Epochs : {args.epochs}")
    print(f"  Batch  : {args.batch}")

    if args.mode == "classify":
        train_classifier(args)
    else:
        train_yolo(args)


if __name__ == "__main__":
    main()
