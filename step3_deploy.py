#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BƯỚC 3 — DEPLOY MODEL VÀO HỆ THỐNG
=====================================
Script này:
1. Kiểm tra model đã train (inference thử)
2. Copy weight vào backend/ai/models/
3. Cập nhật backend/ai/config.yaml
4. Gọi API reload để áp dụng ngay (không cần restart)

Chạy:
    python step3_deploy.py --model models_output/classifier.onnx --labels models_output/class_labels.json
    python step3_deploy.py --model models_output/yolo_uav/weights/best.onnx --type yolo
"""

import argparse
import json
import shutil
import time
from pathlib import Path


def test_classifier_onnx(model_path: Path, labels: list[str], img_size: int = 224):
    """Thử inference classifier ONNX với ảnh test ngẫu nhiên."""
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model_path),
               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name

        # Ảnh giả
        dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        t0    = time.perf_counter()
        out   = sess.run(None, {inp_name: dummy})[0][0]
        ms    = (time.perf_counter() - t0) * 1000

        # Softmax
        exp = np.exp(out - out.max())
        prob = exp / exp.sum()
        top  = prob.argsort()[::-1][:3]

        print(f"\n  ✓ Classifier ONNX OK ({ms:.1f} ms/inference)")
        print(f"  Output ({len(out)} classes): ", end="")
        for i in top:
            lbl = labels[i] if i < len(labels) else f"class_{i}"
            print(f"{lbl}={prob[i]:.3f}  ", end="")
        print()
        return True
    except Exception as e:
        print(f"  [!] Test classifier thất bại: {e}")
        return False


def test_yolo_onnx(model_path: Path):
    """Thử inference YOLO ONNX."""
    try:
        import numpy as np
        import onnxruntime as ort

        sess     = ort.InferenceSession(str(model_path),
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        inp_shape = sess.get_inputs()[0].shape
        h = inp_shape[2] if isinstance(inp_shape[2], int) else 640
        w = inp_shape[3] if isinstance(inp_shape[3], int) else 640

        dummy = np.random.randn(1, 3, h, w).astype(np.float32)
        t0    = time.perf_counter()
        out   = sess.run(None, {inp_name: dummy})
        ms    = (time.perf_counter() - t0) * 1000

        print(f"\n  ✓ YOLO ONNX OK ({ms:.1f} ms/inference)")
        print(f"  Output shape: {[o.shape for o in out]}")
        return True
    except Exception as e:
        print(f"  [!] Test YOLO thất bại: {e}")
        return False


def update_config(
    config_path: Path,
    model_type:  str,   # "classify" | "yolo"
    weight_name: str,   # tên file trong models/
    classes:     list[str],
    img_size:    int,
    engine:      str = "onnx",
):
    """Cập nhật backend/ai/config.yaml."""
    try:
        import yaml
    except ImportError:
        print("  [!] Cài yaml: pip install pyyaml")
        raise

    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    if model_type == "yolo":
        cfg.setdefault("detection", {})
        cfg["detection"].update({
            "engine":      engine,
            "weight":      weight_name,
            "conf_thresh": 0.45,
            "iou_thresh":  0.50,
            "img_size":    img_size,
            "classes":     classes,
            "device":      "cpu",
        })
    else:
        cfg.setdefault("classification", {})
        cfg["classification"].update({
            "engine":   engine,
            "weight":   weight_name,
            "img_size": img_size,
            "classes":  classes,
            "device":   "cpu",
        })

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"  ✓ Cập nhật config: {config_path}")


def reload_api(api_url: str = "http://localhost:8001"):
    """Gọi API reload để áp dụng model mới ngay."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{api_url}/api/ai/reload",
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
        print(f"  ✓ API reload: detection={data.get('detection')} "
              f"classification={data.get('classification')}")
        return True
    except Exception as e:
        print(f"  [!] API reload thất bại (server chưa chạy?): {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy model AI vào hệ thống")
    parser.add_argument("--model",  required=True,
                        help="Đường dẫn model ONNX hoặc .pt đã train")
    parser.add_argument("--labels", default="",
                        help="File class_labels.json (cho classifier)")
    parser.add_argument("--type",   choices=["classify", "yolo"], default="classify")
    parser.add_argument("--backend", default="../backend",
                        help="Đường dẫn thư mục backend/")
    parser.add_argument("--api-url", default="http://localhost:8001")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Kích thước ảnh input (224 cho classifier, 640 cho YOLO)")
    args = parser.parse_args()

    model_src = Path(args.model)
    backend   = Path(args.backend)
    models_dir = backend / "ai" / "models"
    config_path = backend / "ai" / "config.yaml"
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  BƯỚC 3 — DEPLOY MODEL")
    print(f"{'='*55}")
    print(f"  Model  : {model_src}")
    print(f"  Type   : {args.type}")
    print(f"  Backend: {backend}")

    if not model_src.exists():
        print(f"  [!] File không tồn tại: {model_src}")
        return

    # Load class labels
    classes = []
    if args.labels:
        lp = Path(args.labels)
        if lp.exists():
            with open(lp) as f:
                data = json.load(f)
            classes = data.get("classes", [])
            print(f"  Classes: {classes}")

    # Mặc định classes nếu không có labels file
    if not classes:
        if args.type == "yolo":
            classes = ["uav_signal", "controller_signal"]
        else:
            classes = ["no_uav", "uav"]

    # Xác định img_size từ model type
    img_size = args.img_size
    if args.type == "yolo":
        img_size = 640
    engine = "onnx" if model_src.suffix == ".onnx" else "torchscript"

    # ── Test inference trước khi deploy ──────────────────────────────────────
    print(f"\n  Test inference...")
    if args.type == "classify":
        ok = test_classifier_onnx(model_src, classes, img_size)
    else:
        ok = test_yolo_onnx(model_src)

    if not ok:
        print(f"  [!] Model test thất bại — dừng lại")
        return

    # ── Copy model vào backend/ai/models/ ────────────────────────────────────
    dst = models_dir / model_src.name
    shutil.copy2(model_src, dst)
    print(f"\n  ✓ Copy model → {dst}")

    # ── Cập nhật config.yaml ─────────────────────────────────────────────────
    update_config(config_path, args.type, model_src.name, classes, img_size, engine)

    # ── Reload API ────────────────────────────────────────────────────────────
    print(f"\n  Reload API ({args.api_url})...")
    reload_api(args.api_url)

    print(f"""
  {'='*55}
  ✓ DEPLOY HOÀN TẤT!

  Model đã sẵn sàng. Kiểm tra trên UI:
    - Mở http://localhost:5174
    - Bật quét, vào mục "AI nhận diện UAV"
    - Trạng thái detection/classification phải là "ready"
    - Nhấn 📷 để chụp waterfall và kiểm tra kết quả AI

  Nếu cần reload thủ công:
    POST {args.api_url}/api/ai/reload

  Kiểm tra trạng thái:
    GET  {args.api_url}/api/ai/status
  {'='*55}
""")


if __name__ == "__main__":
    main()
