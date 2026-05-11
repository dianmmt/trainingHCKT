#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_onnx.py — Export model PyTorch sang ONNX sau khi train
==============================================================
Dùng khi torch.onnx.export bị lỗi thiếu onnxscript trong quá trình train.

Cài trước:
    pip install onnxscript onnx

Chạy:
    python export_onnx.py \\
        --weights models_binary_efficientnetv2_s_640px/best.pt \\
        --labels  models_binary_efficientnetv2_s_640px/class_labels.json \\
        --out     models_binary_efficientnetv2_s_640px/classifier_binary_640px.onnx
"""

import argparse
import json
from pathlib import Path


def build_model_for_export(model_key: str, num_classes: int, img_size: int):
    """Build model độc lập, không cần import train_your_model."""
    import timm

    # Danh sách tên model theo timm version
    CANDIDATES = {
        "efficientnetv2_s": [
            "tf_efficientnetv2_s.in21ft1k",
            "tf_efficientnetv2_s.in1k",
            "efficientnetv2_s",
            "tf_efficientnetv2_s",
        ],
        "efficientnetv2_m": [
            "tf_efficientnetv2_m.in21ft1k",
            "tf_efficientnetv2_m.in1k",
            "efficientnetv2_m",
        ],
        "swin_s": [
            "swin_small_patch4_window7_224.ms_in22k_ft_in1k",
            "swin_small_patch4_window7_224",
        ],
        "swin_b": [
            "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
            "swin_base_patch4_window7_224",
        ],
        "convnext_b": [
            "convnext_base.fb_in22k_ft_in1k",
            "convnext_base",
        ],
    }

    candidates = CANDIDATES.get(model_key, [model_key])
    available  = set(timm.list_models(pretrained=False))

    chosen = None
    for c in candidates:
        if c in available:
            chosen = c
            break
    if chosen is None:
        # Tìm bất kỳ variant nào có tên gần đúng
        matches = timm.list_models(f"*{model_key.split('_')[0]}*")
        chosen  = sorted(matches)[0] if matches else model_key
        print(f"  [!] Dùng fallback: {chosen}")

    print(f"  timm model: {chosen}")
    try:
        model = timm.create_model(chosen, pretrained=False,
                                  num_classes=num_classes, img_size=img_size)
    except TypeError:
        model = timm.create_model(chosen, pretrained=False, num_classes=num_classes)
    return model


def export(args):
    import torch
    import timm

    # Load metadata
    with open(args.labels, encoding="utf-8") as f:
        info = json.load(f)

    model_key  = info["model"]
    num_classes = info["num_classes"]
    img_size   = info["img_size"]

    print(f"  Model     : {model_key}")
    print(f"  Classes   : {info['classes']}")
    print(f"  Img size  : {img_size}×{img_size}")

    # Rebuild model (không cần pretrained — sẽ load weights từ best.pt)
    model = build_model_for_export(model_key, num_classes, img_size)

    # Load weights
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  Weights   : {args.weights}")

    dummy = torch.randn(1, 3, img_size, img_size)
    out_path = Path(args.out)

    # ── Thử legacy exporter trước ─────────────────────────────────────────────
    try:
        torch.onnx.export(
            model, dummy, str(out_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        print(f"  ✓ ONNX (legacy) → {out_path}")

    except ModuleNotFoundError as e:
        if "onnxscript" not in str(e):
            raise
        print(f"  onnxscript chưa cài → pip install onnxscript")
        print(f"  Thử dynamo exporter...")

        try:
            prog = torch.onnx.dynamo_export(
                model, dummy,
                export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
            )
            prog.save(str(out_path))
            print(f"  ✓ ONNX (dynamo) → {out_path}")

        except Exception as e2:
            # Fallback: TorchScript
            print(f"  dynamo thất bại: {e2}")
            ts_path = out_path.with_suffix(".torchscript")
            scripted = torch.jit.trace(model, dummy)
            scripted.save(str(ts_path))
            print(f"  ✓ TorchScript → {ts_path}")
            print(f"  → Cài onnxscript rồi chạy lại để có ONNX")
            return

    # Verify ONNX
    try:
        import onnx
        m = onnx.load(str(out_path))
        onnx.checker.check_model(m)
        print(f"  ✓ ONNX verified OK")
    except ImportError:
        print(f"  (onnx chưa cài, bỏ qua verify)")

    # Test inference với onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out_path),
                                    providers=["CUDAExecutionProvider",
                                               "CPUExecutionProvider"])
        import numpy as np
        x    = dummy.numpy()
        out  = sess.run(None, {"input": x})[0]
        print(f"  ✓ ORT inference OK — output shape: {out.shape}")
    except ImportError:
        print(f"  (onnxruntime chưa cài, bỏ qua test inference)")
    except Exception as e:
        print(f"  ⚠ ORT test: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch → ONNX")
    parser.add_argument("--weights", required=True, help="best.pt")
    parser.add_argument("--labels",  required=True, help="class_labels.json")
    parser.add_argument("--out",     required=True, help="output .onnx path")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  EXPORT ONNX")
    print(f"{'='*50}")
    export(args)
    print()


if __name__ == "__main__":
    main()