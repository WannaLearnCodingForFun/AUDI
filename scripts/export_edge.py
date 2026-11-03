#!/usr/bin/env python3
"""Export AUDI's networks for an edge device (upgrade 4.1): ONNX, then TensorRT FP16 / INT8.

  python scripts/export_edge.py --out models/edge              # ONNX for every available model
  python scripts/export_edge.py --out models/edge --trt fp16   # + TensorRT engines (on the Jetson: trtexec)
  python scripts/export_edge.py --bench models/edge             # fps, p95 latency, accuracy loss vs PyTorch

Exports
  backbone.onnx        DINOv2 ViT-S/14 (CLS + patch tokens), dynamic batch
  condition_head.onnx  per-asset-type MLP heads (one graph per type)
  reid_head.onnx       1.3 identity head (if trained)
  change_net.onnx      2.1 Siamese change network (if trained)
  detector.onnx        3.2 fine-tuned YOLO or YOLO-World with the configured prompts (Ultralytics export)
TensorRT: `trtexec --onnx=X.onnx --saveEngine=X.plan --fp16` (or `--int8 --calib=<cache>` built from
dataset/processed val images with `--calib-images`). Power: read `tegrastats` (VDD_IN) during --bench on the Jetson.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ml.config import cfg, get_nested, load_config, resolve_path  # noqa: E402


def export_module(module: torch.nn.Module, args, path: Path, inputs, outputs, dynamic) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    torch.onnx.export(module, args, str(path), input_names=inputs, output_names=outputs, dynamic_axes=dynamic, opset_version=17,
                      dynamo=False)
    return path


class BackboneWrap(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x):
        f = self.m.forward_features(x)
        return f["x_norm_clstoken"], f["x_norm_patchtokens"]


def export_all(config, out: Path, image_size: int) -> dict:
    done = {}
    from ml.embedding_model import AssetEmbeddingModel

    emb = AssetEmbeddingModel.from_config(config)
    bb = emb.backbone
    if hasattr(bb, "model") and hasattr(bb.model, "forward_features"):
        x = torch.randn(1, 3, image_size, image_size)
        done["backbone"] = str(export_module(BackboneWrap(bb.model.cpu()), (x,), out / "backbone.onnx", ["image"], ["cls", "patches"],
                                             {"image": {0: "batch"}, "cls": {0: "batch"}, "patches": {0: "batch"}}))
    ck = resolve_path(cfg(config, "condition_classifier", "checkpoint"))
    if ck.exists():
        from ml.condition_classifier import MultiHeadConditionClassifier

        clf = MultiHeadConditionClassifier.load(ck, emb)
        for t in clf.label_maps:
            head = clf.net.heads[t]
            done[f"condition_head_{t}"] = str(export_module(head.cpu(), (torch.randn(1, emb.embedding_dim),), out / f"condition_head_{t}.onnx",
                                                             ["embedding"], ["logits"], {"embedding": {0: "batch"}, "logits": {0: "batch"}}))
    from ml.reid_head import IdentityEncoder

    enc = IdentityEncoder.from_config(config, emb.embedding_dim)
    if enc.head is not None:
        done["reid_head"] = str(export_module(enc.head, (torch.randn(1, emb.embedding_dim),), out / "reid_head.onnx", ["embedding"], ["identity"],
                                              {"embedding": {0: "batch"}, "identity": {0: "batch"}}))
    cn = resolve_path(get_nested(config, "change_mask", "checkpoint", default="") or "models/_none")
    if cn.is_file():
        from ml.siamese_change import SiameseChangeNet

        net = SiameseChangeNet.load(cn)
        x = torch.randn(1, 3, 128, 128)
        done["change_net"] = str(export_module(net, (x, x), out / "change_net.onnx", ["reference", "current"], ["logits"],
                                               {"reference": {0: "b", 2: "h", 3: "w"}, "current": {0: "b", 2: "h", 3: "w"}}))
    try:
        dcfg = cfg(config, "detection")
        w = resolve_path(dcfg.get("finetuned_weights", ""))
        from ultralytics import YOLO

        if w.is_file():
            exported = YOLO(str(w)).export(format="onnx", imgsz=int(dcfg["image_size"]))
        else:
            from ml.detector import YoloWorldDetector

            det = YoloWorldDetector({**config, "detection": {**dcfg, "onnx_export": False}})
            exported = det.model.export(format="onnx", imgsz=int(dcfg["image_size"]))
        shutil.copy(exported, out / "detector.onnx")
        done["detector"] = str(out / "detector.onnx")
    except Exception as exc:  # noqa: BLE001 - detector weights are optional here
        done["detector"] = f"skipped: {exc}"
    return done


def build_trt(out: Path, precision: str, calib_images: Path = None) -> dict:
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    res = {}
    for onnx in sorted(out.glob("*.onnx")):
        cmd = [trtexec, f"--onnx={onnx}", f"--saveEngine={onnx.with_suffix('.plan')}"]
        cmd += ["--fp16"] if precision == "fp16" else ["--int8", "--fp16"] + ([f"--calib={out / 'int8_calib.cache'}"] if calib_images else [])
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            res[onnx.name] = "ok" if p.returncode == 0 else p.stderr[-400:]
        except FileNotFoundError:
            res[onnx.name] = "trtexec not found (run on the Jetson / a TensorRT container)"
    return res


def bench(out: Path, config, runs: int = 100) -> dict:
    import onnxruntime as ort

    providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    rep = {"providers": providers}
    for onnx in sorted(out.glob("*.onnx")):
        sess = ort.InferenceSession(str(onnx), providers=providers)
        feeds = {}
        for i in sess.get_inputs():
            shape = [1 if not isinstance(d, int) else d for d in i.shape]
            if len(shape) == 4:
                shape[2] = shape[2] if isinstance(i.shape[2], int) else 224
                shape[3] = shape[3] if isinstance(i.shape[3], int) else 224
            feeds[i.name] = np.random.rand(*shape).astype(np.float32)
        for _ in range(5):
            sess.run(None, feeds)
        lat = []
        for _ in range(runs):
            t = time.perf_counter()
            sess.run(None, feeds)
            lat.append((time.perf_counter() - t) * 1000)
        rep[onnx.name] = {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)), "fps": 1000.0 / float(np.mean(lat))}
    # accuracy loss: backbone embeddings ONNX vs PyTorch on the same random batch
    bb = out / "backbone.onnx"
    if bb.exists():
        from ml.embedding_model import AssetEmbeddingModel

        emb = AssetEmbeddingModel.from_config(config)
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            ref = BackboneWrap(emb.backbone.model.cpu())(x)[0].numpy()
        got = ort.InferenceSession(str(bb), providers=providers).run(None, {"image": x.numpy()})[0]
        cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
        rep["backbone_cosine_vs_torch_min"] = float(cos.min())
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=resolve_path("models/edge"))
    ap.add_argument("--trt", choices=["fp16", "int8"], default=None)
    ap.add_argument("--calib-images", type=Path, default=None)
    ap.add_argument("--bench", type=Path, default=None)
    args = ap.parse_args(argv)
    config = load_config()
    if args.bench:
        r = bench(args.bench, config)
    else:
        r = {"exported": export_all(config, args.out, int(cfg(config, "model", "image_size")))}
        if args.trt:
            r["tensorrt"] = build_trt(args.out, args.trt, args.calib_images)
    p = resolve_path("reports/edge/export.json" if not args.bench else "reports/edge/bench.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r, indent=2))
    print(json.dumps(r, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
