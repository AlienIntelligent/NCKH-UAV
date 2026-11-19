#!/usr/bin/env python3
"""
run_local.py — pipeline NCKH-UAV LOCAL (GPU), THIẾT KẾ ĐÚNG hộp đen.

Surrogate (YOLOv8n) train trên DOTA-only  → nhóm tối ưu.
Target    (RT-DETR) train trên DIOR-only  → nhóm giữ kín (khác kiến trúc + khác dữ liệu).
Patch tối ưu bằng DE trên surrogate/DOTA, rồi đo TRANSFER trên target/DIOR.
Có ĐỐI CHỨNG patch ngẫu nhiên cùng diện tích.

Stage: prepare → surrogate → target → attack

Ví dụ:
    python local/run_local.py
    python local/run_local.py --stages attack
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ───────────────────────── helpers ─────────────────────────
def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def banner(txt):
    print("\n" + "=" * 70 + f"\n  {txt}\n" + "=" * 70)


def _dset_root(cfg, which):      # which = 'dota' | 'dior'
    return Path(cfg["paths"]["work_dir"]) / f"yolo_{which}"


def _yaml(cfg, which):
    return str(_dset_root(cfg, which) / "data.yaml")


def _val_dirs(cfg, which):
    r = _dset_root(cfg, which)
    return str(r / "val" / "images"), str(r / "val" / "labels")


def _build_dataset(pairs, root, split_ratio, seed):
    """Split (img,lbl) pairs into train/val under root, hard-link, write data.yaml."""
    for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    k = int(len(pairs) * split_ratio)
    for subset, ps in [("train", pairs[:k]), ("val", pairs[k:])]:
        for ip, lp in ps:
            link_or_copy(ip, str(root / subset / "images" / Path(ip).name))
            link_or_copy(lp, str(root / subset / "labels" / Path(lp).name))
    (root / "data.yaml").write_text(
        f"path: {root.as_posix()}\ntrain: train/images\nval: val/images\n"
        f'nc: 2\nnames: ["aircraft", "vehicle"]\n'
    )
    return k, len(pairs) - k


def _collect(img_dir, lbl_dir):
    pairs = []
    for p in Path(img_dir).iterdir():
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        lp = Path(lbl_dir) / (p.stem + ".txt")
        if lp.exists() and lp.stat().st_size > 5:
            pairs.append((str(p), str(lp)))
    return pairs


# ───────────────────── stage 1: prepare ─────────────────────
def stage_prepare(cfg):
    import cv2
    from PIL import Image
    from tqdm import tqdm

    Image.MAX_IMAGE_PIXELS = None
    P = cfg["paths"]
    work = Path(P["work_dir"])
    seed = cfg["environment"]["seed"]
    sr = cfg["train_common"]["split_ratio"]

    # ── DOTA slicing ──
    DOTA_CLASSES = {"plane": 0, "small-vehicle": 1, "large-vehicle": 1}
    SLICE, OVL = 1024, 0.2
    d_img = work / "DOTA" / "processed" / "images"; d_img.mkdir(parents=True, exist_ok=True)
    d_lbl = work / "DOTA" / "processed" / "labels"; d_lbl.mkdir(parents=True, exist_ok=True)
    raw = sorted(glob.glob(os.path.join(P["dota_images"], "*.png")))
    banner(f"PREPARE — DOTA slicing ({len(raw)} images)")

    def parse_dota(lp):
        anns = []
        with open(lp) as f:
            for line in f:
                q = line.strip().split()
                if len(q) < 9 or q[8] not in DOTA_CLASSES:
                    continue
                c = list(map(float, q[:8])); xs, ys = c[0::2], c[1::2]
                anns.append((min(xs), min(ys), max(xs), max(ys), DOTA_CLASSES[q[8]]))
        return anns

    n_slices = 0
    for ip in tqdm(raw, desc="DOTA"):
        stem = Path(ip).stem
        lp = os.path.join(P["dota_labels"], stem + ".txt")
        if not os.path.isfile(lp):
            continue
        anns = parse_dota(lp)
        if not anns:
            continue
        try:
            im = np.array(Image.open(ip))
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif im.shape[2] == 4:
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2BGR)
            else:
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        except Exception:
            continue
        h, w = im.shape[:2]; stride = int(SLICE * (1 - OVL))
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                x1, y1 = min(x0 + SLICE, w), min(y0 + SLICE, h)
                x0a, y0a = max(0, x1 - SLICE), max(0, y1 - SLICE)
                sa = []
                for (ax1, ay1, ax2, ay2, cid) in anns:
                    ix1, iy1 = max(ax1, x0a), max(ay1, y0a)
                    ix2, iy2 = min(ax2, x1), min(ay2, y1)
                    iw, ih = ix2 - ix1, iy2 - iy1
                    if iw > 10 and ih > 10 and (iw * ih) / max((ax2 - ax1) * (ay2 - ay1), 1) > 0.3:
                        sw, sh = x1 - x0a, y1 - y0a
                        cx = ((ix1 - x0a) + (ix2 - x0a)) / 2 / sw
                        cy = ((iy1 - y0a) + (iy2 - y0a)) / 2 / sh
                        sa.append(f"{cid} {cx:.6f} {cy:.6f} {iw / sw:.6f} {ih / sh:.6f}")
                if not sa:
                    continue
                nm = f"{stem}_{x0a}_{y0a}"
                cv2.imwrite(str(d_img / f"{nm}.jpg"), im[y0a:y1, x0a:x1])
                with open(str(d_lbl / f"{nm}.txt"), "w") as f:
                    f.write("\n".join(sa) + "\n")
                n_slices += 1
    print(f"DOTA slices: {n_slices}")

    # ── DIOR convert ──
    from src.data_prep.process_dior import main as process_dior_main
    o_img = work / "DIOR" / "processed" / "images"; o_img.mkdir(parents=True, exist_ok=True)
    o_lbl = work / "DIOR" / "processed" / "labels"; o_lbl.mkdir(parents=True, exist_ok=True)
    banner("PREPARE — DIOR XML→YOLO")
    process_dior_main(input_img_dir=P["dior_jpeg"], input_xml_dir=P["dior_ann"],
                      out_img_dir=str(o_img), out_lbl_dir=str(o_lbl), use_symlink=False)

    # ── Build TWO separate datasets ──
    banner("PREPARE — build yolo_dota (surrogate) & yolo_dior (target) SEPARATELY")
    dota_pairs = _collect(d_img, d_lbl)
    dior_pairs = _collect(o_img, o_lbl)
    tr_a, va_a = _build_dataset(dota_pairs, _dset_root(cfg, "dota"), sr, seed)
    tr_i, va_i = _build_dataset(dior_pairs, _dset_root(cfg, "dior"), sr, seed)
    print(f"DOTA  → train {tr_a} / val {va_a}  ({_yaml(cfg,'dota')})")
    print(f"DIOR  → train {tr_i} / val {va_i}  ({_yaml(cfg,'dior')})")


# ──────────────── stage 2/3: train ────────────────
def _train(cfg, which, name, base_weights):
    from ultralytics import YOLO
    sc = cfg[which]
    runs = Path(cfg["paths"]["runs_dir"])
    if sc.get("reuse"):
        w = sc["weights"]
        assert w and os.path.isfile(w), f"{which}.reuse=true nhưng weights không tồn tại: {w}"
        banner(f"{which.upper()} — reuse {w}")
        return w
    ds = "dota" if which == "surrogate" else "dior"
    banner(f"{which.upper()} — train {name} on {ds.upper()}-only "
           f"{sc['epochs']}ep imgsz={sc['imgsz']} batch={sc['batch']}")
    m = YOLO(base_weights)
    m.train(data=_yaml(cfg, ds), epochs=sc["epochs"], imgsz=sc["imgsz"], batch=sc["batch"],
            device=cfg["environment"]["device"], project=str(runs / "detect"), name=name,
            exist_ok=True, patience=cfg["train_common"]["patience"],
            cache=cfg["train_common"]["cache"], save=True, plots=True, verbose=True)
    w = str(runs / "detect" / name / "weights" / "best.pt")
    print(f"✓ {which} → {w}")
    return w


def stage_surrogate(cfg):
    base = str(ROOT / "yolov8n.pt") if (ROOT / "yolov8n.pt").exists() else "yolov8n.pt"
    return _train(cfg, "surrogate", "surrogate_yolov8n_dota", base)


def stage_target(cfg):
    return _train(cfg, "target", "target_rtdetr_dior", "rtdetr-l.pt")


# ──────────────── stage 4: attack + transfer ────────────────
def stage_attack(cfg, surr_pt, target_pt):
    from src.attack.de_attack_real import (
        DEAttackReal, plot_convergence, plot_patch, save_attack_examples)

    ac = cfg["attack"]
    seed = cfg["environment"]["seed"]
    dev = "cpu" if cfg["environment"]["device"] == "cpu" else "0"
    dota_vi, dota_vl = _val_dirs(cfg, "dota")   # optimize + surrogate held-out
    dior_vi, dior_vl = _val_dirs(cfg, "dior")   # transfer held-out
    adir = Path(cfg["paths"]["attack_dir"]); adir.mkdir(parents=True, exist_ok=True)

    banner("ATTACK — surrogate=YOLOv8n/DOTA  target=RT-DETR/DIOR")
    atk_s = DEAttackReal(surr_pt,   device=dev, conf_threshold=0.25, iou_threshold=0.5, verbose=True)
    atk_t = DEAttackReal(target_pt, device=dev, conf_threshold=0.25, iou_threshold=0.5, verbose=True)

    results = {}
    for c in ac["configs"]:
        name = c["name"]; ps = c["patch_size"]; npa = c["num_patches"]
        is_rand = c.get("random", False)
        banner(f"CONFIG {name} | patch={ps} n={npa} opt_pos={c['optimize_positions']} random={is_rand}")

        if is_rand:
            # Đối chứng: patch ngẫu nhiên cùng diện tích, KHÔNG tối ưu
            np.random.seed(seed)
            best_patch = np.random.randint(0, 256, (ps, ps, 3), dtype=np.uint8)
            pos, history, opt_paths = None, None, None
        else:
            atk_s.prepare_attack_set(dota_vi, dota_vl, n_images=ac["n_attack"], seed=seed)
            opt_paths = [d["img_path"] for d in atk_s.attack_data]  # exclude from held-out
            best_patch, history = atk_s.run_de(
                patch_size=ps, num_patches=npa, optimize_positions=c["optimize_positions"],
                pop_size=ac["pop_size"], max_gen=ac["max_gen"], F=ac["F"], CR=ac["CR"],
                w_obj=ac["w_obj"], w_fp=ac["w_fp"], w_tv=ac["w_tv"], w_nps=ac["w_nps"])
            pos = atk_s.last_positions

        cdir = adir / name; cdir.mkdir(exist_ok=True)
        np.save(str(cdir / "best_patch.npy"), best_patch)
        if pos is not None:
            np.save(str(cdir / "positions.npy"), pos)
        if history is not None:
            plot_convergence(history, str(cdir / "convergence.png"))
        plot_patch(best_patch, str(cdir / "patch.png"))

        # Surrogate held-out (DOTA) — disjoint from the optimisation set
        m_s = atk_s.evaluate_patch(best_patch, positions=pos, img_dir=dota_vi, lbl_dir=dota_vl,
                                   n_images=ac["n_heldout"], num_patches=npa, seed=99,
                                   exclude=opt_paths)
        atk_s.prepare_attack_set(dota_vi, dota_vl, n_images=6, seed=seed)
        save_attack_examples(atk_s, best_patch, num_patches=npa, positions=pos,
                             save_dir=str(cdir / "examples_surrogate_dota"), n=6)

        # Transfer → target (DIOR)
        m_t = atk_t.evaluate_patch(best_patch, positions=pos, img_dir=dior_vi, lbl_dir=dior_vl,
                                   n_images=ac["n_heldout"], num_patches=npa, seed=99)
        atk_t.prepare_attack_set(dior_vi, dior_vl, n_images=6, seed=seed)
        save_attack_examples(atk_t, best_patch, num_patches=npa, positions=pos,
                             save_dir=str(cdir / "examples_target_dior"), n=6)

        results[name] = dict(config=c, surrogate=m_s, target=m_t,
                             positions=None if pos is None else pos.tolist())
        print(f"  [{name}] surrogate(DOTA) ASR={m_s['ASR']:.3f} | transfer(DIOR) ASR={m_t['ASR']:.3f}")

    # ── Table ──
    banner("SUMMARY — transfer DOTA→DIOR (held-out model & data) + random control")
    hdr = (f'{"Config":>16s} | {"surr ASR":>9s} {"AORR":>6s} | '
           f'{"TRANSFER ASR":>12s} {"AORR":>6s} | {"transfer%":>9s}')
    print(hdr); print("-" * len(hdr))
    rand_t = results.get("random_single", {}).get("target", {}).get("ASR", None)
    rows = []
    for name, r in results.items():
        s, t = r["surrogate"], r["target"]
        tr = (t["ASR"] / s["ASR"] * 100) if s["ASR"] > 0 else 0.0
        tag = "  (control)" if r["config"].get("random") else ""
        print(f'{name:>16s} | {s["ASR"]:>9.3f} {s["AORR"]:>6.3f} | '
              f'{t["ASR"]:>12.3f} {t["AORR"]:>6.3f} | {tr:>8.1f}%{tag}')
        rows.append([name, s["ASR"], s["AORR"], t["ASR"], t["AORR"], tr,
                     int(bool(r["config"].get("random")))])
    print("-" * len(hdr))
    if rand_t is not None:
        print(f"\nĐối chứng: patch ngẫu nhiên đạt transfer ASR = {rand_t:.3f}. "
              f"Phương pháp PHẢI vượt rõ mức này mới có ý nghĩa.")

    with open(adir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(adir / "summary.csv", "w", encoding="utf-8") as f:
        f.write("config,surr_ASR,surr_AORR,transfer_ASR,transfer_AORR,transfer_pct,is_random\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\n✓ results.json + summary.csv → {adir}")
    return results


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent / "config_local.yaml"))
    ap.add_argument("--stages", default="")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()] or cfg["run"]["stages"]
    print(f"Project root: {ROOT}\nStages: {stages}")
    t0 = time.time()

    surr_pt = cfg["surrogate"].get("weights") or None
    target_pt = cfg["target"].get("weights") or None

    if "prepare" in stages:
        stage_prepare(cfg)
    if "surrogate" in stages:
        surr_pt = stage_surrogate(cfg)
    elif cfg["surrogate"].get("reuse"):
        surr_pt = cfg["surrogate"]["weights"]
    if "target" in stages:
        target_pt = stage_target(cfg)
    elif cfg["target"].get("reuse"):
        target_pt = cfg["target"]["weights"]
    if "attack" in stages:
        assert surr_pt and os.path.isfile(surr_pt), \
            "Cần surrogate weights (DOTA) — chạy stage 'surrogate' hoặc đặt surrogate.reuse+weights."
        assert target_pt and os.path.isfile(target_pt), \
            "Cần target weights (DIOR) — chạy stage 'target' hoặc đặt target.reuse+weights."
        stage_attack(cfg, surr_pt, target_pt)

    print(f"\n✅ DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
