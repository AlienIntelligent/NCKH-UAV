"""
Real Differential Evolution (DE) adversarial attack against a trained YOLOv8 surrogate.

Optimizes adversarial patches that suppress object detection (aircraft/vehicle)
while maintaining physical constraints (TV smoothness, NPS printability).

Black-box attack: optimizes against surrogate, then transfers to unknown target.

Supports TWO optimisation modes:
  * appearance-only (default)  — DE optimises patch pixels; patch is placed at a
    fixed location (bbox centre, or centre+corners for num_patches>1).
  * appearance + POSITION      — DE *also* optimises where each mini-patch sits
    inside the target bbox (DeMPAA / Du et al. style). One shared appearance is
    placed at `num_patches` learned positions.

References:
  - Tang et al. 2024 — Black-box DE vs aerial detectors
  - DeMPAA — Multi-mini-patch adversarial attack (learned placement)
  - Du et al. WACV 2022 — Physical attacks on aerial detectors
"""

import numpy as np
import torch
import cv2
import os
import time
import glob
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def calc_iou(box1, box2):
    """IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def calc_tv_loss(patch):
    """Total Variation — penalizes noisy high-frequency patches."""
    # patch shape: (H, W, 3), float [0, 255]
    dx = np.abs(patch[:, 1:, :] - patch[:, :-1, :])
    dy = np.abs(patch[1:, :, :] - patch[:-1, :, :])
    return (np.mean(dx) + np.mean(dy)) / 255.0


def calc_nps_loss(patch):
    """Non-Printability Score — simplified as colour variance penalty."""
    return np.std(patch / 255.0)


# ---------------------------------------------------------------------------
# Core attack class
# ---------------------------------------------------------------------------

class DEAttackReal:
    """
    Custom fixed-population Differential Evolution attack.

    Unlike scipy.optimize.differential_evolution (whose population scales
    with ndim), this uses a constant population size — practical for the
    high-dimensional patch space (32x32x3 = 3 072 dims).
    """

    def __init__(
        self,
        model_path,
        device="0",
        conf_threshold=0.25,
        iou_threshold=0.5,
        verbose=True,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.verbose = verbose

        # Load surrogate
        if verbose:
            print(f"[DEAttack] Loading surrogate: {model_path}")
        self.model = YOLO(model_path)

        # Will be set by prepare_attack_set()
        self.attack_data = []  # list of dicts {img, clean_dets}

        # Positions of the best patch from the most recent run_de() call
        # (None when appearance-only). Read this after run_de to pass to
        # evaluate_patch / save_attack_examples, including for transfer eval.
        self.last_positions = None

    # ------------------------------------------------------------------
    # 1. Prepare attack image set
    # ------------------------------------------------------------------
    def prepare_attack_set(self, img_dir, lbl_dir=None, n_images=30, seed=42):
        """
        Select validation images that have clean detections.

        Args:
            img_dir:  path to images (e.g. data/yolo_dataset/val/images)
            lbl_dir:  path to labels — used only to pre-filter non-empty
            n_images: how many images to attack (more ⇒ slower but more robust)
            seed:     reproducibility
        """
        rng = random.Random(seed)

        # Gather candidate image paths
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        all_imgs = []
        for ext in exts:
            all_imgs.extend(glob.glob(os.path.join(img_dir, ext)))
        all_imgs.sort()

        # Pre-filter: keep only images with non-empty label files (if lbl_dir given)
        if lbl_dir is not None:
            filtered = []
            for ip in all_imgs:
                stem = Path(ip).stem
                lp = os.path.join(lbl_dir, stem + ".txt")
                if os.path.isfile(lp) and os.path.getsize(lp) > 5:
                    filtered.append(ip)
            all_imgs = filtered

        if self.verbose:
            print(f"[DEAttack] Candidates with labels: {len(all_imgs)}")

        rng.shuffle(all_imgs)
        candidates = all_imgs[: n_images * 3]  # over-select, then keep those with dets

        self.attack_data = []
        for img_path in candidates:
            if len(self.attack_data) >= n_images:
                break

            img = cv2.imread(img_path)
            if img is None:
                continue

            results = self.model.predict(
                source=img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                continue

            boxes = results[0].boxes
            clean_dets = []
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu())
                cls = int(boxes.cls[i].cpu())
                clean_dets.append(dict(bbox=[x1, y1, x2, y2], conf=conf, cls=cls))

            self.attack_data.append(dict(
                img_path=img_path,
                img=img,          # pre-loaded for speed
                clean_dets=clean_dets,
            ))

        total_targets = sum(len(d["clean_dets"]) for d in self.attack_data)
        if self.verbose:
            print(f"[DEAttack] Attack set ready: {len(self.attack_data)} images, "
                  f"{total_targets} target detections")
        return len(self.attack_data)

    # ------------------------------------------------------------------
    # 2. Candidate decoding (appearance + optional positions)
    # ------------------------------------------------------------------
    @staticmethod
    def _decode_candidate(cand, patch_size, num_patches, optimize_positions):
        """
        Split a flat DE candidate vector into (appearance, positions).

        Layout:
          [ patch_size*patch_size*3 appearance pixels in 0..255 ]
          [ num_patches*2 position values in 0..255 ]   (only if optimize_positions)

        positions are returned as an (num_patches, 2) array of (ox, oy) in [0, 1],
        each the *centre* of a mini-patch expressed as a fraction of the target
        bbox width/height. Returns positions=None when optimize_positions is False.
        """
        app_dim = patch_size * patch_size * 3
        appearance = cand[:app_dim].reshape(patch_size, patch_size, 3).astype(np.uint8)
        positions = None
        if optimize_positions:
            raw = cand[app_dim: app_dim + num_patches * 2]
            positions = np.clip(raw / 255.0, 0.0, 1.0).reshape(num_patches, 2)
        return appearance, positions

    # ------------------------------------------------------------------
    # 3. Patch placement
    # ------------------------------------------------------------------
    @staticmethod
    def apply_patch(img, detections, patch, num_patches=1, positions=None):
        """
        Place patch on image at each detection location.

        If `positions` is given (shape [num_patches, 2], values in [0,1]):
            place `num_patches` copies of `patch` at the learned positions inside
            each target bbox — (ox, oy) is the patch CENTRE as a fraction of the
            bbox width/height. Patches are clamped to stay within the image.

        If `positions` is None:
            num_patches == 1 : single patch centred on bbox
            num_patches  > 1 : fixed multi-mini-patches at centre + corners
        """
        patched = img.copy()
        h, w = img.shape[:2]
        ps = patch.shape[0]  # square patch

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # ── Learned positions ───────────────────────────────────────
            if positions is not None:
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                for k in range(num_patches):
                    ox = float(positions[k][0])
                    oy = float(positions[k][1])
                    cx = x1 + ox * bw
                    cy = y1 + oy * bh
                    px1 = int(round(cx - ps / 2.0))
                    py1 = int(round(cy - ps / 2.0))
                    # keep the patch fully inside the image
                    px1 = max(0, min(px1, w - ps)) if w >= ps else 0
                    py1 = max(0, min(py1, h - ps)) if h >= ps else 0
                    px2 = min(w, px1 + ps)
                    py2 = min(h, py1 + ps)
                    pw, ph = px2 - px1, py2 - py1
                    if pw > 0 and ph > 0:
                        patched[py1:py2, px1:px2] = patch[:ph, :pw]
                continue

            # ── Fixed placement (legacy behaviour) ──────────────────────
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if num_patches == 1:
                # Single patch at centre
                px1 = max(0, cx - ps // 2)
                py1 = max(0, cy - ps // 2)
                px2 = min(w, px1 + ps)
                py2 = min(h, py1 + ps)
                pw, ph = px2 - px1, py2 - py1
                if pw > 0 and ph > 0:
                    patched[py1:py2, px1:px2] = patch[:ph, :pw]
            else:
                # Fixed multi-mini-patch (centre + corners)
                mini_ps = max(4, ps // 2)
                mini_patch = cv2.resize(patch, (mini_ps, mini_ps))
                corner_positions = [
                    (cx, cy),
                    (int(x1), int(y1)),
                    (int(x2), int(y1)),
                    (int(x1), int(y2)),
                    (int(x2), int(y2)),
                ]
                for mx, my in corner_positions[:num_patches]:
                    mpx1 = max(0, mx - mini_ps // 2)
                    mpy1 = max(0, my - mini_ps // 2)
                    mpx2 = min(w, mpx1 + mini_ps)
                    mpy2 = min(h, mpy1 + mini_ps)
                    mw, mh = mpx2 - mpx1, mpy2 - mpy1
                    if mw > 0 and mh > 0:
                        patched[mpy1:mpy2, mpx1:mpx2] = mini_patch[:mh, :mw]

        return patched

    # ------------------------------------------------------------------
    # 4. Fitness function
    # ------------------------------------------------------------------
    def fitness(self, cand, patch_size, num_patches, optimize_positions=False,
                w_obj=1.0, w_fp=0.3, w_tv=0.01, w_nps=0.001):
        """
        Evaluate one candidate across all attack images.

        `cand` is a flat vector: appearance pixels (+ position values if
        optimize_positions). Returns scalar (lower is better for the attacker).
        """
        patch, positions = self._decode_candidate(
            cand, patch_size, num_patches, optimize_positions)

        total_obj = 0.0
        total_fp = 0.0
        total_targets = 0

        for data in self.attack_data:
            img = data["img"]
            clean_dets = data["clean_dets"]

            patched_img = self.apply_patch(img, clean_dets, patch,
                                           num_patches, positions)

            results = self.model.predict(
                source=patched_img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            patched_confs = []
            patched_bboxes = []
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                patched_confs = results[0].boxes.conf.cpu().numpy()
                patched_bboxes = results[0].boxes.xyxy.cpu().numpy()

            # Objectness: for each clean target, find max surviving conf
            for cd in clean_dets:
                total_targets += 1
                cb = cd["bbox"]
                max_surv = 0.0
                for j in range(len(patched_bboxes)):
                    if calc_iou(cb, patched_bboxes[j]) > 0.3:
                        max_surv = max(max_surv, patched_confs[j])
                total_obj += max_surv  # we want this → 0

            # False positives
            new_fp = max(0, len(patched_confs) - len(clean_dets))
            total_fp += new_fp

        avg_obj = total_obj / max(1, total_targets)
        avg_fp = total_fp / max(1, len(self.attack_data))

        tv = calc_tv_loss(patch.astype(np.float32))
        nps = calc_nps_loss(patch.astype(np.float32))

        return w_obj * avg_obj + w_fp * avg_fp + w_tv * tv + w_nps * nps

    # ------------------------------------------------------------------
    # 5. DE optimisation loop
    # ------------------------------------------------------------------
    def run_de(
        self,
        patch_size=32,
        num_patches=1,
        optimize_positions=False,
        pop_size=20,
        max_gen=50,
        F=0.8,
        CR=0.7,
        w_obj=1.0,
        w_fp=0.3,
        w_tv=0.01,
        w_nps=0.001,
    ):
        """
        Run fixed-population DE.  Strategy: DE/best/1/bin.

        optimize_positions=False : DE searches patch appearance only (legacy).
        optimize_positions=True  : DE searches appearance AND the (ox, oy) centre
                                    of each of `num_patches` mini-patches inside
                                    the target bbox.

        Returns: (best_patch [H,W,3 uint8], history [list of best fitness]).
        The best positions (or None) are stored on self.last_positions — pass
        them to evaluate_patch()/save_attack_examples(), including transfer eval.
        """
        app_dim = patch_size * patch_size * 3
        ndim = app_dim + (num_patches * 2 if optimize_positions else 0)

        mode = "APPEARANCE + POSITION" if optimize_positions else "APPEARANCE ONLY"
        print(f"\n{'=' * 60}")
        print(f"  DE ATTACK  —  patch {patch_size}x{patch_size} RGB  ({ndim} dims)")
        print(f"  mode={mode}")
        print(f"  pop={pop_size}  gen={max_gen}  F={F}  CR={CR}")
        print(f"  images={len(self.attack_data)}  num_patches/obj={num_patches}")
        print(f"  weights: obj={w_obj} fp={w_fp} tv={w_tv} nps={w_nps}")
        print(f"{'=' * 60}\n")

        t0 = time.time()

        # Initialise population
        pop = np.random.uniform(0, 255, (pop_size, ndim))

        def _eval(x):
            return self.fitness(x, patch_size, num_patches, optimize_positions,
                                w_obj, w_fp, w_tv, w_nps)

        fit = np.array([_eval(ind) for ind in pop])

        best_idx = np.argmin(fit)
        best_fit = fit[best_idx]
        best_ind = pop[best_idx].copy()
        history = [best_fit]

        print(f"  Gen  0 | best {best_fit:.4f} | mean {np.mean(fit):.4f} "
              f"| {time.time() - t0:.0f}s")

        for g in range(1, max_gen + 1):
            for i in range(pop_size):
                # Pick 2 random distinct members (not i)
                idxs = [j for j in range(pop_size) if j != i]
                a, b = np.random.choice(idxs, 2, replace=False)

                # DE/best/1 mutation
                mutant = best_ind + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, 0, 255)

                # Binomial crossover
                mask = np.random.rand(ndim) < CR
                if not mask.any():
                    mask[np.random.randint(ndim)] = True
                trial = np.where(mask, mutant, pop[i])

                trial_fit = _eval(trial)
                if trial_fit <= fit[i]:
                    pop[i] = trial
                    fit[i] = trial_fit

            gen_best = np.argmin(fit)
            if fit[gen_best] < best_fit:
                best_fit = fit[gen_best]
                best_ind = pop[gen_best].copy()

            history.append(best_fit)

            if g % 5 == 0 or g == max_gen:
                elapsed = time.time() - t0
                print(f"  Gen {g:>2} | best {best_fit:.4f} | mean {np.mean(fit):.4f} "
                      f"| {elapsed:.0f}s")

        best_patch, best_positions = self._decode_candidate(
            best_ind, patch_size, num_patches, optimize_positions)
        self.last_positions = best_positions

        elapsed = time.time() - t0
        print(f"\n{'=' * 60}")
        print(f"  DE DONE — best fitness {best_fit:.4f}  ({elapsed:.1f}s)")
        if best_positions is not None:
            pos_str = ", ".join(f"({p[0]:.2f},{p[1]:.2f})" for p in best_positions)
            print(f"  best positions (bbox-frac): {pos_str}")
        print(f"{'=' * 60}")

        return best_patch, history

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    def evaluate_patch(self, patch, positions=None, img_dir=None, lbl_dir=None,
                       n_images=None, num_patches=1, seed=99):
        """
        Evaluate an optimised patch.  Returns dict with ASR, AORR, etc.

        positions: pass self.last_positions (or the positions from the run that
            produced `patch`) to reproduce the learned placement. None ⇒ fixed
            placement. For transfer eval, pass the SAME positions — they are
            bbox-relative fractions and therefore model-agnostic.

        If img_dir is given, prepares a NEW eval set (independent of attack set).
        Otherwise re-uses the attack set.
        """
        if img_dir is not None:
            # Build a fresh eval set
            saved = self.attack_data
            self.prepare_attack_set(img_dir, lbl_dir,
                                    n_images=n_images or 50, seed=seed)
            eval_data = self.attack_data
            self.attack_data = saved
        else:
            eval_data = self.attack_data

        total_targets = 0
        suppressed = 0
        total_conf_red = 0.0
        total_new_fp = 0

        for data in eval_data:
            img = data["img"]
            patched_img = self.apply_patch(img, data["clean_dets"], patch,
                                           num_patches, positions)

            results = self.model.predict(
                source=patched_img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

            p_bboxes, p_confs = [], []
            if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                p_bboxes = results[0].boxes.xyxy.cpu().numpy()
                p_confs = results[0].boxes.conf.cpu().numpy()

            for cd in data["clean_dets"]:
                total_targets += 1
                cb = cd["bbox"]
                clean_conf = cd["conf"]
                max_match = 0.0
                for j in range(len(p_bboxes)):
                    if calc_iou(cb, p_bboxes[j]) > self.iou_threshold:
                        max_match = max(max_match, p_confs[j])
                if max_match < self.conf_threshold:
                    suppressed += 1
                total_conf_red += (clean_conf - max_match)

            new_fp = max(0, len(p_confs) - len(data["clean_dets"]))
            total_new_fp += new_fp

        metrics = dict(
            ASR=suppressed / max(1, total_targets),
            AORR=total_conf_red / max(1, total_targets),
            targets_total=total_targets,
            targets_suppressed=suppressed,
            avg_new_fp=total_new_fp / max(1, len(eval_data)),
            eval_images=len(eval_data),
        )
        return metrics


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_convergence(history, save_path="de_convergence.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(history, "b-", lw=2)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (lower ⇒ better attack)")
    plt.title("DE Attack — Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved convergence plot → {save_path}")


def plot_patch(patch, save_path="best_patch.png"):
    plt.figure(figsize=(3, 3))
    plt.imshow(patch[:, :, ::-1])  # BGR → RGB
    plt.title(f"Adversarial Patch {patch.shape[0]}×{patch.shape[1]}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved patch visualisation → {save_path}")


def save_attack_examples(attack, patch, num_patches=1, positions=None,
                         save_dir="attack_examples", n=6):
    """
    Side-by-side clean (green boxes) vs patched (red boxes) images.
    Pass `positions` (e.g. attack.last_positions) to render learned placement.
    """
    os.makedirs(save_dir, exist_ok=True)

    for idx, data in enumerate(attack.attack_data[:n]):
        img = data["img"]
        patched = attack.apply_patch(img, data["clean_dets"], patch,
                                     num_patches, positions)

        # --- clean visualisation ---
        vis_clean = img.copy()
        for det in data["clean_dets"]:
            b = [int(v) for v in det["bbox"]]
            cv2.rectangle(vis_clean, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(vis_clean, f'{det["conf"]:.2f}',
                        (b[0], max(b[1] - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # --- patched visualisation ---
        res = attack.model.predict(
            source=patched, conf=attack.conf_threshold,
            iou=attack.iou_threshold, device=attack.device, verbose=False,
        )
        vis_patch = patched.copy()
        if len(res) > 0 and res[0].boxes is not None:
            for j in range(len(res[0].boxes)):
                bx = res[0].boxes.xyxy[j].cpu().numpy().astype(int)
                cf = float(res[0].boxes.conf[j].cpu())
                cv2.rectangle(vis_patch, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
                cv2.putText(vis_patch, f'{cf:.2f}',
                            (bx[0], max(bx[1] - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # --- combine ---
        TARGET_H = 512
        def resize_keep(im):
            scale = TARGET_H / im.shape[0]
            return cv2.resize(im, None, fx=scale, fy=scale)

        rc = resize_keep(vis_clean)
        rp = resize_keep(vis_patch)
        # pad to same width
        max_w = max(rc.shape[1], rp.shape[1])
        def pad_w(im, tw):
            if im.shape[1] < tw:
                return np.pad(im, ((0, 0), (0, tw - im.shape[1]), (0, 0)))
            return im
        rc, rp = pad_w(rc, max_w), pad_w(rp, max_w)

        sep = np.full((TARGET_H, 4, 3), 200, dtype=np.uint8)
        combined = np.hstack([rc, sep, rp])

        out = os.path.join(save_dir, f"example_{idx}.jpg")
        cv2.imwrite(out, combined, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"  Saved {min(n, len(attack.attack_data))} examples → {save_dir}/")
