"""Where-students-look, step 1b: is the 'DeepGaze loses to a top-left prior on UIs' result real?

Follow-up to colab_gaze_eval.py (same work dir, UEyes test split already extracted). Adds:
  1. pipeline self-check: a ground-truth heatmap scored against itself (CC must be 1, KLD 0) and against a
     different image's heatmap (must be clearly lower), so the metric code is shown able to tell;
  2. fixations rescaled by coordinate (the first run resized fixation maps nearest-neighbour, which can
     drop points); NSS on the rescaled points;
  3. a data-driven prior: the mean 3 s heatmap of the UEyes TRAIN split per UI type (read by range
     requests), used both as a baseline and as DeepGaze IIE's centre bias, which separates "wrong prior"
     from "useless features";
  4. paired Wilcoxon tests per UI type on per-image CC and NSS.
  python3 colab_gaze_check.py --work /mnt/local-scratch/gaze [--save DIR]
"""
import argparse, csv, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy.ndimage import zoom
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser(); ap.add_argument("--work", default="/mnt/local-scratch/gaze")
ap.add_argument("--save", default=""); ap.add_argument("--maxside", type=int, default=1024)
a = ap.parse_args(); W = Path(a.work); R = W / "results"; R.mkdir(parents=True, exist_ok=True)
def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    open(R / "check.log", "a").write(line + "\n")
    if a.save: os.makedirs(a.save, exist_ok=True); [shutil.copy(p, a.save) for p in R.iterdir()]
URL = "https://zenodo.org/records/8010312/files/UEyes_dataset.zip?download=1"
root = next(p.parent for p in (W / "ds").rglob("image_types.csv"))
rows = [r for r in ([c.strip() for c in l.strip().split(";")] for l in open(root / "image_types.csv", encoding="utf-8-sig"))
        if len(r) >= 4 and r[3].lower() in ("train", "test")]
test = [r for r in rows if r[3].lower() == "test"]; train = [r for r in rows if r[3].lower() == "train"]
P = 64  # prior resolution (P x P), resized per image

# 3. train-split mean heatmap per UI type, by range requests (heatmaps only)
prior_file = W / "train_priors.npz"
if not prior_file.exists():
    from remotezip import RemoteZip
    acc = {}; n = {}
    with RemoteZip(URL) as z:
        names = {Path(x).stem: x for x in z.namelist() if "heatmaps_3s/" in x and "overlay" not in x and not x.startswith("__")}
        for i, r in enumerate(train):
            nm = names.get(Path(r[0]).stem)
            if not nm: continue
            import io
            h = np.asarray(Image.open(io.BytesIO(z.read(nm))).convert("L").resize((P, P), Image.BILINEAR), np.float64)
            h = h / max(h.sum(), 1e-9); acc[r[1]] = acc.get(r[1], 0) + h; n[r[1]] = n.get(r[1], 0) + 1
            if i % 300 == 0: log(f"train heatmaps {i}/{len(train)}")
    np.savez(prior_file, **{k: acc[k] / n[k] for k in acc}); log(f"train priors from {n}")
PRI = dict(np.load(prior_file))

def find(d, stem):
    for e in (".png", ".jpg", ".jpeg"):
        for p in root.rglob(f"{d}/{stem}{e}"): return p
def nd(m):
    m = m.astype(np.float64) - m.min(); s = m.sum(); return m / s if s > 0 else np.full_like(m, 1 / m.size)
def cc(p, h): return float(np.corrcoef(nd(p).ravel(), nd(h).ravel())[0, 1])
def kld(p, h): p, h = nd(p), nd(h); return float(np.sum(h * np.log(1e-12 + h / (p + 1e-12))))
def nss(p, pts):
    z = (p - p.mean()) / (p.std() + 1e-12); return float(np.mean([z[y, x] for y, x in pts])) if len(pts) else float("nan")
def gauss(H, Wd, cy, cx, sy, sx):
    y, x = np.mgrid[0:H, 0:Wd]; return np.exp(-(((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2) / 2)

import deepgaze_pytorch
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(dev).eval()
CB = np.load(W / "centerbias_mit1003.npy")
def deepgaze(x, logprior):
    with torch.no_grad():
        return np.exp(model(torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32, device=dev),
                            torch.tensor(logprior[None], dtype=torch.float32, device=dev))[0, 0].cpu().numpy())

out = []; selfcheck = []; heats = []
for r in test:
    stem, cat = Path(r[0]).stem, r[1]
    ip, hp, fp = find("images", stem), find("heatmaps_3s", stem), find("fixmaps_3s", stem)
    if not (ip and hp and fp): continue
    img = Image.open(ip).convert("RGB"); s = min(1.0, a.maxside / max(img.size))
    img = img.resize((round(img.width * s), round(img.height * s))); x = np.asarray(img); H, Wd = x.shape[:2]
    heat = np.asarray(Image.open(hp).convert("L").resize((Wd, H), Image.BILINEAR), np.float64)
    fx = np.asarray(Image.open(fp).convert("L")); fy, fxx = np.nonzero(fx > 0)   # 2. rescale coordinates
    pts = sorted(set(zip(np.clip((fy * H / fx.shape[0]).astype(int), 0, H - 1), np.clip((fxx * Wd / fx.shape[1]).astype(int), 0, Wd - 1))))
    heats.append(heat)
    pr = zoom(PRI[cat], (H / P, Wd / P), order=1); pr = np.maximum(pr, 1e-12)
    cb_mit = zoom(CB, (H / CB.shape[0], Wd / CB.shape[1]), order=0, mode="nearest"); cb_mit -= np.logaddexp.reduce(cb_mit.ravel())
    lp = np.log(pr / pr.sum())
    preds = {"deepgaze_iie (MIT prior)": deepgaze(x, cb_mit), "deepgaze_iie (UEyes train prior)": deepgaze(x, lp),
             "UEyes train-mean prior": pr, "top_left_prior": gauss(H, Wd, H * .25, Wd * .3, H / 3, Wd / 3),
             "centre_gaussian": gauss(H, Wd, H / 2, Wd / 2, H / 4, Wd / 4)}
    for k, p in preds.items(): out.append({"image": stem, "category": cat, "model": k, "cc": cc(p, heat), "kld": kld(p, heat), "nss": nss(p, pts), "fixations": len(pts)})
# 1. self-check: GT vs itself, and vs the next image's GT (resized)
for i, h in enumerate(heats):
    o = np.asarray(Image.fromarray(heats[(i + 1) % len(heats)]).resize((h.shape[1], h.shape[0])))
    selfcheck.append((cc(h, h), kld(h, h), cc(o, h)))
sc = np.array(selfcheck)
lines = [f"self-check over {len(sc)} images: CC(gt,gt) mean {sc[:,0].mean():.4f}, KLD(gt,gt) mean {sc[:,1].mean():.2e}, CC(other gt, gt) mean {sc[:,2].mean():.3f}", "",
         "| UI type | model | n | CC | KLD | NSS (rescaled fixations) |", "|---|---|---|---|---|---|"]
cats = sorted({o["category"] for o in out}); models = list(dict.fromkeys(o["model"] for o in out)); summ = {}
for c in cats:
    for m in models:
        v = [o for o in out if o["category"] == c and o["model"] == m]
        mm = {k: float(np.nanmean([o[k] for o in v])) for k in ("cc", "kld", "nss")}; summ[(c, m)] = (v, mm)
        lines.append(f"| {c} | {m} | {len(v)} | {mm['cc']:.3f} | {mm['kld']:.3f} | {mm['nss']:.3f} |")
lines += ["", "Paired Wilcoxon (per image) vs the UEyes train-mean prior:"]
for c in cats:
    base = summ[(c, "UEyes train-mean prior")][0]
    for m in ("deepgaze_iie (MIT prior)", "deepgaze_iie (UEyes train prior)", "top_left_prior"):
        v = summ[(c, m)][0]
        for k in ("cc", "nss"):
            d = [o[k] - b[k] for o, b in zip(v, base)]
            pval = wilcoxon(d).pvalue if any(d) else 1.0
            lines.append(f"- {c} {m} {k}: median diff {np.median(d):+.3f}, p = {pval:.3g}")
(R / "check.md").write_text("\n".join(lines) + "\n")
with open(R / "check_per_image.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print("\n".join(lines)); log("GAZE_CHECK_DONE")
