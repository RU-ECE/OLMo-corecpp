"""Where-students-look, step 1: does an off-the-shelf saliency model predict human gaze on user interfaces?

Runs on Colab (GPU if present). Evaluates DeepGaze IIE (natural-image saliency, MIT1003 centre bias) against
the UEyes eye-tracking data (Jiang et al., CHI 2023; 62 people, 1,980 UI screenshots, CC-BY-4.0) on the
UEyes test split, per UI type, next to two baselines that need no model: a centred Gaussian and a
top-left-weighted prior (people read UIs from the top left). A model is useful for the "looked-at" gate
only where it beats both baselines; the numbers say where it does.

Metrics (standard saliency benchmark definitions, as in the UEyes paper): CC and KLD against the 3 s
heatmaps, NSS against the 3 s fixation maps. Higher CC/NSS and lower KLD are better.

  python3 colab_gaze_eval.py --work /content/gaze [--limit N] [--save /content/drive/MyDrive/gaze-eval]
"""
import argparse, csv, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="/content/gaze")
ap.add_argument("--limit", type=int, default=0, help="test images per category (0 = all)")
ap.add_argument("--save", default="")
ap.add_argument("--maxside", type=int, default=1024)
a = ap.parse_args()
W = Path(a.work); W.mkdir(parents=True, exist_ok=True); R = W / "results"; R.mkdir(exist_ok=True)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"; print(line, flush=True)
    with open(R / "run.log", "a") as f: f.write(line + "\n")
    save()


def save():
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        for p in R.iterdir(): shutil.copy(p, a.save)


URL = "https://zenodo.org/records/8010312/files/UEyes_dataset.zip?download=1"
# The full archive is 12.9 GB and Zenodo drops long downloads (a wget -c restarted from zero here), so read
# only what the evaluation needs straight out of the remote zip with HTTP range requests: info.csv, then the
# test split's images, 3 s heatmaps and 3 s fixation maps. Each member's CRC is checked by zipfile on read.
def read_types(path):
    """image_types.csv rows as [image, type, block, split]; a header line, if any, is skipped."""
    out = []
    for line in open(path, encoding="utf-8-sig"):
        r = [c.strip() for c in line.strip().split(";")]
        if len(r) >= 4 and r[3].lower() in ("train", "test"): out.append(r)
    return out


if not (W / "ok.unzipped").exists():
    try:
        from remotezip import RemoteZip
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "-q", "install", "remotezip"], check=True)
        from remotezip import RemoteZip
    log("reading UEyes (Zenodo 8010312) by range requests")
    with RemoteZip(URL) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
        info = "UEyes_dataset/image_types.csv"   # image;type;block;Train|Test (no info.csv in the archive)
        z.extract(info, W / "ds")
        test_stems = {Path(r[0]).stem for r in read_types(W / "ds" / info) if r[3].lower() == "test"}
        want = [n for n in names if Path(n).stem in test_stems and "overlay" not in n
                and any(k in n for k in ("/images/", "heatmaps_3s/", "fixmaps_3s/"))]
        log(f"{len(test_stems)} test images; extracting {len(want)} members")
        for i, n in enumerate(want):
            z.extract(n, W / "ds")
            if i % 300 == 0: log(f"  {i}/{len(want)}")
    (W / "ok.unzipped").touch()
root = next(p.parent for p in (W / "ds").rglob("image_types.csv"))
log(f"dataset at {root}")

# Never share the GPU with a timed benchmark on the same runtime: wait until every listed job has exited.
WAIT_FOR = os.environ.get("GAZE_WAIT_FOR", "colab_slmtp_scale.sh|bench_spec_decode")
while subprocess.run(["pgrep", "-f", WAIT_FOR], capture_output=True).returncode == 0:
    log(f"waiting: a timed GPU job is running ({WAIT_FOR})"); time.sleep(120)
import numpy as np, torch
from PIL import Image
from scipy.ndimage import zoom
try:
    import deepgaze_pytorch
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "git+https://github.com/matthias-k/DeepGaze.git"], check=True)
    import deepgaze_pytorch
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(dev).eval()
cb_path = W / "centerbias_mit1003.npy"
if not cb_path.exists():
    subprocess.run(["wget", "-q", "-O", str(cb_path),
                    "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"], check=True)
CB = np.load(cb_path)

rows = read_types(root / "image_types.csv")
test = [r for r in rows if r[3].lower() == "test"]
log(f"{len(rows)} images, {len(test)} in the test split")
if not test: sys.exit("no test images found: check info.csv columns above")


def find(dirname, stem):
    for ext in (".png", ".jpg", ".jpeg"):
        for p in root.rglob(f"{dirname}/{stem}{ext}"): return p
    return None


def norm_density(m):
    m = m.astype(np.float64); m -= m.min(); s = m.sum(); return m / s if s > 0 else np.full_like(m, 1 / m.size)


def metrics(pred, heat, fix):
    p = norm_density(pred); h = norm_density(heat)
    cc = float(np.corrcoef(p.ravel(), h.ravel())[0, 1])
    eps = 1e-12; kld = float(np.sum(h * np.log(eps + h / (p + eps))))
    z = (p - p.mean()) / (p.std() + 1e-12); nss = float(z[fix > 0].mean()) if (fix > 0).any() else float("nan")
    return cc, kld, nss


def gauss(h, w, cy, cx, sy, sx):
    y, x = np.mgrid[0:h, 0:w]; return np.exp(-(((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2) / 2)


per = {}; out_rows = []
counts = {}
for r in test:
    name, cat = r[0], r[1]
    if a.limit and counts.get(cat, 0) >= a.limit: continue
    stem = Path(name).stem
    ip, hp, fp = find("images", stem), find("heatmaps_3s", stem), find("fixmaps_3s", stem)
    if not (ip and hp and fp): continue
    img = Image.open(ip).convert("RGB")
    s = min(1.0, a.maxside / max(img.size)); img = img.resize((round(img.width * s), round(img.height * s)))
    x = np.asarray(img); H, Wd = x.shape[:2]
    heat = np.asarray(Image.open(hp).convert("L").resize((Wd, H)), dtype=np.float64)
    fixs = np.asarray(Image.open(fp).convert("L").resize((Wd, H), Image.NEAREST))
    cb = zoom(CB, (H / CB.shape[0], Wd / CB.shape[1]), order=0, mode="nearest"); cb -= np.logaddexp.reduce(cb.ravel())
    with torch.no_grad():
        ld = model(torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32, device=dev),
                   torch.tensor(cb[None], dtype=torch.float32, device=dev))[0, 0].cpu().numpy()
    preds = {"deepgaze_iie": np.exp(ld),
             "centre_gaussian": gauss(H, Wd, H / 2, Wd / 2, H / 4, Wd / 4),
             "top_left_prior": gauss(H, Wd, H * 0.25, Wd * 0.3, H / 3, Wd / 3)}
    for k, p in preds.items():
        m = metrics(p, heat, fixs); per.setdefault((cat, k), []).append(m)
        out_rows.append({"image": stem, "category": cat, "model": k, "cc": m[0], "kld": m[1], "nss": m[2]})
    counts[cat] = counts.get(cat, 0) + 1
    if sum(counts.values()) % 50 == 0 or sum(counts.values()) <= 3:
        n = sum(counts.values()); log(f"{n} images scored; last {stem} ({cat}) deepgaze cc {per[(cat, 'deepgaze_iie')][-1][0]:.3f}")

with open(R / "per_image.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
summary = {}
lines = ["| UI type | model | n | CC ↑ | KLD ↓ | NSS ↑ |", "|---|---|---|---|---|---|"]
for (cat, k), ms in sorted(per.items()):
    arr = np.array(ms); mean = np.nanmean(arr, 0)
    summary.setdefault(cat, {})[k] = {"n": len(ms), "cc": mean[0], "kld": mean[1], "nss": mean[2]}
    lines.append(f"| {cat} | {k} | {len(ms)} | {mean[0]:.3f} | {mean[1]:.3f} | {mean[2]:.3f} |")
verdict = {cat: all(v["deepgaze_iie"]["cc"] > v[b]["cc"] and v["deepgaze_iie"]["nss"] > v[b]["nss"]
                    for b in ("centre_gaussian", "top_left_prior")) for cat, v in summary.items()}
lines.append(""); lines += [f"- {cat}: DeepGaze IIE beats both priors on CC and NSS: {'yes' if ok else 'NO'}" for cat, ok in verdict.items()]
(R / "summary.md").write_text("\n".join(lines) + "\n")
json.dump({"summary": summary, "verdict": verdict, "device": dev, "maxside": a.maxside}, open(R / "summary.json", "w"), indent=1)
print("\n".join(lines)); log("GAZE_EVAL_DONE")
