"""
Build a real 4-modality dataset from DeepSense 6G Scenario 32 (V2I mmWave):
camera image, LiDAR point cloud, FMCW radar range-Doppler map, and the
transmitter vehicle's GPS position, labelled with the optimal mmWave beam
(grouped into sectors). Beam prediction is inherently multimodal: no single
sensor identifies the transmitter among distractors reliably.

DeepSense 6G: A. Alkhateeb et al., "DeepSense 6G: A Large-Scale Real-World
Multi-Modal Sensing and Communication Dataset," IEEE Comm. Mag., 2023.
"""

import os
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "data", "deepsense", "scenario32")
PATCH = 32
NPTS = 64
RAD = 32
SECTORS = 8          # 64 beams -> 8 sectors of 8 beams
CLASSES = [f"sector{i}" for i in range(SECTORS)]


def _read_ply_points(path, n=NPTS):
    """Ascii ply -> n sampled [x, y, z] points, centred/scaled."""
    pts = []
    with open(path) as f:
        header = True
        for line in f:
            if header:
                if line.startswith("end_header"):
                    header = False
                continue
            parts = line.split()
            if len(parts) >= 3:
                pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    q = np.asarray(pts, dtype=np.float32)
    if len(q) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    q = q - q.mean(0, keepdims=True)
    q = q / (np.abs(q).max() + 1e-6)
    idx = np.random.choice(len(q), n, replace=(len(q) < n))
    return q[idx]


def _radar_map(path, out=RAD):
    """Complex radar cube (rx, samples, chirps) -> log range-Doppler map."""
    cube = np.load(path)                                  # (4, 256, 250)
    rd = np.fft.fft(cube, axis=1)                         # range FFT
    rd = np.fft.fftshift(np.fft.fft(rd, axis=2), axes=2)  # doppler FFT
    mag = np.abs(rd).mean(axis=0)                         # avg antennas
    mag = np.log1p(mag)
    H, W = mag.shape
    hs, ws = H // out, W // out
    m = mag[:hs * out, :ws * out].reshape(out, hs, out, ws).mean(axis=(1, 3))
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m[None].astype(np.float32)                     # (1, RAD, RAD)


def build(cache="results/deepsense_mm.npz"):
    if os.path.exists(cache):
        print(f"  [deepsense] loading cached dataset {cache}")
        d = np.load(cache)
        return {m: d[m] for m in ("img", "lid", "rad", "gps")}, d["y"], d["beam"]

    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, "scenario32_dev.csv"))
    imgs, lids, rads, gpss, beams = [], [], [], [], []
    lat0 = lon0 = None
    for i, row in df.iterrows():
        try:
            im = Image.open(os.path.join(ROOT, row.unit1_rgb)).convert("RGB")
            im = np.asarray(im.resize((PATCH, PATCH))) / 255.0
            lid = _read_ply_points(os.path.join(ROOT, row.unit1_lidar))
            rad = _radar_map(os.path.join(ROOT, row.unit1_radar))
            lat, lon = np.loadtxt(os.path.join(ROOT, row.unit2_loc))[:2]
        except Exception as e:
            continue
        if lat0 is None:
            lat0, lon0 = lat, lon
        spd = float(row.unit2_spd_over_grnd_kmph) \
            if not np.isnan(row.unit2_spd_over_grnd_kmph) else 0.0
        imgs.append(im.transpose(2, 0, 1).astype(np.float32))
        lids.append(lid)
        rads.append(rad)
        gpss.append([lat - lat0, lon - lon0, spd])
        beams.append(int(row.unit1_beam))
        if i % 400 == 0:
            print(f"    [deepsense] {i}/{len(df)}", flush=True)

    img = np.stack(imgs); lid = np.stack(lids); rad = np.stack(rads)
    gps = np.asarray(gpss, dtype=np.float32)
    # normalize gps features
    gps = (gps - gps.mean(0)) / (gps.std(0) + 1e-6)
    beam = np.asarray(beams)
    y = beam * SECTORS // 64                              # sector label
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, img=img, lid=lid, rad=rad, gps=gps,
                        y=y, beam=beam)
    print(f"  [deepsense] built {len(y)} samples, sector counts "
          f"{np.bincount(y, minlength=SECTORS)} -> {cache}")
    return {"img": img, "lid": lid, "rad": rad, "gps": gps}, y, beam


if __name__ == "__main__":
    np.random.seed(0)
    build()
