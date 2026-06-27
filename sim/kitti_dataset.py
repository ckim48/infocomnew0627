"""
Build a real multimodal object-classification dataset from KITTI.

For each labeled object (classes Car / Pedestrian / Cyclist) we extract two real
sensor modalities used by the paper:
  * camera : the RGB image patch inside the 2D bounding box (resized 32x32),
  * lidar  : the Velodyne points falling inside the 3D box (P points x [x,y,z],
             box-centred and normalised).
The result is a genuine multimodal classification task on real vehicular sensor
data, used to drive the multimodal FL training in sim/real_fl.py.

KITTI object detection benchmark: A. Geiger et al., CVPR 2012.
"""

import os
import numpy as np
from PIL import Image

KITTI = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "kitti", "training")
CLASSES = ["Car", "Pedestrian", "Cyclist"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
PATCH = 32
NPTS = 64


def _read_calib(path):
    out = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k] = np.array([float(x) for x in v.split()])
    R0 = np.eye(4); R0[:3, :3] = out["R0_rect"].reshape(3, 3)
    Tr = np.eye(4); Tr[:3, :4] = out["Tr_velo_to_cam"].reshape(3, 4)
    return R0 @ Tr                                  # velo -> cam(rect) 4x4


def _read_velo(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]


def _points_in_box(pts_cam, loc, dims, ry):
    """Select cam-rect points inside the 3D box; return box-centred coords."""
    h, w, l = dims
    c = np.array([loc[0], loc[1] - h / 2.0, loc[2]])   # box centre (loc = bottom centre)
    p = pts_cam - c
    cosr, sinr = np.cos(-ry), np.sin(-ry)
    xr = cosr * p[:, 0] - sinr * p[:, 2]
    zr = sinr * p[:, 0] + cosr * p[:, 2]
    yr = p[:, 1]
    m = (np.abs(xr) <= l / 2 + 0.1) & (np.abs(zr) <= w / 2 + 0.1) & (np.abs(yr) <= h / 2 + 0.1)
    q = np.stack([xr[m], yr[m], zr[m]], axis=1)
    return q


def _sample_points(q, n=NPTS):
    if len(q) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    # normalise by box scale
    scale = np.abs(q).max() + 1e-6
    q = q / scale
    if len(q) >= n:
        idx = np.random.choice(len(q), n, replace=False)
    else:
        idx = np.random.choice(len(q), n, replace=True)
    return q[idx].astype(np.float32)


def build(cache="results/kitti_multimodal.npz", max_frames=4000, min_pts=8,
          min_box=20, seed=0):
    if os.path.exists(cache):
        print(f"  [kitti] loading cached dataset {cache}")
        d = np.load(cache)
        return d["img"], d["lid"], d["y"], d["frame"], d["boxh"]
    rng = np.random.default_rng(seed)
    label_dir = os.path.join(KITTI, "label_2")
    frames = sorted(f[:-4] for f in os.listdir(label_dir))[:max_frames]
    imgs, lids, ys, frs, boxhs = [], [], [], [], []
    for n, fr in enumerate(frames):
        lbls = []
        with open(os.path.join(label_dir, fr + ".txt")) as f:
            for line in f:
                t = line.split()
                if t[0] not in CLS_IDX:
                    continue
                lbls.append(t)
        if not lbls:
            continue
        try:
            image = np.asarray(Image.open(os.path.join(KITTI, "image_2", fr + ".png")).convert("RGB"))
            V2C = _read_calib(os.path.join(KITTI, "calib", fr + ".txt"))
            velo = _read_velo(os.path.join(KITTI, "velodyne", fr + ".bin"))
        except FileNotFoundError:
            continue
        velo = velo[velo[:, 0] > 0]                     # in front of car
        pc = (V2C @ np.concatenate([velo, np.ones((len(velo), 1))], axis=1).T).T[:, :3]
        for t in lbls:
            cls = CLS_IDX[t[0]]
            x1, y1, x2, y2 = map(float, t[4:8])
            if (x2 - x1) < min_box or (y2 - y1) < min_box:
                continue
            dims = list(map(float, t[8:11]))            # h,w,l
            loc = list(map(float, t[11:14]))
            ry = float(t[14])
            q = _points_in_box(pc, loc, dims, ry)
            if len(q) < min_pts:
                continue
            patch = image[int(y1):int(y2), int(x1):int(x2)]
            if patch.size == 0:
                continue
            patch = np.asarray(Image.fromarray(patch).resize((PATCH, PATCH))) / 255.0
            imgs.append(patch.transpose(2, 0, 1).astype(np.float32))
            lids.append(_sample_points(q))
            ys.append(cls); frs.append(int(fr)); boxhs.append(y2 - y1)
        if n % 500 == 0:
            print(f"    [kitti] frame {n}/{len(frames)}  objects so far {len(ys)}")
    img = np.stack(imgs); lid = np.stack(lids); y = np.array(ys)
    frame = np.array(frs); boxh = np.array(boxhs, dtype=np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, img=img, lid=lid, y=y, frame=frame, boxh=boxh)
    print(f"  [kitti] built {len(y)} objects, classes {np.bincount(y)} -> {cache}")
    return img, lid, y, frame, boxh


if __name__ == "__main__":
    build()
