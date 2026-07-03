"""
Build a real multimodal object-classification dataset from nuScenes (mini),
mirroring sim/kitti_dataset.py so it plugs into the same real multimodal FL.

For each annotated object we extract:
  * camera : the RGB image patch inside its projected box (resized 32x32),
  * lidar  : the LiDAR points inside its 3D box (P points x [x,y,z], box-centred).
Classes: Car / Pedestrian / Cyclist (3 classes, matching the KITTI task).

nuScenes: H. Caesar et al., CVPR 2020.
"""

import os
import numpy as np
from PIL import Image

NUSC_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "nuscenes")
CLASSES = ["Car", "Pedestrian", "Cyclist"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
PATCH = 32
NPTS = 64
CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
RADARS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
          "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
NRAD = 16                     # radar returns kept per object (zero-padded)


def _map_class(cat):
    if cat.startswith("vehicle.car") or cat.startswith("vehicle.truck") \
            or cat.startswith("vehicle.van"):
        return CLS_IDX["Car"]
    if cat.startswith("human.pedestrian"):
        return CLS_IDX["Pedestrian"]
    if cat.startswith("vehicle.bicycle") or cat.startswith("vehicle.motorcycle"):
        return CLS_IDX["Cyclist"]
    return None


def _sample_points(q, n=NPTS):
    if len(q) == 0:
        return np.zeros((n, 3), dtype=np.float32)
    scale = np.abs(q).max() + 1e-6
    q = q / scale
    idx = np.random.choice(len(q), n, replace=(len(q) < n))
    return q[idx].astype(np.float32)


def _radar_points(nusc, sample, ann_token, scale=1.6):
    """Radar returns near the annotated box, aggregated over all 5 radars,
    box-centred. Features per return: [x, y, vx_comp, vy_comp, rcs]."""
    from nuscenes.utils.data_classes import RadarPointCloud
    from nuscenes.utils.geometry_utils import points_in_box
    out = []
    for rname in RADARS:
        rtok = sample["data"].get(rname)
        if rtok is None:
            continue
        rpath, boxes_r, _ = nusc.get_sample_data(rtok,
                                                 selected_anntokens=[ann_token])
        if not boxes_r:
            continue
        box = boxes_r[0].copy()
        box.wlh = box.wlh * scale                 # radar hits scatter around the box
        try:
            rpc = RadarPointCloud.from_file(rpath)
        except Exception:
            continue
        pts = rpc.points                          # 18 x N (radar frame)
        mask = points_in_box(box, pts[:3, :])
        if not mask.any():
            continue
        sel = pts[:, mask]
        xy = (sel[:2, :].T - box.center[:2])      # box-centred x, y
        # vx_comp, vy_comp are rows 8, 9; rcs is row 5 (nuScenes radar layout)
        feat = np.concatenate([xy, sel[8:10, :].T, sel[5:6, :].T], axis=1)
        out.append(feat)
    if not out:
        return np.zeros((NRAD, 5), dtype=np.float32)
    q = np.concatenate(out, axis=0)
    q[:, :2] /= (np.abs(q[:, :2]).max() + 1e-6)
    q[:, 2:4] /= 10.0                             # typical |v| scale
    q[:, 4] /= 20.0                               # typical rcs scale
    idx = np.random.choice(len(q), NRAD, replace=(len(q) < NRAD))
    return q[idx].astype(np.float32)


def build(cache="results/nuscenes_mm.npz", min_pts=5, min_box=12,
          with_radar=False):
    if with_radar:
        cache = cache.replace(".npz", "3.npz")
    if os.path.exists(cache):
        print(f"  [nusc] loading cached dataset {cache}")
        d = np.load(cache)
        if with_radar:
            return d["img"], d["lid"], d["rad"], d["y"], d["frame"], d["boxh"]
        return d["img"], d["lid"], d["y"], d["frame"], d["boxh"]

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import points_in_box, view_points, BoxVisibility

    nusc = NuScenes(version="v1.0-mini", dataroot=NUSC_ROOT, verbose=False)
    imgs, lids, rads, ys, frs, boxhs = [], [], [], [], [], []
    for si, sample in enumerate(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        lpath, boxes_l, _ = nusc.get_sample_data(lidar_token)
        try:
            pc = LidarPointCloud.from_file(lpath)
        except Exception:
            continue
        pts = pc.points[:3, :]                              # (3, N) lidar frame
        box_by_tok = {b.token: b for b in boxes_l}
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cls = _map_class(ann["category_name"])
            if cls is None or ann_token not in box_by_tok:
                continue
            box_l = box_by_tok[ann_token]
            mask = points_in_box(box_l, pts)
            if mask.sum() < min_pts:
                continue
            q = (pts[:, mask].T - box_l.center) @ box_l.rotation_matrix  # box-centred
            # camera crop: first camera where the box is visible
            patch = None
            for cam in CAMS:
                cam_token = sample["data"][cam]
                _, boxes_c, K = nusc.get_sample_data(
                    cam_token, box_vis_level=BoxVisibility.ANY,
                    selected_anntokens=[ann_token])
                if not boxes_c:
                    continue
                corners = view_points(boxes_c[0].corners(), K, normalize=True)[:2]
                x1, y1 = corners.min(1); x2, y2 = corners.max(1)
                if (x2 - x1) < min_box or (y2 - y1) < min_box:
                    continue
                impath = nusc.get_sample_data_path(cam_token)
                try:
                    im = np.asarray(Image.open(impath).convert("RGB"))
                except Exception:
                    continue
                H, W = im.shape[:2]
                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(W, int(x2)); y2 = min(H, int(y2))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                patch = im[y1:y2, x1:x2]; boxh = y2 - y1
                break
            if patch is None or patch.size == 0:
                continue
            patch = np.asarray(Image.fromarray(patch).resize((PATCH, PATCH))) / 255.0
            imgs.append(patch.transpose(2, 0, 1).astype(np.float32))
            lids.append(_sample_points(q))
            if with_radar:
                rads.append(_radar_points(nusc, sample, ann_token))
            ys.append(cls); frs.append(si); boxhs.append(float(boxh))
        if si % 100 == 0:
            print(f"    [nusc] sample {si}/{len(nusc.sample)}  objects so far {len(ys)}")
    img = np.stack(imgs); lid = np.stack(lids); y = np.array(ys)
    frame = np.array(frs); boxh = np.array(boxhs, dtype=np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if with_radar:
        rad = np.stack(rads)
        np.savez_compressed(cache, img=img, lid=lid, rad=rad, y=y,
                            frame=frame, boxh=boxh)
        print(f"  [nusc] built {len(y)} objects (with radar), "
              f"classes {np.bincount(y)} -> {cache}")
        return img, lid, rad, y, frame, boxh
    np.savez_compressed(cache, img=img, lid=lid, y=y, frame=frame, boxh=boxh)
    print(f"  [nusc] built {len(y)} objects, classes {np.bincount(y)} -> {cache}")
    return img, lid, y, frame, boxh


if __name__ == "__main__":
    build()
