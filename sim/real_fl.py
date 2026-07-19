"""
Real multimodal federated learning on KITTI, driven by the InTAS mobility and
the proposed encoder caching/forwarding algorithm.

RealMFL holds genuine modality-specific encoders (camera CNN, LiDAR PointNet)
and a local fusion head per vehicle. Encoders are exchanged through the SAME
CachingForwarding decision logic used in the abstract simulator, but here:
  * local_train() runs real SGD on each vehicle's KITTI data partition,
  * commit() performs real FedAvg of encoder weights (Eq. 2),
  * accuracy is the real classification accuracy of the fused model.
Modality/quality heterogeneity is realized by corrupting poor vehicles' data
(blurred/low-light camera, sparse LiDAR), so poor vehicles genuinely need to
receive strong encoders from others.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn

from .config import Config, SCHEMES
from .multimodal_model import make_encoder, FusionHead, LocFusionHead, encoder_forward, FEAT, NCLS
from .kitti_dataset import build as build_kitti, CLASSES

# schemes compared on the real FL backend: ours + framework baselines +
# published multimodal-FL benchmarks (mmFedMC IEEE ICC'24, AutoFed MobiCom'23)
REAL_SCHEMES = SCHEMES + ["mmFedMC", "AutoFed"]


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _fedavg(state_dicts, weights):
    w = np.array(weights, dtype=np.float64); w = w / w.sum()
    out = copy.deepcopy(state_dicts[0])
    for k in out:
        out[k] = sum(float(w[i]) * state_dicts[i][k].float() for i in range(len(state_dicts)))
    return out


class RealMFL:
    """Real multimodal FL backend exposing the interface CachingForwarding needs."""

    def __init__(self, cfg, rng, modality_avail, data, device=None):
        self.loc = "bb" in data
        self.cfg = cfg
        self.rng = rng
        self.R = cfg.modalities                       # ["camera", "lidar"]
        self.N = cfg.num_vehicles
        self.avail = [sorted(a) for a in modality_avail]   # deterministic ordered lists
        self.device = device or _device()
        y = data["y"]
        # modality arrays: either an explicit dict (data["mods"]) or the
        # legacy img/lid pair
        mods = data.get("mods") or {"camera": data["img"], "lidar": data["lid"]}
        self.ncls = int(data.get("ncls", NCLS))
        self.val = data["val"]; self.test = data["test"]

        # Data-volume heterogeneity. A minority of "rich" vehicles hold large
        # local datasets and train full modality encoders (strong encoders).
        # The majority "poor" vehicles hold only a handful of samples: they
        # cannot train a useful encoder, so they FREEZE their encoder and only
        # adapt their lightweight local fusion head -- relying on a strong
        # encoder received from others. Reaching poor vehicles with a good
        # encoder is therefore what determines their accuracy.
        self.D, self.Q, self.strength, self.theta, self.pairs = {}, {}, {}, {}, []
        self.enc, self.head, self.opt, self.local, self.rich = {}, {}, {}, {}, {}
        rich_mask = data.get("rich_mask")
        if rich_mask is not None:
            # partitioned scenario: strong-data vehicles only within a region
            pr = cfg.frac_good * self.N / max(int(np.sum(rich_mask)), 1)
            riches = [bool(rich_mask[i]) and (rng.random() < pr)
                      for i in range(self.N)]
        else:
            riches = [rng.random() < cfg.frac_good for _ in range(self.N)]
        n_rich = max(sum(riches), 1)
        pool = rng.permutation(data["train_idx"])
        poor_sizes = {i: int(rng.integers(4, 12)) for i in range(self.N) if not riches[i]}
        overlap = bool(data.get("overlap", False))
        if overlap:
            # small dataset: vehicles draw (possibly overlapping) local sets --
            # nearby vehicles observing the same scene is realistic anyway
            rich_each = max(min(len(pool) // 2, 60), 30)
        else:
            rich_budget = len(pool) - sum(poor_sizes.values())
            rich_each = max(rich_budget // n_rich, 30)
        cur = 0
        for i in range(self.N):
            size = rich_each if riches[i] else poor_sizes[i]
            if overlap:
                self.local[i] = rng.choice(pool, min(size, len(pool)),
                                           replace=False)
            else:
                e = min(cur + size, len(pool)); self.local[i] = pool[cur:e]; cur = e
            if len(self.local[i]) == 0:
                self.local[i] = pool[:5]
            self.rich[i] = riches[i]
            for r in self.avail[i]:
                self.D[(i, r)] = len(self.local[i])
                self.Q[(i, r)] = rng.uniform(0.8, 1.0) if riches[i] else rng.uniform(0.1, 0.3)
                self.strength[(i, r)] = self.Q[(i, r)]
                self.theta[(i, r)] = self.Q[(i, r)]
                self.pairs.append((i, r))
            # encoders: trainable for rich vehicles, frozen for poor vehicles
            self.enc[i] = {r: make_encoder(r).to(self.device) for r in self.avail[i]}
            Head = LocFusionHead if self.loc else FusionHead
            self.head[i] = Head(self.avail[i], ncls=self.ncls).to(self.device)
            params = list(self.head[i].parameters())
            if riches[i]:
                for r in self.avail[i]:
                    params += list(self.enc[i][r].parameters())
            else:
                for r in self.avail[i]:
                    for p in self.enc[i][r].parameters():
                        p.requires_grad_(False)
            self.opt[i] = torch.optim.Adam(params, lr=1e-3)
        # registry of current encoder weights (for forwarding/aggregation)
        self.t = {m: torch.tensor(a, device=self.device) for m, a in mods.items()}
        self.y_t = torch.tensor(y, device=self.device, dtype=torch.long)
        if self.loc:
            self.bb_t = torch.tensor(data["bb"], device=self.device)
        self.acc = np.zeros(self.N)
        self._corrupt = {i: (self.Q[(i, self.avail[i][0])] < 0.5) for i in range(self.N)}
        # sensor-spec tiers: even data-rich vehicles may carry low-spec
        # sensors for individual modalities (e.g., few-beam LiDAR, low-res
        # camera), degrading that modality's training data only
        self.spec_low = {(i, r) for i in range(self.N) for r in self.avail[i]
                         if rng.random() < getattr(cfg, "spec_low_prob", 0.0)}

    # ---- data access with per-(vehicle, modality) quality corruption ----
    def _batch(self, idx, vehicle):
        x = {m: t[idx].clone() for m, t in self.t.items()}
        allc = self._corrupt.get(vehicle, False)     # degraded sensing (poor)

        def bad(r):
            return allc or (vehicle, r) in self.spec_low
        if "camera" in x and bad("camera"):
            img = x["camera"] + 0.25 * torch.randn_like(x["camera"])
            x["camera"] = torch.clamp(img * 0.6, 0, 1)     # noise + low light
        if "lidar" in x and bad("lidar"):
            mask = (torch.rand_like(x["lidar"][..., :1]) < 0.6).float()
            x["lidar"] = x["lidar"] * mask                 # sparse LiDAR
        if "radar" in x and bad("radar"):
            x["radar"] = x["radar"] + 0.25 * torch.randn_like(x["radar"])
        if "gps" in x and bad("gps"):
            x["gps"] = x["gps"] + 0.1 * torch.randn_like(x["gps"])
        return x, self.y_t[idx]

    def Dmr(self, m, r):
        return self.D.get((m, r), 1)

    # ---- real local training (Eq. 1) ----
    def local_train(self):
        """One round of local SGD; returns the mean local training loss."""
        ce = nn.CrossEntropyLoss()
        tot, cnt = 0.0, 0
        for i in range(self.N):
            idx = self.local[i]
            if len(idx) == 0:
                continue
            self._set_train(i, True)
            for _ in range(self.cfg.local_epochs):
                b = idx[self.rng.choice(len(idx), min(64, len(idx)), replace=False)]
                x, y = self._batch(b, i)
                feats = {r: self.enc[i][r](x[r]) for r in self.avail[i]}
                if self.loc:
                    logits, box = self.head[i].forward_box(feats)
                    loss = ce(logits, y) + 5.0 * nn.functional.smooth_l1_loss(
                        box, self.bb_t[torch.tensor(b, device=self.device)])
                else:
                    logits = self.head[i](feats)
                    loss = ce(logits, y)
                self.opt[i].zero_grad(); loss.backward(); self.opt[i].step()
                tot += float(loss); cnt += 1
        return tot / max(cnt, 1)

    def _set_train(self, i, t):
        self.head[i].train(t)
        for r in self.avail[i]:
            self.enc[i][r].train(t)

    # ---- real evaluation on the shared clean test/val split ----
    @torch.no_grad()
    def evaluate(self, which="val", return_loss=False):
        idx, yv = (self.val if which == "val" else self.test)
        idx_t = torch.tensor(idx, device=self.device)
        y_t = torch.tensor(yv, device=self.device, dtype=torch.long)
        x = {m: t[idx_t] for m, t in self.t.items()}
        accs = np.zeros(self.N); losses = np.zeros(self.N)
        for i in range(self.N):
            self._set_train(i, False)
            feats = {r: self.enc[i][r](x[r]) for r in self.avail[i]}
            logits = self.head[i](feats)
            accs[i] = float((logits.argmax(1) == y_t).float().mean())
            if return_loss:
                losses[i] = float(nn.functional.cross_entropy(logits, y_t))
        return (accs, losses) if return_loss else accs

    def evaluate_class(self, which="test"):
        """Per-vehicle per-class accuracy [N, C]: the service-level view
        (Car / Pedestrian / Cyclist recognition) of the same classifier."""
        idx, yv = (self.val if which == "val" else self.test)
        idx_t = torch.tensor(idx, device=self.device)
        y_t = torch.tensor(yv, device=self.device, dtype=torch.long)
        x = {m: t[idx_t] for m, t in self.t.items()}
        out = np.zeros((self.N, self.ncls))
        masks = [y_t == c for c in range(self.ncls)]
        with torch.no_grad():
            for i in range(self.N):
                self._set_train(i, False)
                feats = {r: self.enc[i][r](x[r]) for r in self.avail[i]}
                pred = self.head[i](feats).argmax(1)
                for c, mc in enumerate(masks):
                    out[i, c] = float((pred[mc] == c).float().mean())
        return out

    def evaluate_loc(self, which="test"):
        """Per-vehicle mean IoU of the regressed box on the held-out split
        (RoI localization task only)."""
        idx, _ = (self.val if which == "val" else self.test)
        idx_t = torch.tensor(idx, device=self.device)
        x = {m: t[idx_t] for m, t in self.t.items()}
        gt = self.bb_t[idx_t]
        g1 = gt[:, :2] - gt[:, 2:] / 2
        g2 = gt[:, :2] + gt[:, 2:] / 2
        garea = (g2 - g1).clamp(min=0).prod(1)
        out = np.zeros(self.N)
        with torch.no_grad():
            for i in range(self.N):
                self._set_train(i, False)
                feats = {r: self.enc[i][r](x[r]) for r in self.avail[i]}
                _, box = self.head[i].forward_box(feats)
                p1 = box[:, :2] - box[:, 2:] / 2
                p2 = box[:, :2] + box[:, 2:] / 2
                lt = torch.maximum(p1, g1)
                rb = torch.minimum(p2, g2)
                inter = (rb - lt).clamp(min=0).prod(1)
                union = (p2 - p1).clamp(min=0).prod(1) + garea - inter
                out[i] = float((inter / union.clamp(min=1e-9)).mean())
        return out

    def refresh_strengths(self):
        if getattr(self.cfg, "per_modality_strength", False):
            self._refresh_permod()
            return
        self.acc = self.evaluate("val")
        for (i, r) in self.pairs:
            self.strength[(i, r)] = float(self.acc[i])
            self.theta[(i, r)] = float(self.acc[i])

    @torch.no_grad()
    def _refresh_permod(self):
        """Per-modality encoder quality chi_{i,r} via leave-one-out validation
        contribution: strength of (i, r) = vehicle val accuracy weighted by
        modality r's share of it. Matches the paper's per-encoder chi, and
        stops schemes from valuing e.g. a chance-level camera encoder just
        because its owner is accurate overall (via another modality)."""
        idx, yv = self.val
        idx_t = torch.tensor(idx, device=self.device)
        y_t = torch.tensor(yv, device=self.device, dtype=torch.long)
        x = {m: t[idx_t] for m, t in self.t.items()}
        self.acc = np.zeros(self.N)
        for i in range(self.N):
            self._set_train(i, False)
            feats = {r: self.enc[i][r](x[r]) for r in self.avail[i]}
            base = float((self.head[i](feats).argmax(1) == y_t).float().mean())
            self.acc[i] = base
            contrib = {}
            for r in self.avail[i]:
                fz = dict(feats)
                fz[r] = torch.zeros_like(feats[r])
                a = float((self.head[i](fz).argmax(1) == y_t).float().mean())
                contrib[r] = max(base - a, 0.0)
            tot = sum(contrib.values())
            for r in self.avail[i]:
                w = contrib[r] / tot if tot > 1e-9 else 1.0 / len(self.avail[i])
                v = base * w * len(self.avail[i])       # keep scale ~ base
                v = float(np.clip(v, 0.0, 1.0))
                self.strength[(i, r)] = v
                self.theta[(i, r)] = v

    # ---- decision proxies used by CachingForwarding ----
    def q_eff(self, i, r, extra=None):
        return float(self.strength[(i, r)])

    def val_loss(self, i, r, q_eff=None):
        qe = self.q_eff(i, r) if q_eff is None else q_eff
        return float((1.0 - qe) ** 2)

    def local_val_loss(self, i, r):
        return self.val_loss(i, r)

    def gain_single(self, i, r, m, s_m):
        if isinstance(s_m, dict):                 # immutable version snapshot
            return self._gain_snapshot(i, r, m, s_m)
        g = max(float(s_m) - float(self.strength[(i, r)]), 0.0)
        return g, None, None

    def _enc_weight(self, i, r):
        """Sample-size weight n_i of vehicle i's own modality-r encoder in
        aggregation (Eq. 1): poor vehicles freeze their encoders and train
        only the fusion head, so their encoder-training sample count is 0 and
        a received encoder replaces the frozen one (omega = 1)."""
        return float(self.Dmr(i, r)) if self.rich.get(i, False) else 0.0

    # ---- encoder-version support (new FACE system model) ----
    def snapshot_encoder(self, i, r):
        """Immutable CPU snapshot theta_x of the current modality-r encoder."""
        return {k: v.detach().cpu().clone()
                for k, v in self.enc[i][r].state_dict().items()}

    @torch.no_grad()
    def _val_acc_single(self, i):
        idx, yv = self.val
        idx_t = torch.tensor(idx, device=self.device)
        y_t = torch.tensor(yv, device=self.device, dtype=torch.long)
        x = {m: t[idx_t] for m, t in self.t.items()}
        self._set_train(i, False)
        feats = {r: self.enc[i][r](x[r]) for r in self.avail[i]}
        return float((self.head[i](feats).argmax(1) == y_t).float().mean())

    @torch.no_grad()
    def _gain_snapshot(self, i, r, m, sd):
        """Realized normalized validation gain Delta_{i,x} (Eq. 2): evaluate
        the candidate model that aggregates the immutable snapshot into the
        local modality-r encoder, then restore the local model."""
        base = self._val_acc_single(i)
        backup = {k: v.detach().clone()
                  for k, v in self.enc[i][r].state_dict().items()}
        cand = _fedavg(
            [backup, {k: v.to(self.device) for k, v in sd.items()}],
            [self._enc_weight(i, r), self.Dmr(m, r)])
        self.enc[i][r].load_state_dict(cand)
        after = self._val_acc_single(i)
        self.enc[i][r].load_state_dict(backup)
        g = max(after - base, 0.0) / max(1.0 - base, 1e-6)
        return g, base, after

    # ---- v4 aggregation: FedAvg over the aggregation set + acceptance test
    # + leave-one-out attribution (Sec. III-E / eq:loo_attribution) ----
    @torch.no_grad()
    def aggregate_test(self, i, r, cands):
        """cands: list of (src, snapshot_dict). Averages the local encoder
        with all candidates weighted by training data volumes, accepts the
        tentative encoder only if validation accuracy does not drop, and
        returns (accepted, {idx: v_x}, base_acc, full_acc) with normalized
        LOO attributions v_x for each candidate."""
        if not cands or r not in self.avail[i]:
            return False, {}, 0.0, 0.0
        base = self._val_acc_single(i)
        backup = {k: v.detach().clone()
                  for k, v in self.enc[i][r].state_dict().items()}
        snaps = [{k: v.to(self.device) for k, v in sd.items()}
                 for (_, sd) in cands]
        w_loc = self._enc_weight(i, r)
        # sequential (greedy) acceptance in the caller's examination order
        # (decreasing predicted reward): a candidate joins the aggregation
        # set only if the tentative aggregate does not decrease validation
        # accuracy -- a stale or harmful encoder is rejected instead of
        # dragging the whole average down, and every accepted step keeps
        # the loss non-increasing (Prop. stability)
        attr = {a: 0.0 for a in range(len(cands))}
        denom = max(1.0 - base, 1e-6)
        kept, cur_sd, cur_acc = [], backup, base
        for a in range(len(cands)):
            sds = [backup] + [snaps[b] for b in kept] + [snaps[a]]
            ws = [w_loc] + [self.Dmr(cands[b][0], r) for b in kept] \
                + [self.Dmr(cands[a][0], r)]
            trial = _fedavg(sds, ws)
            self.enc[i][r].load_state_dict(trial)
            acc_t = self._val_acc_single(i)
            if acc_t + 1e-9 >= cur_acc:              # acceptance test
                attr[a] = max(acc_t - cur_acc, 0.0) / denom
                kept.append(a)
                cur_sd, cur_acc = trial, acc_t
        self.enc[i][r].load_state_dict(cur_sd if kept else backup)
        return bool(kept), attr, base, cur_acc

    # ---- real FedAvg aggregation of received encoders (Eq. 2) ----
    def commit(self, i, r, received):
        if not received or r not in self.avail[i]:
            return
        sds = [self.enc[i][r].state_dict()]
        ws = [self._enc_weight(i, r)]
        for (m, s_m) in received:
            if isinstance(s_m, dict):             # immutable version snapshot
                sds.append({k: v.to(self.device) for k, v in s_m.items()})
                ws.append(self.Dmr(m, r))
            elif r in self.enc.get(m, {}):
                sds.append(self.enc[m][r].state_dict())
                ws.append(self.Dmr(m, r))
        if len(sds) > 1:
            self.enc[i][r].load_state_dict(_fedavg(sds, ws))

    # ---- real metrics ----
    def mean_accuracy(self):
        return float(self.evaluate("test").mean())

    def poor_accuracy(self, thr=0.5):
        accs = self.evaluate("test")
        poor = [i for i in range(self.N)
                if self.Q[(i, self.avail[i][0])] < thr]
        return float(np.mean(accs[poor])) if poor else 0.0

    def tail_accuracy(self, q=0.1):
        accs = self.evaluate("test")
        thr = np.quantile(accs, q)
        return float(accs[accs <= thr].mean())

    def poor_mask(self, thr=0.5):
        return np.array([self.Q[(i, self.avail[i][0])] < thr for i in range(self.N)])


def _prep_data(cfg, seed, dataset="kitti", per_class=None, min_class_count=0,
               loc=False):
    """Load a real multimodal dataset (KITTI or nuScenes) and class-balance it
    so the task is non-trivial. Classes with fewer than `min_class_count`
    samples are dropped (e.g. the very rare nuScenes Cyclist class), which lets
    the remaining classes keep far more data. Returns balanced img/lid/y plus
    train/val/test."""
    if dataset == "deepsense":
        return _prep_deepsense(seed, min_class_count or 250)
    rad = None
    bb = None
    if dataset == "nuscenes":
        from .nuscenes_dataset import build as _bld
        img, lid, rad, y, frame, boxh = _bld(with_radar=True)
    elif loc:
        from .kitti_dataset import build_loc
        img, lid, y, frame, bb = build_loc()
    else:
        img, lid, y, frame, boxh = build_kitti(cache="results/kitti_mm_all.npz")
    rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=NCLS)
    use_classes = [c for c in range(NCLS) if counts[c] >= max(min_class_count, 1)]
    cap = per_class if per_class else int(min(counts[c] for c in use_classes))
    keep = []
    for c in use_classes:
        ci = np.where(y == c)[0]
        keep.append(rng.choice(ci, min(cap, len(ci)), replace=False))
    keep = rng.permutation(np.concatenate(keep))
    img, lid, y = img[keep], lid[keep], y[keep]
    if bb is not None:
        bb = bb[keep]
    mods = {"camera": img, "lidar": lid}
    if rad is not None:
        mods["radar"] = rad[keep]
    n = len(y); perm = rng.permutation(n)
    n_test = int(0.20 * n); n_val = int(0.12 * n)
    test_idx = perm[:n_test]; val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    print(f"  [data] balanced classes {np.bincount(y)}  "
          f"train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")
    out = dict(mods=mods, y=y,
               val=(val_idx, y[val_idx]), test=(test_idx, y[test_idx]),
               train_idx=train_idx)
    if bb is not None:
        out["bb"] = bb
    return out


def _prep_deepsense(seed, min_class_count=250):
    """DeepSense 6G scenario 32: 4 modalities, beam-sector labels. Small
    (~1.4k balanced samples), so vehicles draw overlapping local sets."""
    from .deepsense_dataset import build as _bld
    mods, y, _beam = _bld()
    rng = np.random.default_rng(seed)
    counts = np.bincount(y)
    use = [c for c in range(len(counts)) if counts[c] >= min_class_count]
    cap = int(min(counts[c] for c in use))
    keep = []
    for c in use:
        ci = np.where(y == c)[0]
        keep.append(rng.choice(ci, min(cap, len(ci)), replace=False))
    keep = rng.permutation(np.concatenate(keep))
    y = np.searchsorted(np.array(use), y[keep])
    mods = {{"img": "camera", "lid": "lidar", "rad": "radar",
             "gps": "gps"}[k]: v[keep] for k, v in mods.items()}
    n = len(y); perm = rng.permutation(n)
    n_test = int(0.20 * n); n_val = int(0.12 * n)
    test_idx = perm[:n_test]; val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    print(f"  [deepsense] balanced classes {np.bincount(y)}  "
          f"train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}")
    return dict(mods=mods, y=y, ncls=len(use), overlap=True,
                val=(val_idx, y[val_idx]), test=(test_idx, y[test_idx]),
                train_idx=train_idx)


def run_real_all(cfg=None, seeds=None, device=None, dataset="kitti", min_class_count=0):
    """Real multimodal FL over InTAS mobility for all schemes; real accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .simulator import prepare, make_modality_availability
    from .algorithm import CachingForwarding
    from .plotting import STYLE

    cfg = cfg or Config()
    cfg.modalities = ["camera", "lidar"]
    cfg.modality_prob = {"camera": 1.0, "lidar": 0.85}
    device = device or _device()
    seeds = seeds or [cfg.seed]
    os.makedirs(cfg.figures_dir, exist_ok=True)

    road, mob, gammas = prepare(cfg, device)
    data = _prep_data(cfg, cfg.seed, dataset=dataset, min_class_count=min_class_count)

    metric_keys = ["acc", "poor", "loss", "vloss", "tloss", "tx", "qlen", "util"]
    stacks = {s: {m: [] for m in metric_keys} for s in REAL_SCHEMES}
    for sd in seeds:
        avail = make_modality_availability(cfg, np.random.default_rng(sd + 7))
        for scheme in REAL_SCHEMES:
            torch.manual_seed(sd)          # paired: same init/noise per seed
            rng = np.random.default_rng(sd)
            mfl = RealMFL(cfg, rng, avail, data, device=device)
            alg = CachingForwarding(cfg, mfl, mob, scheme, seed=sd)
            pm = mfl.poor_mask()
            acc_h, poor_h, loss_h, vloss_h, tloss_h, tx_h, q_h, u_h = \
                [], [], [], [], [], [], [], []
            for k in range(mob.Krounds):
                mob.k = k
                train_loss = mfl.local_train()
                mfl.refresh_strengths()
                g = gammas[k] if alg.flags["use_dis"] or alg.flags["cache_policy"] == "psi" \
                    else np.zeros(mob.N)
                selected = alg.run_round(k, g, gamma_eval=gammas[k])
                accs, losses = mfl.evaluate("test", return_loss=True)
                acc_h.append(float(accs.mean()))
                poor_h.append(float(accs[pm].mean()) if pm.any() else 0.0)
                loss_h.append(train_loss)
                # paper-defined validation loss L^val = (1 - Q^eff)^2 (Eq. in
                # mfl.py), with Q^eff the real per-vehicle validation accuracy
                # captured by refresh_strengths()
                vloss_h.append(float(np.mean((1.0 - mfl.acc) ** 2)))
                tloss_h.append(float(losses.mean()))
                tx_h.append(len(selected))
                q_h.append(np.mean(list(alg.Q.values())))
                u_h.append(alg.last_utility)
            stacks[scheme]["acc"].append(acc_h)
            stacks[scheme]["poor"].append(poor_h)
            stacks[scheme]["loss"].append(loss_h)
            stacks[scheme]["vloss"].append(vloss_h)
            stacks[scheme]["tloss"].append(tloss_h)
            stacks[scheme]["tx"].append(tx_h)
            stacks[scheme]["qlen"].append(q_h)
            stacks[scheme]["util"].append(u_h)
            print(f"  [real seed {sd}] {scheme:16s} acc {acc_h[-1]:.3f} "
                  f"poor {poor_h[-1]:.3f} tx/round {np.mean(tx_h):.1f}")
            del mfl, alg
            if device == "cuda":
                torch.cuda.empty_cache()

    results = {}
    for s in REAL_SCHEMES:
        results[s] = {}
        for m in metric_keys:
            arr = np.stack(stacks[s][m])
            results[s][m] = arr.mean(0); results[s][m + "_std"] = arr.std(0)
            results[s][m + "_all"] = arr        # per-seed curves (for paired stats)
    tag = dataset
    np.savez(os.path.join(cfg.results_dir, f"metrics_real_{tag}.npz"),
             **{f"{s}__{k}": v for s, d in results.items() for k, v in d.items()})

    # figures
    K = mob.Krounds; x = np.arange(1, K + 1); mi = np.arange(0, K, max(K // 12, 1))
    for key, ylab, fname, loc in [
            ("acc", "Mean test accuracy", f"fig_real_{tag}_accuracy.png", "lower right"),
            ("poor", "Poor-data vehicle accuracy", f"fig_real_{tag}_poor.png", "lower right")]:
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        for s in REAL_SCHEMES:
            ax.plot(x, results[s][key], label=s, markevery=mi, ms=5, **STYLE[s])
            ax.fill_between(x, results[s][key] - results[s][key + "_std"],
                            results[s][key] + results[s][key + "_std"],
                            color=STYLE[s]["color"], alpha=0.15, lw=0)
        ax.set_xlabel("Global round $k$"); ax.set_ylabel(ylab)
        ax.grid(True, ls=":", alpha=0.6); ax.legend(fontsize=9, loc=loc)
        fig.tight_layout()
        p = os.path.join(cfg.figures_dir, fname)
        fig.savefig(p, dpi=200); fig.savefig(p.replace(".png", ".pdf")); plt.close(fig)
        print("  saved", p)

    print(f"=== REAL multimodal FL ({tag}) final ===")
    for s in REAL_SCHEMES:
        print(f"  {s:16s} acc {results[s]['acc'][-1]:.3f}  poor {results[s]['poor'][-1]:.3f}")
    return results


def centralized_sanity(epochs=8):
    """Quick centralized check that the multimodal model learns KITTI objects."""
    cfg = Config()
    dev = _device()
    d = _prep_data(cfg, cfg.seed)
    img = torch.tensor(d["img"], device=dev); lid = torch.tensor(d["lid"], device=dev)
    y = torch.tensor(d["y"], device=dev, dtype=torch.long)
    tr = d["train_idx"]; te, yte = d["test"]
    enc = {r: make_encoder(r).to(dev) for r in ["camera", "lidar"]}
    head = FusionHead(["camera", "lidar"]).to(dev)
    params = list(head.parameters())
    for r in enc: params += list(enc[r].parameters())
    opt = torch.optim.Adam(params, lr=1e-3); ce = nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        for _ in range(max(1, len(tr) // 128)):
            b = tr[rng.choice(len(tr), 128, replace=False)]
            feats = {"camera": enc["camera"](img[b]), "lidar": enc["lidar"](lid[b])}
            loss = ce(head(feats), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            te_t = torch.tensor(te, device=dev)
            feats = {"camera": enc["camera"](img[te_t]), "lidar": enc["lidar"](lid[te_t])}
            acc = float((head(feats).argmax(1) == torch.tensor(yte, device=dev)).float().mean())
        print(f"  [sanity] epoch {ep}  test acc {acc:.3f}")
    return acc


def main_config():
    """Operating point of the paper's real multimodal FL comparison."""
    cfg = Config()
    cfg.num_vehicles = 80
    cfg.comm_range = 220.0          # moderate density -> receiver-side contention
    cfg.gat_epochs = 30
    cfg.frac_good = 0.15            # scarce strong (data-rich) sources
    cfg.cache_capacity_mb = 30.0
    cfg.contact_time_per_round = 1.8
    cfg.K = 150                        # enough rounds to reach convergence
    cfg.local_epochs = 6              # more local steps/round -> earlier plateau
    return cfg


def main():
    """Reproduce the real multimodal FL (KITTI) comparison from the paper."""
    cfg = main_config()
    # KITTI: 3 classes; nuScenes: drop the very rare Cyclist (<800) -> 2 classes
    run_real_all(cfg, seeds=[2026, 2027, 2028], dataset="kitti", min_class_count=0)
    run_real_all(cfg, seeds=[2026, 2027, 2028], dataset="nuscenes", min_class_count=800)


if __name__ == "__main__":
    main()
