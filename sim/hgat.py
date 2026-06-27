"""
Hierarchical GAT-based mobility prediction (Sec. III-E, Eq. (5)-(10)).

Two stacked graph-attention stages:
  road stage :  h_e^road = GAT_road(z_e, G^road)                       (Eq. 5)
  veh  stage :  h_i^veh  = GAT_veh(x_i, {h_e^road}_reach, G^com(k))    (Eq. 6)
then a transition head produces o_{i,e,e'} (Eq. 8), softmax -> pi (transition
prob), multi-step reachability P^(h), and the future contact score
Gamma_j^road(k) (Eq. 10).

Sparse edge-based attention is used so the road graph (|V| ~ 7,900 segments)
is handled without dense |V| x |V| tensors. The transition head is trained
(self-supervised) to match realized turns from the InTAS trace.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def segment_softmax(logit, dst, N):
    """Softmax of edge logits grouped by destination node. logit: [E,H]."""
    m = logit.max(dim=0, keepdim=True)[0]            # global per-head max (stable)
    ex = torch.exp(logit - m)                        # [E,H]
    H = logit.size(1)
    denom = torch.zeros(N, H, device=logit.device).index_add_(0, dst, ex)
    return ex / (denom[dst] + 1e-16)


class SparseGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.a_src = nn.Parameter(torch.zeros(heads, out_dim))
        self.a_dst = nn.Parameter(torch.zeros(heads, out_dim))
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        h = self.W(x).view(N, self.heads, self.out_dim)          # [N,H,F]
        e_src = (h[src] * self.a_src).sum(-1)                    # [E,H]
        e_dst = (h[dst] * self.a_dst).sum(-1)                    # [E,H]
        att = segment_softmax(self.leaky(e_src + e_dst), dst, N) # [E,H]
        msg = att.unsqueeze(-1) * h[src]                         # [E,H,F]
        out = torch.zeros(N, self.heads, self.out_dim, device=x.device)
        out.index_add_(0, dst, msg)
        return F.elu(out.reshape(N, self.heads * self.out_dim))


def add_self_loops(edge_index, N, device):
    sl = torch.arange(N, device=device)
    sl = torch.stack([sl, sl])
    return torch.cat([edge_index, sl], dim=1)


class HierGAT(nn.Module):
    def __init__(self, cfg, road_feat_dim, veh_feat_dim):
        super().__init__()
        H, hid = cfg.gat_heads, cfg.gat_hidden
        self.road_gat1 = SparseGATLayer(road_feat_dim, hid, H)
        self.road_gat2 = SparseGATLayer(hid * H, hid, 1)
        self.veh_gat1 = SparseGATLayer(veh_feat_dim, hid, H)
        self.veh_gat2 = SparseGATLayer(hid * H, hid, 1)
        self.turn_emb = nn.Embedding(4, 8)                       # d_{delta(e,e')}
        self.w_o = nn.Linear(hid + hid + hid + 8, 1)            # transition head (Eq. 8)

    def encode_road(self, z, road_ei):
        h = self.road_gat1(z, road_ei)
        return self.road_gat2(h, road_ei)                       # [V, hid]

    def encode_veh(self, x, com_ei, road_emb, veh_seg):
        h0 = torch.cat([x, road_emb[veh_seg]], dim=1)           # inject current segment
        h = self.veh_gat1(h0, com_ei)
        return self.veh_gat2(h, com_ei)                         # [N, hid]

    def transition_logits(self, veh_emb, road_emb, i, e, succ, turn_lab):
        hv, he = veh_emb[i], road_emb[e]
        feats = []
        for e2 in succ:
            d = self.turn_emb(torch.tensor(turn_lab[(e, e2)], device=hv.device))
            feats.append(torch.cat([hv, he, road_emb[e2], d]))
        return self.w_o(torch.stack(feats)).squeeze(-1)         # [|succ|]


def build_features(mob):
    """Vehicle feature x_i(k): [pos(2), speed, prog, normalized degree]."""
    xy = mob.vehicle_xy()
    xy = (xy - xy.mean(0)) / (xy.std(0) + 1e-6)
    A = mob.v2v_graph()
    src, dst = np.where(A > 0)
    com_ei = np.stack([src, dst]) if len(src) else np.zeros((2, 0), dtype=np.int64)
    deg = A.sum(1, keepdims=True)
    spd = (mob.speed[:, None] - mob.speed.mean()) / (mob.speed.std() + 1e-6)
    prog = mob.prog[:, None]
    x = np.concatenate([xy, spd, prog, deg / max(deg.max(), 1)], axis=1).astype(np.float32)
    return x, com_ei.astype(np.int64)


def train_hgat(cfg, road, mob, device="cpu", warmup_rounds=40):
    """Self-supervised training of the predictor against realized InTAS turns."""
    mob.reset()
    z0 = mob.traffic_state()
    road_feat_dim = z0.shape[1]
    x0, _ = build_features(mob)
    model = HierGAT(cfg, road_feat_dim, x0.shape[1] + cfg.gat_hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.gat_lr)

    road_ei = torch.tensor(road.edge_index, device=device)      # already has self-loops
    turn_lab = road.turn

    # collect warm-up snapshots (state + realized transition distribution)
    snapshots = []
    nsnap = min(warmup_rounds, mob.Krounds)
    for _ in range(nsnap):
        z = mob.traffic_state()
        x, com_ei = build_features(mob)
        labels = mob.transition_probs_true()
        snapshots.append((torch.tensor(z, device=device),
                          torch.tensor(x, device=device),
                          add_self_loops(torch.tensor(com_ei, device=device), mob.N, device),
                          torch.tensor(mob.seg, device=device, dtype=torch.long),
                          labels))
        mob.step()
    mob.reset()

    for ep in range(cfg.gat_epochs):
        tot, nb = 0.0, 0
        for (zt, xt, com_ei, segt, labels) in snapshots:
            road_emb = model.encode_road(zt, road_ei)
            veh_emb = model.encode_veh(xt, com_ei, road_emb, segt)
            loss, ns = 0.0, 0
            for i, (e, succ, w) in enumerate(labels):
                if len(succ) < 2 or w.max() < 0.999:            # only confident one-hot labels
                    continue
                logit = model.transition_logits(veh_emb, road_emb, i, e, succ, turn_lab)
                logp = F.log_softmax(logit, dim=0)
                target = torch.tensor(w, device=device, dtype=torch.float32)
                loss = loss - (target * logp).sum()
                ns += 1
            if ns == 0:
                continue
            loss = loss / ns
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if ep % 30 == 0 or ep == cfg.gat_epochs - 1:
            print(f"    [HGAT] epoch {ep:3d}  CE-loss {tot/max(nb,1):.4f}")
    model.eval()
    return model, road_ei


@torch.no_grad()
def future_contact_scores(cfg, road, mob, model, road_ei, device="cpu"):
    """
    Gamma_j^road(k): the expected number of *future V2V co-locations* of vehicle
    j (Eq. 10), computed from learned road-aware transition probabilities.

    For every vehicle we propagate a distribution over road segments h hops into
    the future, P_j^(h)(e). The predicted cohort occupancy of a segment is
    O^(h)(e) = sum_j P_j^(h)(e). The future contact potential of vehicle j is the
    discounted expected co-location with other cohort vehicles along its
    predicted trajectory:
        Gamma_j = sum_h gamma^h  sum_e  P_j^(h)(e) * (O^(h)(e) - P_j^(h)(e)).
    This is road-aware (paths follow the topology) and traffic-aware (it counts
    where the cohort is actually heading), unlike a topology-blind density count.
    """
    z = mob.traffic_state()
    x, com_ei = build_features(mob)
    zt = torch.tensor(z, device=device)
    xt = torch.tensor(x, device=device)
    com_ei_t = add_self_loops(torch.tensor(com_ei, device=device), mob.N, device)
    segt = torch.tensor(mob.seg, device=device, dtype=torch.long)
    road_emb = model.encode_road(zt, road_ei)
    veh_emb = model.encode_veh(xt, com_ei_t, road_emb, segt)
    turn_lab = road.turn
    N = mob.N

    # transition-prob cache per (vehicle, segment)
    tcache = {}
    def trans(i, e):
        key = (i, e)
        if key not in tcache:
            succ = road.successors[e]
            if len(succ) <= 1:
                tcache[key] = (succ, np.array([1.0]) if succ else np.array([]))
            else:
                logit = model.transition_logits(veh_emb, road_emb, i, e, succ, turn_lab)
                tcache[key] = (succ, torch.softmax(logit, dim=0).cpu().numpy())
        return tcache[key]

    # h=0 distributions: each vehicle at its current segment
    dists = [{int(mob.seg[i]): 1.0} for i in range(N)]
    gamma = np.zeros(N)
    for h in range(1, cfg.H_max + 1):
        # advance every vehicle's segment distribution by one hop
        new_dists = []
        for i in range(N):
            nd = {}
            for e, p_e in dists[i].items():
                if p_e <= 1e-6:
                    continue
                succ, pi = trans(i, e)
                for idx, e2 in enumerate(succ):
                    nd[e2] = nd.get(e2, 0.0) + p_e * float(pi[idx])
            new_dists.append(nd)
        dists = new_dists
        # predicted cohort occupancy O^(h)(e)
        occ = {}
        for i in range(N):
            for e, p in dists[i].items():
                occ[e] = occ.get(e, 0.0) + p
        # expected co-locations (exclude self)
        disc = cfg.gamma_disc ** h
        for i in range(N):
            s = 0.0
            for e, p in dists[i].items():
                s += p * (occ.get(e, 0.0) - p)
            gamma[i] += disc * s
    if gamma.max() > 0:
        gamma = gamma / (gamma.mean() + 1e-9)
    return gamma
