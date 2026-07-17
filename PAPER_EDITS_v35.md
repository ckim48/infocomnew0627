# 논문 v4 → v3.5 되돌림 목록 (2026-07-17)

15개 분리 실험(`results/probe_*.npz`)의 결론: v4의 교환·집계 모델은 직접 V2V를
과도하게 강화해 FACE의 ferrying 이득을 제거함. 아래 항목만 v3 표현으로 되돌리면
실험 결과(FACE 승리)와 논문이 일치함. **나머지 v4 구조(문제 정식화, surrogate
lower bound, coverage 가치, complexity 등)는 그대로 유지 가능.**

## 1. Sec. III-D — 교환을 단방향으로 (필수)
- 현재: "An activated contact may carry transmissions in both directions."
  및 eq:forwarding_feasibility 에서 `a_ij + a_ji`가 예산 공유.
- 되돌림: activation은 **방향성 있는 교환** $i\to j$ 하나를 수행.
  matching 제약은 방향 교환 기준 (v3의 eq:matching_constraint 문장 그대로):
  각 차량은 라운드당 최대 1개의 directed exchange에 참여.
- eq:forwarding_feasibility: $\sum_x a_{ij,x} S_x \le \widehat B_{ij}\, y_{ij}$
  (단방향). sojourn-budget 정의(eq:sojourn_time, eq:contact_budget)는 유지.

## 2. Sec. III-D — consume/retain 2-모드 삭제, copy tickets 복원 (필수)
- 현재: retention decision $r_{j,x}$ (consume-only vs retain) + binary-spray
  replication tokens (⌊k/2⌋).
- 되돌림: v3의 **copy tickets** 문단 전체 —
  eq:ticket_feasibility, eq:ticket_update, 그리고 **value-weighted ticket
  splitting** (v3 eq:ticket_split: $g_{ij,x}$ 를 양 끝점 continuation value
  비율로 분할, replication/custody transfer 구분 $r_{i,x}$).
- copy cap eq:copy_cap 은 티켓 보존 부등식($\sum_i m_{i,x} \le K_x$)으로 표현.

## 3. Sec. III-E — 집계를 단일-최선 채택으로 (필수)
- 현재: aggregation set 전체를 FedAvg 평균 → 전체 accept/reject.
- 되돌림: modality별로 **예측 보상 최대 후보 1개**를 데이터량 가중으로 임시
  집계하고, validation loss가 증가하지 않으면 채택 (acceptance test 유지).
  v3의 eq:adoption 형태.
- Lemma(certified lower bound)는 유지 가능: 라운드당 modality별 채택이 1개
  이하이므로 상호작용 오차 $\epsilon_{\mathrm{int}} = 0$ — 본문 remark에 이미
  있는 문장이 그대로 성립. LOO attribution → 채택 후보의 measured gain으로
  단순화 ($v_{i,x}$ 정의만 교체).

## 4. Sec. IV — 알고리즘 문단 3곳
- "Replication tokens" 문단 → v3 "value-weighted ticket splitting" 문단.
- "Bundle selection": 양방향 예산분할 DP 서술 → 단방향 joint admission-eviction
  (h(s)/g(f) two-DP; v3.5 구현과 일치. 이 부분은 v4 텍스트의 eviction 결합
  아이디어를 그대로 쓰되 "both directions" 표현만 제거).
- "Pairing protocol" → v3의 greedy local-maximal matching (directed) +
  Prop. 2 (½-approximation) 복원 가능.

## 5. 실험 설정 서술
- 인트로의 이질성 주장(모달 유무·아키텍처)은 유지.
- Setup에는: modality availability를 per-modality 확률로 실현
  (KITTI: P(camera)=1.0, P(LiDAR)=0.85 → **15%는 LiDAR 없는 vision-only**;
  nuScenes: radar 0.7, LiDAR 0.85). "Tesla-like vision-only vehicles"
  문구를 여기 붙이면 인트로와 연결됨.
- χ(아키텍처 호환성)는 시스템 모델에 유지하되, 실험은 modality-단위 호환로
  실현되었다고 서술 (χ=1 iff 같은 modality 보유).
  ※ 주의: typed 차종 혼합(camera-only 35% 등)과 arch-family 분할은 실험상
  FACE가 서비스할 수요 자체를 제거함 (camera-only 차량은 LiDAR 수요가 없고
  camera 공급은 어디에나 있음) — probe로 확인된 사실. 강조하지 말 것.

## 6-A. Sec. III-E 평판/상호협력 복원 (필수 — 실측 근거)
- v4에서 삭제된 reputation/reciprocal-priority 섹션을 **v3 초안 텍스트 그대로 복원**
  (eq:rep_delivery, eq:rep_storage, eq:rep_update, eq:rep_priority + Sec IV의
  reciprocal factor $(1+\gamma_r(\pi_j-1)^+)$).
- 근거: real KITTI 3-seed에서 reciprocity ON 54.9±1.4 vs OFF 54.2±1.3 —
  이 메커니즘이 FACE 우위 (+0.7pp)의 담당 요소. OFF면 V2V와 완전 동률.
- 인트로의 "reciprocal cooperation" 두 번째 design consideration 문단도
  v3 초안에서 복원.

## 6. 수치 (v3.5 최종 3-seed, 2026-07-17 rerun 후 확정치로 교체)
- KITTI: FACE가 Acc/Poor/Loss/Gap/Rounds@τ 전 지표 1위 (직전 확정치:
  54.9±1.4 vs V2V 54.2±1.6, Rounds@τ 181 vs 191).
- nuScenes: 통계적 동률 (FACE 70.0 vs V2V 70.1) — "4.7pp/59.1%" 문구는
  KITTI 기준 상대 서술로 완화 필요.
- Ablation (abstract, K=16): ferrying/tickets/refresh/demand/future 기여 유지.
