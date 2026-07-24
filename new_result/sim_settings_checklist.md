# Vehicular/network settings — reviewer checklist answers

Source: results/contact_stats.npz, data/gangnam/seoul_v2x_trace.npz,
sim/face.py (matching/transfer), sim/config.py. Contact-duration and
density computed from the trace (150 m V2V range, dt = 10.22 s).

| 항목 | 값 |
|---|---|
| 전체 vehicle 수 | N = 180 (시내 전체 1,042개 V2X 단말 중 강남 영역 선별) |
| Simulation duration | 250 rounds x 10.22 s ~= 42.6 min |
| Round 시간 Δ_rd | 10.22 s (API 폴링 주기 중앙값) |
| 총 round 수 | 250 (trace K = 89 rounds 순환 재생, steady-state) |
| Trace 시간대 | 2026-06-30 18:42-18:57 KST (저녁 첨두, 15.2 min) |
| Vehicle density | 0.79 veh/km^2 (180대 / 16.8 x 13.5 km = 227 km^2); 차량 수 상수 — 라운드별 평균 이웃 수 1.07-1.66으로 변동 표현 |
| Contacts /veh/round | mean 1.28 (median 1, max 10); 라운드당 접촉 보유 52.9% |
| Contact duration | mean 73 s, median 41 s, p90 133 s |
| Matching | greedy max-weight matching, half-duplex single-peer; committed exchange = contact graph의 matching, greedy 1/2-근사 (Prop. 2) |
| 실패 transmission | (i) airtime 예산 초과 S/(r·p_tx) > 1.6 s → 전송 포기, (ii) Bernoulli 링크 실패 (성공확률 p_tx = 1-(d/r)^2) → airtime 소모·미전달; copy는 발신측에 남아 다음 라운드 재후보 |
| 양방향 전송 | 불가 — directed transfer; commit 시 양 끝점 모두 해당 라운드 contention 제외 |

## LaTeX snippet (Sec. V-A 보강)

The trace covers $N{=}180$ vehicles (selected from 1{,}042 city-wide
V2X terminals) over a $16.8\times13.5$\,km area (0.79\,veh/km$^2$),
collected on a weekday evening peak (18:42--18:57 KST; 89 snapshots at
$\Delta_{\mathrm{rd}}{=}10.2$\,s). FL runs $T{=}250$ rounds by
replaying the window cyclically (steady-state traffic),
$\approx$43\,min of driving. With the 150\,m V2V range a vehicle sees
1.28 in-range peers per round on average (median 1, max 10) and holds
at least one contact in 53\% of rounds; contacts persist for 73\,s on
average (median 41\,s, 90th pct.\ 133\,s). Contention is resolved by
greedy max-weight matching under the half-duplex single-peer
constraint; each committed exchange is directed and removes both
endpoints from contention for the round. A transfer whose airtime
$S_x/(r\,p^{\mathrm{tx}}_{ij})$ exceeds $\bar T^{\mathrm{con}}{=}1.6$\,s
is aborted, and completed airtime succeeds with probability
$p^{\mathrm{tx}}_{ij}$; in either case the airtime is lost while the
copy remains at the sender and re-enters candidacy in later rounds.
