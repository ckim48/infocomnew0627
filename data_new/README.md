# data_new — fig_seoul_map를 외부에서 그대로 그리기 위한 데이터

`fig_seoul_map(.pdf)`(abstract utility 2×2)와 `fig_seoul_map_realacc*`
(real-backend accuracy 변형)을 시뮬레이터 없이 재현하는 데 필요한 전부.
생성: `python3 data_new/export_seoul_map_data.py`
검증용 단독 플로터(이 폴더의 CSV만 읽음, 오프라인 동작):

```bash
python3 data_new/plot_seoul_map.py --metric utility          # 도로 CSV 배경
python3 data_new/plot_seoul_map.py --metric realacc --tiles  # 논문과 같은 타일 배경
```

→ `preview_seoul_map_utility.png`가 검증 결과물.

## 파일

- `seoul_map_utility.csv` — 차량 180대 × (lon, lat, x/y_web_mercator,
  heading_deg, utility_{FACE,CachedDFL,V2V,LearningAware}).
  fig_seoul_map의 정확한 플롯 값 (abstract q_eff, seed 2026,
  `results/v2x_map_cache.npz`).
- `seoul_map_realacc.csv` — 같은 위치/헤딩에 per-scheme 실측 테스트 정확도
  (400라운드 real run, `results/metrics_v2x_real_kitti_map400.npz`).
- `seoul_roads.csv` — 그림 extent 안의 SUMO 도로망 폴리라인 (120,490
  vertex). 컬럼: edge_id, seq(폴리라인 순서), priority(선굵기 스타일용),
  lon, lat, x/y_web_mercator. edge_id로 groupby → seq 정렬 → LineString.
- `seoul_car_glyph.csv` — 차량 글리프 다각형 (unit 길이, +x 방향).
  body 8점 + cabin 5점. 그리는 법: heading_deg로 회전 → car_length_m로
  스케일 → (x_web_mercator, y_web_mercator)로 평행이동.
  cabin은 검정 alpha 0.35 오버레이.
- `seoul_map_meta.csv` — 나머지 전부: extent(xlim/ylim, EPSG:3857),
  베이스맵(CartoDB Positron, zoom 15, 타일 URL), colormap RdYlGn과
  vmin/vmax(utility 0.2–1.0, realacc 0.05–0.95), car_length 규칙
  (extent 폭/42), halo/테두리/colorbar 스타일, 패널 순서, 데이터 출처.

## 좌표계

CSV의 `x/y_web_mercator`는 EPSG:3857 (타일 배경과 바로 호환),
`lon/lat`은 WGS84. 자체 배경(위성 등)을 쓰려면 lon/lat을 원하는
투영으로 변환하면 됨.
