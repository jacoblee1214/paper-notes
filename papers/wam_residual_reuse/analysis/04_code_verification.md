# 04. 코드 대조: 논문 ↔ 공식 저장소 (정적 읽기)

- 대상: `repo/` (gabhinav001/cache-or-denoise-less, 커밋 `71867934ea`, 2026-11-08). 희소 체크아웃으로 `README.md`, `probes/`, `repro/`(스크립트, 로그, 기대 표·그림 데이터), `examples/`, `results/budget_shift/summary.json`만 받았습니다.
- **실행하지 않았습니다.** `import`, 설치, 체크포인트나 LIBERO 설정도 하지 않았습니다. 저장소 내용은 신뢰하지 않는 데이터로 다루었습니다.
- 주요 파일: `probes/fw_libero.py`(966줄, 롤아웃, 캐시 하네스, 지연, 스케줄 배치), `probes/gate_cache.py`(수집 전 동일성 게이트), `repro/experiments/composed2step.sh`(복합 캐시 설정).
- 차이 항목은 D1–Dn으로 붙였습니다.

## 1. 대조 결과

| 번호 | 논문 주장 | 코드 | 판정 |
|---|---|---|---|
| D1 | 식 (1) `σ_k = s·u/(1+(s−1)·u)`, 캐시 비율 = 1 − σ_m (§2, A.3) | `probes/fw_libero.py` 435–448행 `cached_path_fraction`: `u = 1 − prefix/n`, `1 − s·u/(1+(s−1)·u)` | **일치** (수식과 docstring "8스텝 80% (s=1), 44% (s=5)"까지 같음) |
| D2 | 스텝 0–7 캐시, 마지막 두 스텝은 정확 계산 (§2) | 451행 `placements`의 "every block, all but the last two denoise steps" 규칙. K=10에서 [0,7] | **일치 (K 일반화 방식은 본문에 없음)**. 코드는 K에 따라 "마지막 두 스텝 제외" 규칙을 일반화하므로 K=2에서는 캐시가 0이 됩니다. 복합 캐시(K=2, 스텝 0 캐시)는 `composed2step.sh`에서 `@prefix=1`로 덮어씁니다. 논문 본문은 이 덮어쓰기를 "스텝 0 캐시"라고만 써서 규칙 차이가 드러나지 않습니다 |
| D3 | 게이트: 재생 행동이 무캐시와 상대 L2 0.5% 이내 (A.10) | `probes/gate_cache.py` 33행 `MAX_IDENTITY_ERR = 5e-3`, 주석에 "재회수된 메모리 캐시는 1e-1대" | **일치** |
| D4 | 전체 스택 잔차 `h_30 = h_0 + R` (C³ache 방식, §2) | 여러 블록 잔차를 더하는 `CacheHarness`와 별도로 `StackHarness`(263–300행)가 30블록 전체 잔차를 한 번 더하고, 블록 단위 대비 BF16 반올림 차이를 주석으로 남김 | **일치**. 논문이 블록 단위 캐시도 따로 쟀다는 점(A.8)과 맞음 |
| D5 | 기본 설정: 호출마다 같은 초기 잠재 노이즈 (§2, §4) | 510–514행: "FastWAM builds its initial action latent from a generator seeded with cfg.seed on EVERY call". `resample_noise=1`이면 호출마다 새 노이즈 | **일치** |
| D6 | 에피소드 잠재 대조군: 에피소드마다 첫 노이즈를 모든 호출에 사용 (§4, A.5) | 518–526행: 에피소드별 재시드, `episode_noise` 분기 | **일치** (구현 세부는 `fw_libero.py` 전체를 읽지 않아 완전 확인은 아님) |
| D7 | 지연 실행: 호출이 d 제어 스텝 늦으면 앞 d개 행동을 이전 청크의 행동으로 대체 (§3, A.5) | 418–424행 `delayed_actions`: 스텝 t..t+d−1에 이전 청크의 `replan..replan+d−1` 인덱스 행동 사용, 첫 청크 전에는 정지 행동 | **일치**. 404–414행 `parse_delay`는 갱신 후와 재사용 후 지연을 따로 받아 A7의 "5/2" 표기와 맞음 |
| D8 | CUDA 그래프에서 캐시된 잔차가 호출 사이에 덮어써짐 (A.8) | 36–43행 주석: 그래프가 소유한 메모리는 다음 실행에서 회수되므로 잔차를 그래프 밖으로 복사해야 함 | **일치**. 논문은 이 비용을 측정하지 않았다고 씀 (코드 주석과 같음) |
| D9 | 실행 재현성: 249쌍 중 11,725개 쌍체 결과에서 7개 변경 (A.11) | `results/budget_shift/summary.json` `determinism_exhaustive`: pairs 249, paired_episodes 11725, outcome_changes 7 | **일치**. 같은 파일의 `determinism_total`(36 비교, 2400 쌍체, 변경 0)은 범위가 다른 별도 집계이며 논문과 모순되지 않음 |

## 2. 논문에 없고 저장소에 있는 것

- **라이선스 파일이 없습니다.** GitHub API는 `license: null`을 돌려줍니다. 재배포 권한이 불분명하므로 우리 폴더에는 코드를 복사하지 않았습니다.
- **`inputs/source_bundle.tar.gz`(73.75 MB)**는 원정책 번들로 보입니다. 내용을 열지 않았고, 희소 체크아웃에서도 제외했습니다.
- **`results/`의 raw 파일**(수십 MB, 예: `fw_families.npz` 27 MB)은 받지 않았습니다. 대신 `results/budget_shift/summary.json`(171 KB)은 받았습니다.

## 3. 대조에서 확인하지 못한 것

- `probes/make_figure.py`, `make_tables.py`가 만드는 그림과 표는 읽지 않았습니다. 논문 그림은 우리가 쓰지 않기 때문입니다.
- Tango 구간 계산(`probes/` 안의 구간 함수)은 읽었지만 V3에서 우리가 맞추지 못했습니다. 코드에 있는 식과 우리 식이 같은지는 확인하지 않았습니다. 이 항목은 §03 V3의 미검증 이유입니다.
- 원본 체크포인트의 "first-frame" 모드(IDM)와 `commit 0050a44`의 실제 차이는 읽지 않았습니다.

## 4. 결론

코드는 논문 본문의 핵심 정의(식 (1), 캐시 규칙, 게이트 임계값, 노이즈 재샘플 정책, 지연 실행)와 **맞습니다**. 차이가 있다면 D2의 K 일반화 규칙이 본문에 없다는 것 한 가지입니다. 재현 가치 판단은 개인용 연구 문서에서 다룹니다.
