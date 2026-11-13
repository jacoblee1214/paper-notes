# V2D 튜토리얼 (CoRL 2026) 분석 노트

*From Human Videos to Dexterous Robot Capability* 튜토리얼 웹 페이지와 챌린지 페이지,
그리고 저장소 `nvidia-isaac/video_to_data`(커밋 `5226e34`, 2026-11-09)를 읽고 정리한 자료입니다.
**심사를 거친 논문이 아니라 튜토리얼 페이지와 저장소입니다.** 접근일 2026-11-12.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_page_analysis.md` | 페이지 주장, 일정, 챌린지 트랙·지표·규칙, 리더보드 집계 |
| `analysis/02_prior_work.md` | 계보, 관련 저장소, GitHub 표, 라이선스 표 |
| `analysis/03_analysis.md` | 다섯 검증 게이트 분석, 숫자 점검, 기하 게이트 toy |
| `analysis/04_code_verification.md` | 페이지 주장 대 코드, 불일치 목록, 구현 가능성 |
| `analysis/gate_toy.py` | 기하 게이트 toy (numpy, 약 4초) → `gate_toy_results.json` |
| `analysis/page_checks.py` | 숫자 점검 → `page_checks_results.json` (키트 경로를 주면 상수 대조) |
| `analysis/plot_figures.py` | toy 그림 생성 |
| `analysis/extra_kit_metrics.py` | 키트 지표 numpy 포트, 알려진 답 테스트, SP-SR 정의 비교, 떨림 곡선 (numpy, 약 4초) → `extra_kit_metrics_results.json`, `extra_kit_jitter.png` |
| `analysis/extra_lift_hold.py` | lift-hold 성공 판정 numpy 포트, 합성 높이 트레이스 알려진 답, 프레임 속도·잡음·임계 교환 (numpy, 약 1초) → `extra_lift_hold_results.json` |
| `report/figures/` | 직접 계산한 그림 5장 |

## 구현 가능성

맥(CUDA 없음)에서 튜토리얼 파이프라인 재현은 불가능합니다. 저장소는 CUDA, Docker, Isaac Lab을 전제로 합니다.
CPU로 가능한 것은 페이지 숫자 점검, 키트 지표 상수 대조, 기하 게이트 toy입니다.

## 재실행 (CPU)

```bash
python3 -I analysis/gate_toy.py
python3 -I analysis/plot_figures.py
python3 -I analysis/extra_kit_metrics.py
python3 -I analysis/extra_lift_hold.py
python3 -I analysis/page_checks.py -
```

## 출처와 라이선스

- 저장소 `LICENSE`: 문서·스킬 CC BY 4.0, 소스 Apache-2.0. 재구성 제3자 모델 다수는 비상업 또는 비파생 조건입니다.
- 페이지 미디어(사진, 그림, 영상)와 챌린지 키트·FAQ는 포함하지 않았습니다.
- 그림과 코드는 이 분석에서 직접 만든 것입니다.
