# GPT-as-Tactile-Policy (arXiv:2610.28457) 분석 노트

*GPT-6-Astra Understands Touch and Outperform Vision–Tactile–Language–Action Models* (Wang et al., 2026)을
표 재계산으로 검토하고, 같은 구조를 재구현해 UniVTAC 폐루프에서 메커니즘을 떼어 본 자료입니다.
실험은 GPT-6-Astra 대신 Claude Opus 5.5로 했으므로 원문 수치의 재현이 아닙니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations, 리버스 엔지니어링 |
| `analysis/02_prior_work.md` | 인용 문헌 검증, 계보, 누락된 관련 연구 |
| `analysis/04_system_architecture.md` | 재구현 시스템 구성도, 논문 ↔ 모듈 매핑 |
| `analysis/05_claim_verification.md` | 주장 판정, 통계 재계산 (`python3 analysis/stats_check.py`) |
| `analysis/06_code_verification.md` | 재구현 코드의 테스트·E2E·정적 점검 |
| `analysis/07_experiments.md` | 실제 LLM 촉각 probe, UniVTAC 폐루프 2×2, 반사실(거울 반전) 촉각 실험 |
| `analysis/extra_physics_descriptors.py`, `analysis/extra_physics_descriptors_results.json` | UniVTAC 데모 30개에서 센서 자세로 옮긴 전단 등 물리 descriptor의 방향 판별력 (`07` 추가 검증). 데모 원자료는 한 파일씩 받아 처리하고 지우며 저장소에 남기지 않음 |
| `analysis/extra_physics_fields_check.py`, `analysis/extra_physics_fields_check_results.json` | 공개 데모 에피소드 하나의 HDF5 데이터셋 20개를 확인해 힘·재질 필드가 없음을 기록 (`07` 추가 검증 2026-11-14). 파일은 받아서 지움 |

문서 번호 03(연구 방향)은 이 공개본에 넣지 않았습니다. 05는 이 폴더의 이전 형식에서 주장 검증 문서입니다.

## 재실행

```bash
pip install numpy scipy
python3 analysis/stats_check.py      # 원문 표 통계 재계산 (몇 초)
```

재구현 패키지와 실험 원본 로그는 이 공개본에 넣지 않았습니다.

## 출처와 라이선스

- 논문은 CC BY가 아닙니다(arXiv 비독점 배포 라이선스). 그래서 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- `report/figures/chart_tab2.png`는 논문 Tab.2의 수치로, 나머지 `chart_*.png`는 우리 실험 결과로 직접 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
