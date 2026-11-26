# NativeMEM (arXiv:2607.06678, CoRL 2026 포스터) 분석 노트

*NativeMEM: Native Memory Compression for Long-Horizon Robotic Manipulation* (arXiv v2, 2026-10-16)을 표 재계산, 통계 검정, 공개 코드 정적 대조로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2607.06678에서 받을 수 있습니다). 논문 그림은 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 분석에서 하지 않았고 일반 노트북에서는 어렵습니다.** 학습은 H100 여러 장, 시뮬레이션 평가는 CUDA 시뮬레이터, 실로봇은 공개되지 않은 설정이 필요합니다.
- **공개 체크포인트(약 63.6 GB)로 추론하려면 새 환경 설치와 다운로드가 필요합니다.** 이 분석에서는 받지 않았습니다.
- **표 재계산과 토큰 산술은 가능합니다.** `analysis/recompute.py`(수 초)로 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림은 그림 4 한 장만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |
| `report/figures/` | 그래프 네 장(우리가 그림)과 논문 그림 한 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 한 장(그림 4)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2607.06678"). 로봇 사진, 시뮬레이션·실물 프레임이 들어 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/OpenDriveLab/NativeMEM`(Apache-2.0)에 있고 시뮬레이션 평가는 `github.com/Ziyeeee/NativeMEM-RMBench`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.
