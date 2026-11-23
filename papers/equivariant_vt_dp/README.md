# VISTA (arXiv:2610.03333, CoRL 2026 포스터) 분석 노트

*Equivariant Visual-Tactile Diffusion Policy for Contact-Rich Manipulation* (arXiv v1, 2026-10-02)을 표 재계산, 통계 검정, 공식 코드 정적 대조, 동변성 토이로 분석한 자료입니다. 논문 원문 PDF와 논문 그림은 넣지 않았습니다(arXiv:2610.03333에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **시뮬 학습과 평가는 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** NVIDIA GPU 리눅스, Isaac Sim 4.5, cuRobo가 필요하고 시연 데이터는 124.8 GB입니다.
- **공개 체크포인트가 없어 체크포인트 추론도 불가입니다.**
- **공식 코드로 오프라인 학습은 조건부입니다.** 설치와 데이터 다운로드가 필요하고 MPS 호환은 확인하지 못했습니다.
- **구조 소형 재구현과 동변성 토이는 가능합니다.** `analysis/toy_equivariance.py`(수 초, numpy).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 우리가 그린 그래프 다섯 장) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 검정, 토이 결과 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 구현 가능성, 라이선스 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_equivariance.py` | 토이: 식 2–4와 명제 1, 2의 구조 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 |
| `report/figures/` | 우리가 그린 그래프 다섯 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_equivariance.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스로 공개되어 있습니다(CC BY가 아님, arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다. 논문 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/Kenn3o3/Vista`(시뮬), `github.com/Kenn3o3/vt_franka`(실로봇)에 있습니다. 루트는 Apache-2.0이고 ISP에서 가져온 부분은 학술·교육 한정 라이선스입니다. 이 저장소에는 코드를 넣지 않았고 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.
