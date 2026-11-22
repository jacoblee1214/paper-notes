# ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning (arXiv:2503.21860, CVPR 2025) 분석 노트

*ManipTrans* (arXiv v1, 2025-03-27)를 표 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 두 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 코드는 Isaac Gym(Linux, NVIDIA GPU)과 CUDA 빌드 pytorch3d가 필요합니다. 사전학습은 단일 GPU에서 약 1.5일입니다.
- **CPU에서 가능한 것:** 평가 지표 계산(MANO 키포인트 데이터가 있을 때), MuJoCo 기반 작은 추적 토이(1–2주로 추정, 측정하지 않음).
- **데이터:** DexManipNet은 8.31 GB입니다. MANO, SMPL-X, OakInk-V2는 등록이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈 |
| `analysis/02_prior_work.md` | 학회 확인, 선행 계보와 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(분모, 차이 검정, 노이즈 표 단조성, 시간 검산) |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치와 빈칸, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (표준 라이브러리만, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 (matplotlib) |

## 재실행

```bash
python3 analysis/recompute.py
pip install matplotlib
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/ManipTrans/ManipTrans)는 GPL-3.0입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 데이터셋과 체크포인트는 받지 않았습니다.
