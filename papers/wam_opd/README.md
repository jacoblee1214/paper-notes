# WAM-OPD: Joint Video-Action Supervision for World Action Model Post-Training with On-Policy Distillation (arXiv:2608.22364) 분석 노트

*WAM-OPD* (arXiv v2, 2026-10-01, 학회 표기 없음)를 표 재계산, 공식 코드 정적 대조로 분석한 자료입니다. 논문은 arXiv에서 CC BY 4.0으로 공개되어 있습니다. 이 폴더에는 원문 PDF와 실물·모의 프레임이 있는 논문 그림은 넣지 않았고, 저자의 막대그래프 한 장(Fig. 4)을 출처와 함께 넣었습니다. 나머지 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 파이프라인은 수집 GPU에 65 GB 이상 여유 메모리를 요구하고, 절제·Teacher 이력·실로봇 경로는 공개 코드에 없습니다.
- 장난감 축소판은 조건부로 가능합니다(노력 1–2주 추정).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 정수 복원, 검정, 검출 한계, Teacher 대조, 버전 대조 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조(K1–K16, D1–D18), 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문: arXiv:2608.22364, 라이선스 CC BY 4.0(abs 페이지 확인, 2026-11-22). 그림 `report/figures/paper_fig4_ablation.png`는 논문 Fig. 4를 잘라 쓴 것입니다. © Yang 외, CC BY 4.0, 출처 arXiv:2608.22364 v2.
- 공식 코드(https://github.com/UCL-ERL/WAM-OPD)는 Apache-2.0입니다. 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
