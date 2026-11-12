# PHUMA: Physically Reliable Humanoid Locomotion Dataset (arXiv:2510.26236) 분석 노트

휴머노이드 보행 데이터셋 PHUMA의 큐레이션 기준, 리타게팅 손실, 추적 실험 수치를 표 재계산과 장난감 실험으로 점검한 자료입니다.
논문은 CoRL 2026 Spotlight 표기(원문 머리글, 프로젝트 페이지, 저장소 README 일치)이고, 공개 저장소는 Apache-2.0입니다.

## 판정 요약

- 공식 파이프라인 전체 재현은 불가능합니다(SMPL-X 등록, 원본 사람 포즈 비공개, 리타게팅 단일 GPU 약 99일 추정, 정책 학습은 CUDA와 Isaac 계열 시뮬레이터 필요).
- 논문 표의 통계는 대부분 단일 시드입니다. 방법 간 차이 중 일부는 평가 표본 크기 안에 있습니다. 예: Unseen 504개에서 PhySINK와 SINK 차이는 p ≈ 0.36입니다.
- 한 표(AMASS Unseen 총합)의 값이 본문과 다르게 보입니다(78.2 대 70.2).
- 리타게팅 품질 지표는 손실과 같은 양을 재므로 독립 증거가 아닙니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 Figure 2 한 장 포함, 출처 표기) |
| `analysis/01_paper_analysis.md` | 논문 분석: 방법, 실험, 한계, v1 대 v3 차이 |
| `analysis/02_prior_work.md` | 선행연구, 공개 저장소와 데이터 확인 표 |
| `analysis/03_analysis.md` | 표 재계산, 통계 검정, 지표 유도, 장난감 결과 |
| `analysis/04_code_verification.md` | 공개 코드 정적 대조 (D1–D22), 구현 가능성 판정 |
| `analysis/table_check.py` | 표 재계산과 검정 (수 초, numpy·scipy) |
| `analysis/sensitivity_toy.py` | 지표 임계·프레임레이트 장난감 (수 초, numpy·scipy) |
| `analysis/plot_figures.py` | 직접 만든 그래프 (matplotlib) |
| `report/crop_figures.py` | 논문 PDF에서 Figure 2 자르기 (원본 PDF는 포함하지 않음) |

## 출처와 라이선스

- 논문: Kyungmin Lee 외, "PHUMA: Physically Reliable Humanoid Locomotion Dataset", arXiv:2510.26236 v3, CC BY 4.0. https://arxiv.org/abs/2510.26236
- 논문 Figure 2: 원 저자, CC BY 4.0. 이 저장소에서는 PDF에서 잘라 캡션과 함께 넣었습니다.
- 공개 코드: https://github.com/davian-robotics/PHUMA (Apache-2.0). 이 저장소는 코드를 복사하지 않았고, 실행하지 않고 정적으로만 읽었습니다.
- 데이터: Hugging Face DAVIAN-Robotics/PHUMA (카드 표기 Apache-2.0, 원천 약관은 확인하지 않음). **데이터는 내려받지 않았고 이 저장소에 없습니다.**
- 직접 만든 그래프와 장난감 실험은 이 저장소 작성자가 만들었습니다(논문 데이터 아님).

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/sensitivity_toy.py
python3 analysis/plot_figures.py   # report/figures/ 에 그래프를 만듦
```

`crop_figures.py`는 원본 논문 PDF가 있어야 실행됩니다. PDF는 이 저장소에 없습니다.
