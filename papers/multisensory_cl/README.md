# MuSe (arXiv:2606.30988, CoRL 2026 포스터) 분석 노트

*Multisensory Continual Learning: Adapting Pretrained Visuomotor Policies to Force* (arXiv v4, 2026-10-06)를 표 재계산과 통계 검정으로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.30988에서 받을 수 있습니다). 논문 그림은 세 장(도식과 그래프 부분만)을 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **실로봇 결과 재현은 이 맥에서 불가입니다.** UR5e와 UMI-FT 장치, 비공개 데이터가 필요합니다.
- **MuSe 학습도 불가입니다.** MAR-Large 479M을 GPU 8장으로 학습했고 스텝 수가 적혀 있지 않습니다.
- **공식 코드는 아직 없습니다.** 프로젝트 페이지의 Code 링크가 가리키는 저장소는 "Code to be released" 한 줄입니다(2026-11-24).
- **표 재계산은 가능하고 완료했습니다.** `analysis/recompute.py`(수 초, numpy, scipy)입니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 세 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 검정 |
| `analysis/04_claim_verification.md` | 주장 검증, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 세 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF에서 상충하는 권리 고지를 찾지 못했습니다.
- 논문 그림 세 장은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2606.30988"). 그림 2의 구조 도식, 그림 3 아래쪽 막대그래프, 그림 4입니다. 실로봇·과제 사진과 데이터셋 프레임은 넣지 않았습니다.
- 공식 모델 코드는 공개되어 있지 않습니다. 이 저장소에는 외부 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.
