# DexMimicGen (arXiv:2410.24185, ICRA 2025) 분석 노트

*DexMimicGen: Automated Data Generation for Bimanual Dexterous Manipulation via Imitation Learning* (arXiv v2, 2025-03-06)를 공식 코드 정적 대조, 표 재계산, 통계 검정으로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 표 값으로 그린 것입니다.

## 구현 가능성 (요약)

- **공개 코드에는 논문의 데이터 생성 방법이 없습니다.** 공개 저장소는 시뮬레이션 환경 9개, 재생 스크립트, BC-RNN 학습 설정입니다.
- **생성기 재구현은 이 맥에서 조건부로 가능합니다.** 환경 설치와 재생은 CPU로 가능하지만 설치 승인이 먼저 필요합니다. 생성기는 논문 설명으로 다시 짜야 합니다.
- **학습과 실로봇 재현은 이 맥에서 불가능합니다.** BC-RNN과 DP 학습은 GPU를 전제하고, 실로봇은 하드웨어가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 결과, 부록, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산(표 사이 일치, 버전 차이, 통계, 실로봇 구간), 주장 판정 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 차이 목록, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 값 옮김, 일치 검사, 검정, 구간 (numpy, scipy, 1초 미만) |
| `analysis/recompute_results.json` | 위 스크립트의 결과 |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-22). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 학회 표기 ICRA 2025는 IEEE DOI 메타데이터(Crossref)로 확인했습니다.
- 공식 코드(https://github.com/NVlabs/dexmimicgen)는 NVIDIA Source Code License(비상업 연구·평가용)입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 데이터셋(Hugging Face 카드 기준 CC BY-NC-SA 4.0, README는 CC-BY 4.0이라고 씀)은 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 검사 스크립트, 분석 문서는 이 저장소 작성자가 만들었습니다.
