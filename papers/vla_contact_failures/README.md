# VLA 접촉 실패 분석 (arXiv:2608.01402, CoRL 2026 Spotlight) 노트

*Demystifying When and Why VLAs Fail in Contact-Rich Tasks and How to Fix Them* (arXiv v2, 2026-10-16)을 표 재계산, 다중 비교 보정, 검정력 토이, 공식 코드 정적 대조로 읽은 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(라이선스 CC BY-SA 4.0).

## 구현 가능성 (요약)

- **공식 재현: 이 맥에서는 불가.** 공식 코드(`stanford-iprl-lab/fact`, Apache-2.0)는 JAX 전용이고 NVIDIA GPU가 필요합니다. 학습 데이터와 실로봇도 필요합니다.
- **정적 대조: 가능, 완료.** LN 샘플러, 힘 AdaRMS, TCN, 창 길이는 논문과 맞습니다. 접촉 게이트 범위(이력 토큰 제외), 학습 스텝(30k 대 20k), π0 힘 조건, 기준선 재구현, 노이즈 대체 코드는 공개본 기준으로 다르거나 없습니다.
- **CPU 토이:** LN 질량 토이와 40회 검정 토이는 수 초 안에 돕니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, Holm 보정, 검정력, 주장 일관성 판정, 토이 결과 |
| `analysis/04_code_verification.md` | 논문과 공식 코드 대조, 구현 가능성, 라이선스 |
| `analysis/table_check.py` | 표 재계산과 검정 (scipy, 수 초) |
| `analysis/toy_schedule.py` | LN과 Beta의 노이즈 질량 지도 (numpy, 1초 미만) |
| `analysis/toy_power.py` | 40회 두 팔 검정의 검정력과 보정 없는 다중 비교 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib markdown-it-py
python3 analysis/table_check.py
python3 analysis/toy_schedule.py
python3 analysis/toy_power.py
python3 analysis/plot_figures.py
```

원문 PDF는 다음 명령으로 받을 수 있습니다(라이선스 CC BY-SA 4.0): `curl -L -o paper.pdf https://arxiv.org/pdf/2608.01402v2`

## 출처와 라이선스

- 논문은 CC BY-SA 4.0입니다(arXiv abs 페이지 확인). 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다.
- 논문의 표 수치는 인용과 재계산을 위해 옮겼습니다. 문장과 그림은 옮기지 않았습니다.
- 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다. 토이의 파라미터는 논문 식과 Beta(1.5, 1)에서 가져왔고 설계 가정은 없습니다. 검정력 토이의 가정(기준 0.30, 참 차이 0.20, 40회)은 명시되어 있습니다.
- 공식 코드는 읽기만 했습니다. 실행, 설치, 가중치 다운로드는 하지 않았습니다.
