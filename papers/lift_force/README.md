# LIFT (arXiv:2607.14236, CoRL 2026 Spotlight) 분석 노트

*Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection* (arXiv v3, 2026-10-12)을 본문 수치 재계산, 통계 검정, 공식 코드의 정적 대조, 소형 numpy 토이로 분석한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다(라이선스 CC BY-SA 4.0).

## 구현 가능성 (요약)

- **구조는 구현할 수 있습니다.** 반응 expert(기존 가중치 복사), 지연 정렬 shifted causal mask, 힘 GRU, 0 초기화 교차어텐션은 소형 PyTorch 모델로 옮길 수 있습니다(추정 3–5일).
- **논문 수치 재현은 이 맥에서 불가합니다.** 공식 코드는 Linux와 CUDA GPU를 전제로 하고, 학습은 방법당 8-GPU 노드가 필요합니다.
- **실로봇 재현은 데이터와 장비가 필요합니다.** 6D 힘 센서, 원격조작 교정 도구, 수집 도구는 공개되지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, v1에서 v3로의 변화 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub 표 |
| `analysis/03_analysis.md` | 수치 재계산, 일관성 검사, 검정, 토이 결과, 힘 지연 산술 |
| `analysis/04_code_verification.md` | 논문과 코드의 대조, 주장 검증, 저자에게 물을 질문, 구현 가능성 |
| `analysis/table_check.py` | 수치 재계산과 검정 (scipy, 수 초) |
| `analysis/toy_init_equiv.py` | 마스크 규칙과 초기화 동일성 토이 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/toy_init_equiv.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 CC BY-SA 4.0입니다(arXiv abs 페이지 확인). 논문 그림과 원문 PDF는 이 저장소에 넣지 않았습니다. 원문은 arXiv:2607.14236에서 받을 수 있습니다.
- 공식 코드(`y-wng/LIFT`)는 Apache-2.0입니다. Gemma 약관 파일이 함께 있습니다. 이 분석에서는 정적으로만 읽었고 실행하지 않았습니다. 가중치와 데이터는 받지 않았습니다.
- 논문 표와 본문의 수치는 인용과 재계산을 위해 옮겼습니다. 그래프와 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
