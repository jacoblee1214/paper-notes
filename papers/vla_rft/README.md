# VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators (arXiv:2510.00406) 분석 노트

*VLA-RFT* (arXiv v1, 2025-10-01)을 표 재계산, 공식 코드 정적 대조, 작은 numpy 토이로 분석한 자료입니다. 학회 표기는 확인되지 않았습니다. 논문 원문과 논문 그림은 넣지 않았습니다(논문 라이선스 arXiv non-exclusive). 이 폴더의 그래프 세 장은 우리가 만든 것입니다.

## 구현 가능성 (요약)

- **논문 수치의 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가능합니다.** 공식 설치는 CUDA 전제(flash-attn, vLLM, FSDP)이고, 세계모델과 정책의 사전학습 스크립트와 가중치가 공개되지 않았습니다.
- **축소판(LIBERO 한 스위트, 소형 세계모델)은 조건부로 가능합니다.** 데이터 약 2.5 GB 다운로드와 MPS 학습 시간을 확인한 뒤 진행합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 주장과 증거의 틈, 버전 정보 |
| `analysis/02_prior_work.md` | 선행 계보, 구성 요소별 출처, GitHub와 자료 표 |
| `analysis/03_analysis.md` | 표 재계산과 일관성 검사, 스위트별 통계, 섭동 민감도, 토이 |
| `analysis/04_code_verification.md` | 논문과 공식 코드의 대조, 불일치와 빈칸, 구현 가능성, 저자 질문 |
| `analysis/recompute.py` | 표 재계산과 일관성 검사 (numpy, 1초 미만) |
| `analysis/toy_reward_validity.py` | 토이: 궤적 유사도 보상이 우회 성공을 어떻게 채점하는지 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 (matplotlib) |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_reward_validity.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 non-exclusive distribution 라이선스로 공개되어 있습니다(abs 페이지 확인, 2026-11-20). 원문과 논문 그림은 이 저장소에 넣지 않았습니다.
- 공식 코드(https://github.com/OpenHelix-Team/VLA-RFT)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 복사하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋은 받지 않았습니다.
