# MiTaS (arXiv:2606.06281, CoRL 2026 포스터) 분석 노트

*Multi-Resolution Tactile Imitation Learning for Contact-Rich Robotic Manipulation* (arXiv v2, 2026-10-21)의 표와 본문 수치를 재계산하고 주장을 검증한 자료입니다. 공식 코드가 공개되어 있지 않아 코드 대조는 하지 않았습니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2606.06281에서 받을 수 있습니다).

## 구현 가능성 (요약)

- **실로봇 성공률 재현은 공개 자료만으로 할 수 없습니다.** 코드와 체크포인트가 없고 Franka, GelSight Mini, Evetac와 과제 장치가 필요합니다.
- **구조 재구현(24.5 M)과 파라미터·지연 측정은 일반 노트북급에서 가능해 보입니다.** 공개 데이터(Hugging Face)로 한 오프라인 학습은 다운로드가 필요하고 성공률의 대리 지표로는 약합니다. 이번 노트에서는 실행하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, v1과 v2의 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 코드와 저장소 표 |
| `analysis/03_analysis.md` | 재계산 V1–V8, 내부 일관성과 작은 계산 K1–K4 |
| `analysis/04_claim_verification.md` | 주장 판정 C1–C17, 저자 확인 항목, 구현 가능성 |
| `analysis/recompute.py`, `recompute_results.json` | 표 재계산과 검정, 검출력과 변동 시뮬레이션 (numpy, scipy, 약 10초) |
| `analysis/digitize_attention.py`, `digitize_attention_results.json` | 부록 그림 9의 어텐션 곡선 판독 (원문 PDF 필요, pymupdf, Pillow) |
| `analysis/plot_figures.py` | 그래프 네 장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib pymupdf pillow
python3 analysis/recompute.py
curl -L -o paper/paper.pdf https://arxiv.org/pdf/2606.06281v2
python3 analysis/digitize_attention.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에서 CC BY 4.0(creativecommons.org/licenses/by/4.0)으로 공개되어 있습니다. 이 저장소의 논문 그림 두 장(`paper_fig2_arch.png`, `paper_fig4.png`)은 © 저자, CC BY 4.0, 출처 arXiv:2606.06281 (v2)이고, 나머지 그래프는 작성자가 그린 것입니다.
- 프로젝트 페이지는 mitas-touch.github.io 입니다. 데이터셋은 Hugging Face `RKrohn/MiTaS`(카드 CC BY 4.0)이며 이 노트는 파일 목록과 크기 정보만 읽었고 데이터는 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다.
