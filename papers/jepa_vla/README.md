# JEPA-VLA (arXiv:2602.11832) 분석 노트

*JEPA-VLA: Video Predictive Embedding is Needed for VLA Models* (Miao et al., Tsinghua University, Huawei Noah's Ark Lab, 2026)을 논문 표와 본문과 대조해 분석한 자료입니다. 분석 대상은 arXiv v1 하나입니다. 학회 표기는 확인되지 않았습니다(arXiv 댓글과 저널 참조 없음). 공식 코드는 찾지 못했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 제외) |
| `analysis/01_paper_analysis.md` | 논문 분석: Abstract → Method → Experiments → 한계 |
| `analysis/02_prior_work.md` | 선행연구 계보와 가까운 대안 |
| `analysis/03_analysis.md` | 수치 재계산, WorldVLA 집계 검사, 장난감 실험, 통계 |
| `analysis/03_recompute.py` | 표 값의 재계산과 검정 (numpy, scipy, 1초 미만) |
| `analysis/03_toy_encoder_probe.py` | 상수·변동 교란에서 인코더가 담는 정보 (numpy, 수 초) |
| `analysis/04_claim_verification.md` | 주장 C1–C32 검증 (코드가 없어 주장 검증으로 대체) |
| `analysis/plot_results.py` | 그래프 3장 생성 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/03_recompute.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 analysis/03_toy_encoder_probe.py
python3 analysis/plot_results.py
```

논문의 표 값은 `analysis/03_recompute.py` 안에 옮겨 적은 값입니다. 원문은 `curl -L -o paper.pdf https://arxiv.org/pdf/2602.11832v1`로 받아 직접 대조할 수 있습니다.

## 출처와 라이선스

- 논문의 라이선스는 arXiv 비독점 배포이며 CC가 아닙니다. 이 저장소에는 **논문 그림과 원문 PDF를 넣지 않았습니다.**
- `report/figures/`의 그래프는 논문 표의 수치와 이 저장소의 장난감 실험으로 직접 그렸습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
