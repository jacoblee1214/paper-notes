# FlashSAC (arXiv:2604.04539) 분석 노트

*FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control* (RSS 2026, arXiv 댓글 기준)을
논문 본문과 부록, 공식 저장소의 설정과 학습 곡선(CSV), 간단한 CPU 토이로 분석한 자료입니다.

**구현할 수 있는지:** 공식 GPU 재현은 CUDA GPU가 필요해서 이 맥에서는 불가능합니다. 공식 CPU 재현은 패키지 설치 승인 후 과제 일부만
조건부로 가능합니다. 알고리즘 재구현과 토이는 CPU에서 가능합니다. 실물 G1 학습과 시각 설정은 공개 코드가 없어 재현할 수 없습니다.
자세한 판정은 `analysis/04_code_verification.md` §4에 있습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (논문 그림은 저자 CC BY 4.0 표기와 함께) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험, 한계, v1→v2 차이 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, 공개 구현 표 |
| `analysis/03_analysis.md` | 공개 학습 곡선 재계산, 유의성, 소거 실험, 유도 검산, 토이 |
| `analysis/04_code_verification.md` | 논문 ↔ 코드 대조(정적 읽기), 재현 가능성 판정 |
| `analysis/recompute_results.py` | 공개 CSV 재계산 (입력: 공식 저장소 `results/`) |
| `analysis/derivation_checks.py` | 목표 엔트로피, Zeta 잡음, 재생 비율, 파라미터 수 계산 |
| `analysis/sac_stabilizer_toy.py` | LQR에서 SAC 안정화 요소의 효과 (CPU, 4 설정 × 3 시드) |
| `analysis/plot_figures.py` | 직접 만든 그래프 |
| `report/crop_figures.py` | 논문 그림을 PDF에서 자르는 스크립트 |

## 재실행

공식 저장소의 학습 곡선 CSV가 필요합니다(이 저장소에는 넣지 않았습니다).

```bash
git clone --depth 1 https://github.com/Holiday-Robot/FlashSAC repo
pip install numpy pandas scipy matplotlib pillow torch
python3 analysis/recompute_results.py
python3 analysis/derivation_checks.py
python3 analysis/sac_stabilizer_toy.py --all      # CPU, 시드 3 × 설정 4 (수 분 ~ 20분)
python3 analysis/plot_figures.py
```

`analysis/sac_stabilizer_toy.py`는 공식 코드를 실행하지 않습니다. 토이는 이 저장소에서 새로 만든 LQR 과제입니다.

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). PDF에 다른 권리 표기가 없습니다. `report/figures/paper_*.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2604.04539입니다. Fig. 1(시뮬레이터 장면 합성)과 Fig. 6(실로봇 사진)은 제3자 장면이 섞여 있어 싣지 않았습니다.
- 공식 코드는 MIT 라이선스입니다(저장소 LICENSE 파일 확인, 커밋 87edc90). 이 저장소에는 코드를 복사해 넣지 않았습니다.
- `report/figures/fig_*.png`와 토이, 재계산은 이 저장소 작성자가 만들었습니다.
- 원문 PDF는 이 저장소에 넣지 않았습니다.
