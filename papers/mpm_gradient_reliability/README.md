# On the Numerical Reliability of Differentiable Physics-Based Optimization (arXiv:2609.34666) 분석 노트

*On the Numerical Reliability of Differentiable Physics-Based Optimization for Robotic Material Manipulation* (Yang, Wei, Lai, Ji; IROS 2026 워크숍 구두 발표)을
표 재계산, 유도, CPU toy로 분석한 자료입니다. 공식 코드는 공개되지 않아(2026-11-11 확인) 코드 대조는 하지 않았습니다.

**구현할 수 있는지:** 공식 재현은 불가능합니다(코드 미공개, 논문 규모는 CUDA GPU 11 GiB 이상 필요). 2D numpy MLS-MPM을 직접 짜는 것은 중간 이상의 작업이고,
toy 수준의 메커니즘 확인은 CPU에서 끝났습니다(`analysis/04_claim_verification.md` §4).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | 논문 구조, MPM 설정, 결과, 읽으면서 본 문제 |
| `analysis/02_prior_work.md` | 선행연구 계보, 가까운 대안, GitHub 표 |
| `analysis/03_analysis.md` | 논문 수치 재계산, FD 오차 모형, 리아푸노프 성장, SVD VJP 조건, toy 결과 |
| `analysis/04_claim_verification.md` | 권리 판정, 주장 검증(C1–C20), 빠진 정보, 구현 가능성 |
| `analysis/table_check.py` | 논문 표와 부록 수치 재계산(Fisher 검정, 비트 예산, Table II 단조성) |
| `analysis/fd_chaos_toy.py` | 혼돈 사상의 FD U 곡선과 항복 클램프의 임계 사건 |
| `analysis/reduction_order_toy.py` | float32 무작위 누적 순서가 기울기 부호를 바꾸는 과정 |
| `analysis/svd_vjp_toy.py` | SVD 역전파의 간격 민감도와 게이지 안전 식 |
| `analysis/extra_lyapunov_noise.py` | 접선의 리아푸노프 기울기와 f32·f64 누적 순서 잡음의 임계 스텝 |
| `analysis/extra_fixedpoint_reduce.py` | 고정소수점 정수 리덕션의 셔플 재현성과 접선 정확성 |
| `analysis/extra_pinn_hessian.py` | 작은 PINN 손실의 이차 도함수에 AD/FD 검사 (torch 필요, CPU 수 초) |
| `analysis/plot_figures.py` | toy 그래프 |
| `report/crop_figures.py` | 논문 그림을 PDF에서 자르는 스크립트 |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/table_check.py
python3 analysis/fd_chaos_toy.py
python3 analysis/reduction_order_toy.py
python3 analysis/svd_vjp_toy.py
python3 analysis/extra_pinn_hessian.py  # torch 필요
python3 analysis/plot_figures.py
```

모두 CPU 단일 스레드에서 1분 이내에 끝납니다. toy의 장, 입자 수, 상수 a, 손실 설정은 가정입니다.

## 출처와 라이선스

- 논문은 **CC BY 4.0**입니다(arXiv abs 페이지 확인). PDF에 다른 권리 표기가 없습니다. `report/figures/paper_fig2_findings.png`는 논문에서 잘라낸 그림이며 © 저자, CC BY 4.0, 출처 arXiv:2609.34666입니다.
- 나머지 그래프는 논문 수치와 toy 시뮬레이션으로 직접 그렸습니다. 원문 PDF는 이 저장소에 넣지 않았습니다.
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
