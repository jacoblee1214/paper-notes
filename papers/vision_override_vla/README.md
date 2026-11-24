# When Vision Overrides Language (arXiv:2602.17659, CoRL 2026 스포트라이트) 분석 노트

*When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs* (arXiv v3, 2026-11-04)를 표 재계산, 통계 검정, 공식 코드 정적 대조, 1차원 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2602.17659에서 받을 수 있습니다). 논문 그림은 두 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **논문 전체 재현은 이 맥(Apple M4 Pro, 24 GiB, CUDA 없음)에서 불가입니다.** 공식 평가는 CUDA 전제 정책 서버와 LIBERO 렌더링이 필요하고, VA 체크포인트 약 40 GB와 실세계 데이터는 받지 않았거나 공개되지 않았습니다.
- **표 재계산, 코드 정적 대조, 구조 토이는 가능하고 끝냈습니다.** 소형 PyTorch 재구현은 가능합니다(노력 3–5일 추정, 측정 전).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 두 장만 포함) |
| `analysis/01_paper_analysis.md` | 판본 차이, 벤치마크, 방법, 결과 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 검정, 토이 결과, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (numpy, scipy, 수 초) |
| `analysis/toy_cag_mixture.py` | 토이: 행동 외삽 대 속도장 CFG 대 분포 기울이기 (numpy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 두 장(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_cag_mixture.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 두 장(그림 3, 4)은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2602.17659"). 실로봇 사진과 시뮬레이션 화면이 있는 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/yuffish/VLA-CF`(MIT)에 있습니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 가중치와 데이터셋은 받지 않았습니다.
