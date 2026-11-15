# What Are We Actually Benchmarking in Robot Manipulation? (arXiv:2606.04233, CoRL 2026) 분석 노트

*What Are We Actually Benchmarking in Robot Manipulation?* (arXiv v1, 2026-06-02)을 공개 집계 파일로 재계산하고, 저장소를 정적으로 읽고, 작은 numpy 토이 두 개로 검정 설계를 확인한 자료입니다. 논문 원문과 논문 그림은 넣지 않았습니다. 이 폴더에는 우리가 만든 그래프만 있습니다.

## 구현 가능성 (요약)

- **통계와 집계 재현은 Apple 실리콘 맥의 CPU에서 구현할 수 있고, 완료했습니다.** 공개 CSV와 numpy, scipy만 씁니다.
- **LIBERO 무언어 프로브 학습과 평가는 조건부로 가능합니다.** 저자 명령은 CUDA만 씁니다. LIBERO 설치와 데이터 다운로드가 먼저 필요하고, MPS 학습 시간은 아직 재지 않았습니다.
- **CALVIN과 SimplerEnv 외부 정책 재평가는 이 맥에서 현실적이지 않습니다.** 대용량 데이터, 외부 VLA의 GPU 추론, 시뮬레이터 렌더링이 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용) |
| `analysis/01_paper_analysis.md` | 논문 구조, 네 진단, 실험 설정, 주장과 증거의 간격 |
| `analysis/02_prior_work.md` | 선행 계보, 가까운 대안, GitHub와 코드 자료 표 |
| `analysis/03_analysis.md` | 표 재계산, 그림 3 재집계와 일관성 검사, 순위, 토이 1 |
| `analysis/04_code_verification.md` | 공식 저장소와 논문의 대조, 빈칸, 구현 가능성 판정 |
| `analysis/recompute.py` | 표 2–9, 그림 3, LIBERO 쌍 불일치의 재계산 (numpy, scipy, 수 초) |
| `analysis/toy_rank_flip.py` | 토이 1: 시도 수, 짝지음, 과제 부분집합에 따른 순위 뒤집힘 (numpy, 1초) |
| `analysis/toy_envelope.py` | 토이 2: 집계 판정 경계(부록 C)를 완전 열거로 검산 (numpy, 1초) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |

## 재실행

공식 저장소를 `repo/`에 얕게 clone한 뒤 실행합니다(재계산은 그 저장소의 CSV를 읽습니다. 공식 스크립트는 실행하지 않습니다).

```bash
git clone --depth 1 https://github.com/ripl/ManipulationBenchmarkAudit repo
pip install numpy scipy matplotlib
python3 -I analysis/recompute.py
python3 -I analysis/toy_rank_flip.py
python3 -I analysis/toy_envelope.py
python3 -I analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스입니다(abs 페이지 확인, 2026-11-15). 논문 그림은 이 폴더에 넣지 않았습니다.
- 공식 저장소(https://github.com/ripl/ManipulationBenchmarkAudit)는 MIT 라이선스입니다. 이 분석에서는 실행하거나 설치하지 않고 정적으로만 읽었습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
- 모델 가중치와 데이터셋(Google Drive)은 받지 않았습니다.
