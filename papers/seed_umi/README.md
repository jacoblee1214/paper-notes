# SEED-UMI (arXiv:2609.11753, CoRL 2026 포스터) 분석 노트

*SEED-UMI: Sharing the Exoskeleton between human and robot for onE-to-one Dexterous demonstration* (arXiv v1, 2026-09-10)를 표 재계산, 통계 검정, 1축 접촉 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2609.11753에서 받을 수 있습니다). 논문 그림은 그림 7 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **외골격 제작과 실로봇 평가는 이 맥(Apple M4 Pro, CUDA 없음)에서 불가입니다.** CAD, 부품 목록, 데이터, 체크포인트가 공개되어 있지 않습니다.
- **공식 코드는 없습니다.** GitHub 저장소는 프로젝트 페이지의 소스입니다.
- **매핑 학습 구조는 합성 환경에서 구현할 수 있습니다.** `analysis/toy_paired_replay.py`(numpy, 1초 미만)로 가장 단순한 형태를 확인했습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 7만 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 한계, 표기 관찰 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과 |
| `analysis/04_claim_verification.md` | 코드 확인, 주장 검증, 구현 가능성 |
| `analysis/recompute.py` | 표 1 재계산과 검정 (numpy, scipy, 약 30초) |
| `analysis/toy_paired_replay.py` | 토이: 식 (1) 정규화 항과 접촉 보정 (numpy, 1초 미만) |
| `analysis/plot_figures.py` | 그래프 세 장 생성 |
| `report/crop_figures.py` | 논문 그림 7 자르기 (원문 PDF 필요) |
| `report/figures/` | 그래프 세 장(우리가 그림)과 논문 그림 7(© 저자, CC BY 4.0) |

## 재실행

```bash
pip install numpy scipy matplotlib
python3 analysis/recompute.py
python3 analysis/toy_paired_replay.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv abs 페이지의 라이선스 링크 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림 7은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2609.11753"). 사진·영상 프레임이 있는 그림과 3D 렌더 그림은 넣지 않았습니다.
- GitHub 저장소 `Tengbo-Yu/SEED-UMI`는 프로젝트 페이지 소스이고 이 저장소에는 넣지 않았습니다. 가중치와 데이터는 받지 않았습니다.
- 표 값과 본문 수치는 인용과 재계산을 위해 옮겼습니다. 그래프, 토이, 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
