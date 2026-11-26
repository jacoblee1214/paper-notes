# TopoRetarget (arXiv:2606.16272) 분석 노트

*TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation* (arXiv v2, 2026-06-22, CoRL 2026 포스터)의 표 수치를 다시 계산하고, 공개 풀이기 코드(`wuji-technology/toporetarget`, MIT, 커밋 `039c540`)를 논문과 정적으로 대조한 자료입니다.

## 구현 가능성 (요약)

- **리타게팅 풀이기는 일반 노트북의 CPU로 구현하고 돌릴 수 있습니다** (공식 코드는 CPU 전용, Linux x86_64 지원 표기).
- **논문 표 2의 강화학습 추적 정책 학습은 NVIDIA GPU(CUDA 12.8)가 필요하고, 실로봇은 Wuji Hand가 필요합니다.**
- **표 1은 ContactPose 데이터와 기준선 설치가 있으면 CPU로 가능**하나 기준선·평가 코드는 공개되지 않았습니다.
- 이 노트는 공식 코드를 실행하거나 설치하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 결과, 한계, 버전과 라이선스 |
| `analysis/02_prior_work.md` | 선행 계보, 공식·관련 저장소 표 |
| `analysis/03_analysis.md` | 표 재계산, 검정, 상호작용 메시 장난감 (V1–V15) |
| `analysis/04_code_verification.md` | 논문과 공개 코드 대조 (D1–D18), 주장 대조 (C1–C9), 구현 가능성 |
| `analysis/recompute.py` | 표 읽기, 재계산, 검정 (numpy, scipy, 수 초) |
| `analysis/toy_interaction_mesh.py` | 그래프 통계와 축소 리타게팅 장난감 (약 1분, 공식 저장소의 예제 클립 데이터 파일을 읽음) |
| `analysis/plot_figures.py` | 그래프 다섯 장 생성 |
| `report/figures/` | 우리가 그린 그래프 다섯 장 |

## 재실행

```bash
pip install numpy scipy matplotlib
# 원문 PDF(arXiv:2606.16272 v2)를 pdftotext -layout으로 옮긴 텍스트를 paper/paper.txt로 저장
python3 analysis/recompute.py
# 공식 저장소를 받아 repo/ 로 두기 (장난감 실험용)
git clone https://github.com/wuji-technology/toporetarget repo && git -C repo checkout 039c54022016a882ef8d3a6d34b4578aee7cb51f
python3 analysis/toy_interaction_mesh.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문은 arXiv 비독점 배포 라이선스(CC BY 아님)로 공개되어 있습니다 (arXiv 라이선스 링크 확인). 원문 PDF에서 저작권 표시를 찾지 못했습니다.
- 이 노트는 논문 그림을 쓰지 않았습니다. 표 수치는 인용과 재계산을 위해 옮겼고, 그래프, 스크립트, 분석 문서는 이 노트 작성자가 만들었습니다.
- 공식 코드(MIT)와 강화학습 저장소 `wuji-technology/wuji-mjlab`(Apache-2.0)의 설정 파일은 읽기만 했고 이 노트에 넣지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.
