# ForceVLA (arXiv:2505.22159, NeurIPS 2025 포스터) 분석 노트

*ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation*(arXiv v3, 2025-09-18)을 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 실험은 재현하지 않았습니다.

## 구현 가능성 (요약)

- **공식 설정 학습과 실로봇 재현은 이 맥에서 불가입니다.** CUDA, 22.5 GB 이상 GPU, 로봇이 필요합니다.
- **공식 체크포인트는 찾지 못했습니다.** 제3자 가중치는 쓰지 않았습니다.
- **구조 측정은 가능합니다.** FVLMoE 블록을 PyTorch로 옮겨 라우팅과 용량 초과를 재는 작업은 CPU·MPS에서 할 수 있습니다(측정 전 추정 2–3일).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 없음) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표, 관련 분석 연결 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 주장 판정 |
| `analysis/04_code_verification.md` | 공식 코드 정적 대조, 차이 목록, 구현 가능성 |
| `analysis/recompute.py` | 표 2·3·5 재계산과 검정 (scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 두 장 생성 |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/plot_figures.py
```

## 출처와 라이선스

- 논문 라이선스: arXiv abs 페이지의 "view license"는 arXiv 비배타 배포 라이선스 1.0을 가리킵니다(CC BY 아님). 논문 그림은 넣지 않았습니다.
- 그래프 두 장은 이 저장소 작성자가 논문 수치에서 만들었습니다. 그림 5의 막대 값은 원문 그림의 라벨을 옮긴 것입니다.
- 공식 코드(`github.com/ft-robotic/ForceVLA`, Apache-2.0)는 이 저장소에 넣지 않았고, 설치하거나 실행하지 않았습니다. 정적으로만 읽었습니다.
- 데이터셋과 가중치는 받지 않았습니다(과제 메타 파일만 확인).
