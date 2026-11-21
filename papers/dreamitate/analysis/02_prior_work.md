# 02 선행연구와 관련 자료

## 1. 계보

- **시각 정책 베이스라인**: Diffusion Policy [2] (행동 복제, 이 논문이 직접 비교하는 유일한 기준선입니다).
- **영상 예측을 정책 표현으로 쓰는 계열**: 논문 §2는 영상 예측을 사전학습 목적으로 쓴 연구 [14–18]을 들고, 이 논문은 영상 예측 모델을 정책 자체의 일부로 씁니다.
- **영상 모델을 세계 모델·시뮬레이터로 쓰는 계열** [34, 5, 28]: 행동을 조건으로 미래 영상을 예측하고 계획에 사용. 이 논문은 "생성 영상을 바로 행동으로 바꾼다"는 점에서 다릅니다.
- **영상-언어 모델로 장기 계획** [35–37]: 이 논문과 직접 경쟁하지 않는 대안입니다.

## 3. 코드·자료 (GitHub, 프로젝트 페이지)

| 항목 | 내용 | 확인 방법 |
|---|---|---|
| 공식 코드 | github.com/cvlab-columbia/dreamitate, 별 60, 포크 0, 기본 브랜치 master | `gh api repos/cvlab-columbia/dreamitate` (2025-11-21) |
| 마지막 push | 2025-06-07 (최신 커밋 `c162da1f71a727939fbff26c0658af203b179d9d`) | 같은 API |
| 라이선스 | GitHub 자동 판정 NOASSERTION. `LICENSE.txt`는 **CC BY-NC 4.0** 전문. `video_model/LICENSE-CODE`는 MIT (Stability AI 코드). `megapose/LICENSE`는 Apache 2.0 (NVIDIA/Inria) | 파일 직접 확인 |
| 프로젝트 페이지 | dreamitate.cs.columbia.edu (논문, arXiv, 영상, 코드 링크) | 페이지 HTML |
| 프로젝트 페이지의 논문 | `assets/dreamitate_arxiv_v2.pdf` (49.7 MB, 17쪽, 본문 Table 2 수치 동일) | 다운로드 확인 |
| 체크포인트·데이터 | `checkpoints.zip`, `finetuned_models.zip`, `dataset.zip`, `megapose-models.zip` (프로젝트 페이지에서 받음, 이번 작업에서는 받지 않음) | README |

코드 클론은 `repo/`에 두고 정적으로만 읽었습니다 (`04_code_verification.md`).

## 4. 연구 맥락 메모

- 이 논문은 CoRL 2024 게재 이후 arXiv에 v1만 있고, 프로젝트 페이지에는 더 긴 v2 PDF가 있습니다. 두 버전의 Table 2 수치는 같습니다.
- 생성 영상을 행동으로 바꾸는 단계가 MegaPose 추적에 의존합니다. 추적 오차가 곧 행동 오차가 되는 구조라서, 이 연결 고리가 실험에서 얼마나 큰 실패 원인인지는 논문에 분해 결과가 없습니다.
