# 02. 선행연구, 공식 코드, 관련 분석

## 1. 계보

| 흐름 | 대표 연구 | Self Forcing과의 관계 (논문 출처) |
|---|---|---|
| 노출 편향과 RNN 시대 기법 | 스케줄 샘플링 계열, 순환망 학습 | 학습 중 자기 출력을 넣는 발상의 뿌리 (§1, 참고문헌 [40, 65, 103]) |
| 영상 확산의 양방향 어텐션 | Wan2.1, SkyReels-V2, LTX-Video 등 | 비교 기준과 초기화 모델. 미래 프레임을 보므로 스트리밍에 불리 (§1) |
| AR 영상 생성 | Pyramid Flow, NOVA, MAGI-1 | 비교 대상. 벡터 양자화 또는 자기회귀 토큰 계열 (§1, Table 1) |
| TF/DF 기반 AR 확산 | Teacher Forcing 블록 마스크, Diffusion Forcing | 주된 대조군. 프레임 단위 MSE 감독 (§3.1, Fig. 1–2) |
| 분포 정합 증류 | DMD, SiD, 적대적 증류 | 목적함수로 재사용. 논문은 증류가 아니라 노출 편향 해결이 목적이라고 구분 (§3.3) |
| 가장 가까운 선행 | CausVid | DF 출력에 DMD를 씌움. 논문은 학습 분포가 추론 분포와 다르다고 지적 (§2 CausVid 단락) |
| 롤링 캐시 | StreamingLLM 계열 | 논문 v1에서 시도했다가 v2에서 삭제 (v1 §3.4) |

## 2. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 | 최근 push | 별 | 상태 |
|---|---|---|---|---|---|
| [guandeh17/Self-Forcing](https://github.com/guandeh17/Self-Forcing) | 공식 구현 (학습, 추론, 데모) | Apache-2.0 (LICENSE 파일, 2025-09-12 추가) | 2025-09-12 | 3,567 | 얕은 복제 `repo/` (커밋 `33593df`, 정적 읽기만) |
| [Self-Forcing 프로젝트 페이지](https://self-forcing.github.io/) | 웹사이트 | 확인하지 않음 | — | — | 학회 표기 없음 |
| [gdhe17/Self-Forcing (Hugging Face)](https://huggingface.co/gdhe17/Self-Forcing) | 체크포인트 (DMD, SiD, GAN, ODE 초기화, 10초 변형, SiD v2) | 모델 카드 메타데이터 apache-2.0, 최상위 license 필드 없음 | 2025-09-23 | 좋아요 132 | 받지 않음 (가중치 다운로드 금지) |
| [tianweiy/CausVid](https://github.com/tianweiy/CausVid) | 선행 구현, ODE 초기화 방법의 출처 | NOASSERTION (확인 필요) | 2025-08-07 | 1,454 | 받지 않음 |
| [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | 기반 모델 코드 (`wan/` 폴더를 복사해 씀) | Apache-2.0 | 2026-03-05 | 17,468 | 받지 않음 |

- 저장소 README는 RTX 4090에서 실시간이라고 씁니다. 논문의 속도 측정은 모두 H100에서 했고 4090 수치는 없습니다(`04` D11).
- 공식 학습 설정 파일은 DMD와 SiD 두 개뿐입니다. GAN 설정 파일은 저장소에 없습니다(`04` D9).
- 데이터 준비 스크립트(ODE 쌍 생성, LMDB 생성)는 있지만, ODE 초기화 절차는 README가 CausVid 저장소를 가리킵니다.

## 4. 참고 문헌 중 바로 읽을 것

- CausVid (논문 [100]): 가장 가까운 선행. 이 논문의 Table 2 DF+DMD 재현 대상입니다.
- Diffusion Forcing (논문 [8, 10]): 프레임 단위 잡음 문맥.
- DMD (논문 [98, 99]), SiD (논문 [112, 113]): 목적함수 원전.
- R3GAN (논문 [29]): 이 논문의 GAN 손실 원전.
- Wan2.1 (논문 [83]): 기반 모델.
