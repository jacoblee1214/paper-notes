# 02. 선행연구와 관련 저장소

## 0. 학회와 공식 코드

- **학회**: ICLR 2023. 이 논문은 `Published as a conference paper at ICLR 2023` 머리말이 있는 PDF(OpenReview 게시본)로 확인했고, iclr.cc 공식 가상 발표 페이지(`iclr.cc/virtual/2023/poster/11309`)에 등재되어 있습니다(top 25% 표기). 발표 형식은 포스터입니다. 이 페이지가 공식 목록이므로 README와 블로그 글의 학회 표기는 "ICLR 2023"입니다.
- **공식 코드**: **찾지 못했습니다.** arXiv 초록 페이지에 코드 링크가 없고, 논문 본문에도 저자 저장소 링크가 없습니다(본문에 인용된 것은 torchdiffeq, guided-diffusion, TF-GAN뿐입니다). 따라서 코드 대조는 하지 않고, 논문 주장 검증(`04`)으로 대신했습니다.
- **프로젝트 페이지**: 찾지 못했습니다. 논문에도 없습니다.

## 1. 선행연구 계보

| 계열 | 대표 연구 | 이 논문과의 관계 |
|---|---|---|
| CNF (연속 정규화 흐름) | Chen et al. 2018 (Neural ODE) | 모델 종류. 학습에 ODE 시뮬레이션이 필요함 |
| 최대우도 CNF | FFJORD (Grathwohl et al. 2018) | 대규모 이미지에 비실용적이라고 논문이 지적 (§5) |
| 시뮬레이션 없는 CNF 학습 | Rozen et al. 2021 (Moser flow): 적분 필요. Ben-Hamu et al. 2022 (manifold): 미니배치에서 편향된 그래디언트 | 이 논문은 조건부 목표로 편향 없는 그래디언트를 얻음 |
| 확산 모델 | DDPM (Ho 2020), Score SDE (Song 2020b), ScoreFlow (Song 2021), iDDPM (Nichol & Dhariwal 2021), VDM (Kingma 2021) | 확산 경로를 흐름 매칭의 특수 경우로 포함. 학습 목표 비교 대상 |
| 빠른 샘플러 | DDIM (Song 2020a), DPM-Solver류 (Zhang & Chen 2022) | 확산 경로의 ODE 관점을 씀. 이 논문은 경로 자체를 바꿈 |
| 동시기 연구 | Liu et al. 2022 (rectified flow, arXiv:2209.03003), Albergo & Vanden-Eijnden 2022 (stochastic interpolants, arXiv:2209.15571) | 같은 조건부 목표를 거의 동시에 제안함 |
| 최적수송 | McCann 1997 (변위 보간), Villani 2009 | OT 조건부 경로의 근거 |

**이 논문의 자리**: 확산 모델의 학습 목표(스코어)를 버리고, 확률 경로를 직접 지정하는 프레임워크입니다. 그 경로로 OT 직선 경로를 제안합니다. 확산 경로도 같은 틀에 들어가므로 "확산을 대체하는 흐름 기반 생성"의 기준점이 되었습니다.

## 2. 가까운 대안 (구현 선택지)

| 방법 | 학습 목표 | 경로 | 비고 |
|---|---|---|---|
| FM-OT (이 논문) | 조건부 속도 회귀 식 (23) | 직선 OT (식 20–22) | 샘플링 NFE가 가장 적음 |
| FM-Dif (이 논문) | 조건부 속도 회귀 | VP 확산 (식 18–19) | 확산 경로 대조군 |
| SM-Dif, DDPM | 스코어 또는 노이즈 예측 | VP 확산 | 표 1 기준선 |
| Rectified flow | 동시기 연구 (식 23과 같은 형태의 목표) | 직선 | 이 논문과 비교 대상. 본 분석에서 수치 비교는 하지 않음 |

## 4. GitHub와 관련 저장소

API 조회 시점은 2026-11-22입니다. 복제(clone)는 하지 않았습니다(공식 코드가 없고, 관련 저장소는 정적 읽기 대상도 아님).

| 저장소 | 역할 | 라이선스 | ★ | 마지막 push | 메모 |
|---|---|---|---|---|---|
| [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) | 2024년 흐름 매칭 라이브러리 (Lipman et al., arXiv:2412.06264 인용). 이 논문의 공식 코드 **아님** | CC BY-NC 4.0 (LICENSE 파일 확인, GitHub 자동 인식 NOASSERTION) | 4,848 | 2026-01-05 | 비상업 라이선스이므로 코드 복사 시 주의 |
| [atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching) (torchcfm) | 저자와 무관한 제3자 구현 라이브러리. 조건부 흐름 매칭 목표를 PyTorch로 제공 | MIT | 2,613 | 2026-07-20 | 이 논문의 OT 목표와 같은 계열 구현. 활발히 관리됨 |
| [aleksandrinvictor/flow-matching](https://github.com/aleksandrinvictor/flow-matching) | 이 논문의 PyTorch 재구현 (제3자) | 없음 (GitHub 미표기) | 24 | 2024-08-10 | 소규모 재구현. 라이선스가 없어 코드 복사 불가 |

- **재현의 출발점으로는 torchcfm이 적합합니다.** MIT이고, 논문의 CFM 목표(식 9, 23)를 구현합니다. 다만 이 논문과 같은 학습 설정은 제공하지 않습니다.
- 검색 결과에는 "Alexis5131/Conditional-Flow-Matching"(★0)도 있었지만, 설명이 논문 링크뿐이고 내용을 확인하지 않았습니다.
