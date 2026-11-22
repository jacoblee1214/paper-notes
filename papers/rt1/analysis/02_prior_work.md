# RT-1 선행연구와 관련 자료 (02)

## 1. 계보: 구성 요소별 출처

RT-1은 새 알고리즘이라기보다 **기존 부품을 실로봇 규모로 조합하고 추론 속도 제약을 맞춘 것**입니다. 구성 요소별 출처는 다음과 같습니다(논문 참고문헌과 §3·§5.1 인용).

| 구성 요소 | 출처 | RT-1에서의 역할 |
|---|---|---|
| 디코더 전용 Transformer, 행동을 토큰 시퀀스로 | Vaswani 2017; Chen 2021 (Decision Transformer); Lee 2022a (Multi-Game DT); Reed 2022 (Gato) | 시퀀스 모델링으로 정책 학습. 인과 마스크 사용 (§2, §5.1) |
| 행동 이산화 (빈) | Reed 2022 (Gato), Lee 2022a | 11개 행동 차원을 256 빈으로 (§5.1) |
| 언어 조건부 FiLM | Perez 2018 | EfficientNet 내부에 FiLM 층 삽입, 0 초기화 (§5.1) |
| ImageNet 사전학습 EfficientNet-B3 | Tan & Le 2019 | 영상 토큰화기의 백본 (§5.1) |
| TokenLearner | Ryoo 2021 | 81 → 8 토큰 압축으로 추론 가속 (§5.1) |
| 문장 임베딩 | Cer 2018 (Universal Sentence Encoder) | 지시 임베딩 512차원 (§5.1) |
| 실로봇 다과제 학습 | Jang 2021 (BC-Z) | 기준선 (§6) |
| 가장 가까운 대규모 실로봇 기준선 | Reed 2022 (Gato) | 기준선, 1.2B에서 37M으로 축소 (§6) |
| 시뮬레이션 실제화 | Ho 2020 (RetinaGAN) | 모델 선택용 실제-시뮬 평가 (부록 C.3) |
| 다른 로봇 데이터 | Kalashnikov 2018 (QT-Opt) | 209k 에피소드 Kuka 데이터 혼합 (§6.3) |
| 장기 과제 계획 | Ahn 2022 (SayCan), Kalashnikov 2021a (MT-Opt, 가치 함수) | SayCan 프레임워크로 장기 평가 (§6.4) |

## 2. 가까운 대안과 차이

- **BC-Z (Jang 2021):** ResNet, 피드포워드, 연속 행동, 이력 없음. RT-1은 Transformer, 이산 행동, 6장 이력을 씁니다. 어블레이션(표 13)에서 "연속 행동"과 "이력 제거"가 각각 성능을 떨어뜨립니다.
- **Gato (Reed 2022):** 패치 기반 토큰, 늦은 언어 결합, 1.2B 파라미터, 추론 1.9초. RT-1은 초기 FiLM 결합과 TokenLearner로 35M에서 15 ms를 씁니다(표 13).
- **Behavior Transformer (Shafiullah 2022):** 논문 §2에서 언급되며, 실로봇 과제가 제한적이라고 평가합니다.
- **SayCan (Ahn 2022):** 언어 모델이 계획을 세우고 affordance 가치 함수가 가능성을 고릅니다. RT-1은 이 프레임워크의 저수준 정책으로 쓰였습니다.

## 4. GitHub와 자료 표

| 구분 | 주소 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| 공식 코드 (RT-1) | https://github.com/google-research/robotics_transformer | Apache-2.0 | 1,739 | 2024-01-31 (API `pushed_at`), 아카이브 표시 | **클론함 (`repo/`, 정적 읽기만, 커밋 4569641, 2023-02-10)**. 모델·토큰화·설정만 있고 학습 루프와 데이터 파이프라인은 없음 |
| 공식 데이터 | gs://gresearch/rt-1-data-release (프로젝트 페이지 링크, Google Cloud Storage) | 확인 안 함 | - | - | 다운로드하지 않음. 크기 미확인 |
| 프로젝트 페이지 | https://robotics-transformer1.github.io/ | - | - | - | 영상, BibTeX (학회 표기 없음, arXiv preprint) |
| SayCan 페이지 | https://say-can.github.io/ | - | - | - | 장기 과제 프레임워크 (논문 인용) |
| 비공식 구현 | 확인하지 않음 | - | - | - | 검색 결과 목록을 만들지 않았음 |

클론에는 체크포인트 세 개(`rt1main`, `rt1multirobot`, `rt1simreal`, 약 175 MB)와 EfficientNet-B3 가중치 h5 파일이 함께 들어 있었습니다. 이 분석은 가중치를 받을 필요가 없으므로 클론 후 `repo/trained_checkpoints/`와 `repo/film_efficientnet/efficientnet_checkpoints/`를 지웠습니다. 현재 `repo/`는 47 MB이며 코드와 테스트만 남았습니다. 코드는 읽기만 했고 실행하거나 import하지 않았습니다.

## 5. 라이선스와 출처 표시

- 논문: arXiv 라이선스 nonexclusive-distrib 1.0. 공개 산출물에는 논문 그림을 넣지 않습니다(`01` §0, `report/`).
- 코드: Apache-2.0. 코드를 인용할 때는 저장소 주소와 커밋 해시를 같이 씁니다.
- 데이터: 라이선스를 확인하지 않았습니다. 데이터는 받지 않았습니다.
