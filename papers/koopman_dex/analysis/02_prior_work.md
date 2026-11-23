# 02. 선행연구와 구성 요소

## 1. 계보

| 줄기 | 대표 | 이 논문과의 관계 |
|---|---|---|
| 동적 시스템 기반 기술 표현 | DMP(Ijspeert 2013, Saveriano 2023), Khansari-Zadeh & Billard 2011, Neural Dynamic Policies(Bahl 2020), 지오메트릭 패브릭(Van Wyk 2024, Xie 2022), 스펙트럼 클리핑 선형 동역학(Guo, Han, Ravichandar 2024), Elastic Motion Policy(2025) | 기술을 고정점이 있는 동역학계의 해로 봅니다(§1). 이 논문은 로봇만이 아니라 **환경 특징까지 같은 동역학계에 넣는다**는 점을 차이로 듭니다 |
| Koopman 연산자 제어와 학습 | Koopman 1931, Korda & Mezić 2018(상태 포함 리프팅 + 선형 예측기), Abraham 2017, Bruder 2021, Shi 외 2026(로봇 학습 서베이), Mauroy 외 2020 | **상태 포함 리프팅**(z = [ξ; ψ])은 Korda & Mezić 2018에서 가져옵니다(§1, 참고문헌 31). 다단계 손실로 인코더와 K를 함께 학습하는 구성은 딥 Koopman 계열의 일반적 구성입니다 |
| Koopman 기반 조작 | **KODex**(Han, Xie, Zhao, Ravichandar, CoRL 2023): 다항 리프팅 + 최소제곱으로 다지 손 기술 학습. **KOROL**(Chen 외 2024): 시각 객체 특징을 Koopman 롤아웃으로 학습 | 같은 연구실의 직계 선행. 두 방법이 이 논문의 기준선이며 최소제곱의 한계를 "end-to-end 경사하강"으로 넘는다고 주장합니다(§4.2) |
| 시각 표현 | CoTracker(Karaev 2024), SAM3, DynaMo(Cui 외 2024, 도메인 내 동역학 사전학습), V-JEPA 2, 잠재 행동 월드 모델 | 입력 특징 두 종류(점 흐름, DynaMo). 동역학 인식 잠재 학습의 정당화로 V-JEPA 2와 Garrido 외를 인용(§2) |
| 반응형 정책과 청킹 | Diffusion Policy(Chi 2025), ACT(Zhao 2023), 3D Diffusion Policy, Bidirectional Decoding(Liu 2024), ImMimic, SAIL | 비판의 대상. 청킹과 시간 앙상블을 "평균하는 휴리스틱"으로 규정(§1) |
| 월드 모델, 비디오 생성 | PlaNet, Dreamer, DayDreamer, UniPi, SuSIE, UniSim, RoboDreamer, GR-1/GR-2, Gen2Act, **UVA**(Li 외 2025) | UBM은 "임의 행동의 영향"이 아니라 "시연 행동의 결합 동역학"을 모델링한다고 구분(부록 A). UVA는 가장 강한 기준선으로 실험에 들어갑니다 |

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문 내 위치 |
|---|---|---|
| 상태 포함 Koopman 리프팅 | Korda & Mezić 2018 | §1, §3.2 |
| 다단계 선형 예측 손실 | 딥 Koopman 학습 일반 | §3.2 |
| 단위 행렬 초기화, K 별도 학습률, 기울기 클리핑 | 이 논문의 경험적 요령 | §3.2, 부록 J |
| 점 추적 | CoTracker, SAM3(마스크) | §3.1, 부록 B |
| 자기지도 시각 특징 | DynaMo | §3.1, 부록 B |
| 모방 기준선 | Diffusion Policy, ACT, UVA, KODex, KOROL | §4.1, 부록 D |
| 시뮬 벤치마크 | DexArt(Bao 2023), Adroit(Rajeswaran 2017), SAPIEN, MuJoCo | §4.1, 부록 C |
| 손과 팔 | LEAP Hand(Shaw 2023), Franka Research 3 | §5 |
| 교시 | 키네스테틱 교시(Akgun 2012) | §5 |

## 4. GitHub와 프로젝트 표

| 대상 | 관계 | 라이선스 | 별/최근 push | 기준 |
|---|---|---|---|---|
| [GT-STAR-Lab/K-UBM](https://github.com/GT-STAR-Lab/K-UBM) | **공식 코드** (프로젝트 페이지가 링크) | **라이선스 파일 없음**(API license: null, 저장소 최상위에 LICENSE 없음) | 6 stars, 3 forks / 2026-09-23 | 커밋 `5066c29dee5db1df9f1d496beb6a8236e3efb148`, 얕은 복제 `repo/` (정적 읽기만) |
| [k-ubm.github.io](https://k-ubm.github.io/) | 프로젝트 페이지 | 푸터에 CC BY-SA 4.0 링크(페이지 템플릿 Academic-project-page-template 표기로 보임) | 해당 없음 | HTML을 `paper/project_page.html`로 저장 |
| [facebookresearch/co-tracker](https://github.com/facebookresearch/co-tracker) | 점 추적 | 저장소 API는 NOASSERTION, 번들 사본의 LICENSE.md는 **CC BY-NC 4.0**(비상업) | 5.2k stars / 2026-03-03 | 번들 `third_party/co-tracker`(로컬 패치 포함) |
| [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | 객체 분할 | SAM License (NOASSERTION) | 12.4k stars / 2026-11-17 | 번들 `third_party/sam3`(가중치는 별도 다운로드) |
| [jeffacce/dynamo_ssl](https://github.com/jeffacce/dynamo_ssl) | DynaMo 공식 구현으로 보이는 저장소 | MIT | 122 stars / 2024-10-27 | 논문이 쓰는 특징의 출처. 이 저장소가 논문 참고문헌 30의 공식 코드인지는 확인하지 않음 |
| [ShuangLI59/unified_video_action](https://github.com/ShuangLI59/unified_video_action) | UVA 공식 코드(기준선) | MIT | 419 stars / 2026-08-21 | 이 논문은 "공식 구현을 기본 하이퍼파라미터로" 썼다고 씀(부록 D) |
| KODex, KOROL 구현 | 기준선 | GitHub 검색으로 공식 저장소를 찾지 못함 | | K-UBM 저장소에도 없음 |
| 번들 `third_party/{dexart, mjrl, mj_envs}` | 시뮬 환경(로컬 수정) | Apache-2.0 | | 외부 메시 자산은 별도 |

- 공식 저장소는 K-UBM 본체(학습·평가 스크립트)와 DexArt 3과제, Adroit 4과제의 학습된 체크포인트 7개와 시뮬 평가 러너를 담고 있습니다. 데이터는 구글 드라이브 링크로 따로 받게 되어 있고 받지 않았습니다.
- 저장소 README의 제목은 v1/v2 제목("Implicit Planners")이고 방법 이름도 "Koopman United Behavioral Models (KUBMs)"로 논문과 다릅니다(`04` D1).
