# 02. 선행연구와 관련 저장소

이 문서는 UCA-Flow의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 관련 저장소가 있는지를 정리합니다. 번호는 논문 참고문헌(`paper/paper.txt`의 References)에서 따온 저자와 연도로 적었습니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| UCA-Flow의 요소 | 출처 (논문 인용) | 관계 |
|---|---|---|
| 평균 속도 u와 순간 속도 v의 관계 (평균 흐름) | MeanFlow, Geng et al. 2025a; Improved MeanFlow, Geng et al. 2025b | 식 9–10의 평균 흐름 항등식과 JVP 방향이 여기서 옴. 논문은 "isolated MeanFlow objective"가 아니라 두 속도를 함께 감독한다고 구별함 (§1) |
| 로봇 정책의 한 스텝 생성 (평균 흐름 기반) | MP1, Sheng et al. 2025 (AAAI'26) | 가장 가까운 기준선. 점군 특징과 평균 흐름을 씀. 논문의 표 2–3에서 가장 강한 기준선 (78.9%, 6.8 ms) |
| 한 스텝 흐름 정책 | FlowPolicy, Zhang et al. 2025 (AAAI'25) | 일관성 흐름 매칭 기반 3D 흐름 정책. 1 NFE 기준선 |
| 3D 점군 정책 | DP3, Ze et al. 2024 (RSS 2024) | 10 NFE 확산 기준선. Simple DP3는 같은 계열의 단순 변형 |
| 확산 정책 | Diffusion Policy, Chi et al. 2025 (IJRR) | 확산 정책 계열의 원류. 이 논문의 기준선 설정에 직접은 쓰이지 않음 |
| 빠른 생성 | DDIM, consistency models, rectified flow (Liu et al. 2022), Song et al. 2023 | 샘플링 단계 감소 계열. 논문은 "conditions injected as external context"라고 대조함 |
| 외부 주입 조건 (AdaLN) | 논문 Table 1의 −Unified/a 변형, 로봇 Transformer 계열(Reuss et al. 2024; Yan et al. 2025) | 조건을 행동 백본에 변조(modulation)로 넣는 방식의 대표. 논문에서 가장 약한 변형 |
| 외부 주입 조건 (교차 어텐션) | 논문 Table 1의 −Unified/c 변형 | 행동 토큰이 조건 토큰을 교차 어텐션으로 읽는 방식 |
| 조건과 행동의 공동 토큰 어텐션 | 논문 인용 없음 | UCA-Flow의 핵심. 조건 토큰을 행동 토큰과 같은 셀프 어텐션 열에 둠. 이 조합이 이미 쓰인 사례는 이 분석에서 찾지 않음 (`01` §2.2) |
| 행동 청크 호라이즌과 관측 창 | DP3, MP1 설정 (Ze 2024; Sheng 2025) | 관측 2, 호라이즌 4 (시뮬), 16 (실로봇). 기준선과 같은 설정을 따름 |
| 점군 인코더 | PointNet (Qi 2017a), PointNet++ (Qi 2017b), PointNeXt (Qian 2022) | 점군을 특징으로 줄이는 부분. 공식 저장소는 PointNet 계열 MLP 인코더를 씀 (`04` D8) |
| 시뮬레이션 벤치마크 | Adroit (Rajeswaran 2017), Meta-World (Yu 2020) | 37과제 구성 |
| 실로봇 기반 | UR5e (로봇), RealSense L515 (카메라) | 실험 장비 |

**가까운 대안과의 차이.** 가장 가까운 대안은 MP1입니다. MP1과 UCA-Flow는 둘 다 한 스텝 평균 흐름 정책이고 점군 입력을 씁니다. 차이는 조건을 어떻게 넣느냐입니다. MP1은 조건을 외부 문맥으로 넣고(논문 §2), UCA-Flow는 조건 토큰을 행동과 같은 열에 둡니다. 논문의 절제 결과는 이 차이가 AdaLN이나 교차 어텐션 대비 성능 차이를 만든다는 증거입니다(`03` §1.5). MP1 자체와 직접 비교하는 절제는 없습니다. 같은 백본에서 MP1 조건 방식만 바꾼 변형이 없으므로, 전체 차이 중 어느 부분이 설계 차이인지는 가릴 수 없습니다.

## 2. 코드와 저장소 (GitHub 표)

| 저장소 | 관계 | 라이선스 | 대조 기준 | 상태 |
|---|---|---|---|---|
| [NTUMARS/UCA-Policy](https://github.com/NTUMARS/UCA-Policy) | 공식 구현. 프로젝트 페이지의 Code 링크(옛 저장소 주소)가 이 저장소로 넘어감. 별 3, 포크 0, 생성 2026-08-19, 마지막 push 2026-10-23 | MIT (루트 LICENSE, "UCA-Flow Authors"). 재사용 코드는 원 라이선스 유지 | `40084a903d03417a4d32b09506c0a48a11b2dba4` (2026-10-23, 얕은 복제) | 정적 읽기만 (`04`). 실행, 설치, import 안 함. 137 MB |
| [uca-policy.github.io/UCA.github.io](https://uca-policy.github.io/UCA.github.io/) | 프로젝트 페이지. 논문의 "Project page" 링크. 페이지 자체는 Academic Project Page 템플릿(CC BY-SA 4.0) | 페이지 템플릿 라이선스 (논문 라이선스와 별개) | 2026-11-22 확인 | 정적으로 확인: "Robotics: Science and Systems (RSS) 2026"으로 표기, 영상의 수치는 정정 전이라는 안내가 있음 |
| [LogSSim/MP1](https://github.com/LogSSim/MP1) | MP1 공식 구현. UCA-Flow 저장소가 일부 코드를 가져옴(`THIRD_PARTY_NOTICES`) | 저장소 API 라이선스 `null`. 저장소 안의 `LICENSES/MP1-MIT.txt`는 MIT 문구를 담음 | 마지막 push 2025-11-26, 별 71 | 정적 확인 안 함. UCA-Flow에서 MP1 파생 코드는 `meanpolicy.py` 계열로 보임(추정, 파일명 기준) |
| [YanjieZe/3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) | DP3 공식 구현 (기준선) | MIT | 마지막 push 2025-10-17, 별 1,475 | 정적 확인 안 함. 저장소 내 `dp3.py` 대조는 `04` §3 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | Diffusion Policy 공식 구현 (기준선 계열) | MIT | 마지막 push 2024-12-24, 별 4,737 | 정적 확인 안 함 |
| FlowPolicy 공식 저장소 | 기준선 | 찾지 못함 | — | 검색 결과 없음 (이 분석의 검색 범위 안) |
| MeanFlow 공식 저장소 | 평균 흐름 원저 | 찾지 못함 | — | 검색 결과 없음 (추측 이름으로 확인한 저장소는 404) |

- 구현 참고: 공식 저장소의 `README.md`는 옛 저장소 주소를 안내합니다. 그 주소는 위 저장소로 넘어갑니다. 공개용 문서에는 `NTUMARS/UCA-Policy`를 씁니다.
- 프로젝트 페이지 BibTeX는 `@misc`이고 학회 이름이 없습니다. 페이지 본문에는 RSS 2026이 적혀 있습니다. 학회 확인은 아래 §4에 있습니다.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **DP3 기준선.** 표 2의 DP3 값(Adroit Hammer 100, Door 56, Pen 46; Meta-World Easy 87.3, Medium 44.5, Hard 32.7, Very Hard 39.4; 평균 68.7)은 본문에서 출처를 적지 않았습니다. DP3 원논문(arXiv v7)은 Easy 28, Medium 11, Hard 6, Very Hard 5의 과제 묶음을 쓰므로 같은 표에서 바로 가져온 값은 아닐 가능성이 큽니다. 원출처는 확인하지 못했습니다(`04` D18).
- **MP1 기준선.** 표 2의 MP1 값(평균 78.9%)은 MP1 원논문(AAAI'26)에서 가져온 것으로 보이나 본문에 출처 표기가 없습니다. 같은 확인이 필요합니다.
- **학회 확인.** 프로젝트 페이지는 RSS 2026을 표기합니다. RSS 2026 공식 accepted 목록(roboticsconference.org/2026/program/papers/)에서 제목과 저자 이름(Zhou Xinyu, Cai Zikun, Yang Jianfei, Chen Jiayu)을 검색했으나 해당 논문을 찾지 못했습니다. 따라서 학회 표기는 "프로젝트 페이지 표기, 공식 목록 미확인"으로 둡니다(`publish/blog_post.md`의 venue_note).
- **정정 (메인 세션, 2026-11-22).** CoRL 2026 공식 채택 목록(`conferences/corl2026/papers.json`)에 포스터(Poster post-p2, 67번)로 올라 있습니다. 따라서 학회는 **CoRL 2026**으로 적습니다. 프로젝트 페이지의 RSS 2026 표기는 공식 목록과 맞지 않습니다.
