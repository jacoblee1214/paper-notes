# 02. 선행 연구, 구성 요소 출처, 저장소 관련 분석, GitHub 자료

## 1. 선행 연구 계보

RFW-S는 세 흐름이 만나는 곳에 있습니다.

1. **대규모 자산 생성과 물리 속성 부여**: Objaverse-XL, ABO, OmniObject3D(시각 자산). 최근에는 ManiTwin-100K(강체, 512 범주, 검증된 파지), PhysX-Omni(강체·변형·관절), Seed3D, PhysX-Anything(물리 자산 생성) (논문 2절, 표 1).
2. **변형체 시뮬레이션 환경**: SoftGym, PlasticineLab, DaXBench, GarmentLab, BEHAVIOR-1K, DeformableRavens (논문 2절).
3. **물리 파라미터 추론과 피드백**: PUGS, Cerberus, PAC-NeRF, PhysTwin(영상에서 물리 재구성), Vid2Sid, Swim2Real, Agentic Real2Sim (논문 2절).

논문의 차별점은 (a) 범주 균형 생성과 **구조 인지 표현 선택**, (b) 보정과 **독립 검증의 분리**입니다. 저자는 Agentic Real2Sim(반복 재생 판정)과 OopsieVerse(손상 인지 시뮬레이션)와 비교합니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 참고 이미지 생성 | Qwen-Image-2.0 (arXiv:2605.10730) | 논문 4.1절. 실행 환경(로컬 또는 API)은 본문에 없음 |
| 3D 생성 | TRELLIS.2 (arXiv:2512.14692) | GPU 필요 |
| 시뮬레이터 | Genesis 1.3.3 (PBD, MPM) | 저장소 `method_contract.schema.json`에 고정 |
| 물리 파라미터 제안 | Qwen3.8-max (API) | 온도 0.1 / 0.2, thinking 끔 (표 3) |
| 보정 방법 비교 대상 | Vid2Sid, Swim2Real (텍스트 전용 재구현), Sobol, rule-based | 논문 E.1절. 원 논문의 재구현이 아닌 "스타일" 재구현 |
| 파지 손상 지표 | 변형 구배 ‖F − I‖_F 기반 대리(proxy) | 논문 6.3절: 회전에도 반응함을 저자가 인정 |
| 통계 | 자산 단위 부트스트랩 5,000회 | `repo/paper/build_support/build_results.py` |

## 4. GitHub, 프로젝트, 데이터 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 | 마지막 푸시 | 비고 |
|---|---|---|---|---|---|
| **공식 코드 RoboFlywheel-Soft** | https://github.com/RoboFlywheel-com/RoboFlywheel-Soft | Apache-2.0 | 1 | 2026-11-20 | 기본 브랜치 main. 최근 커밋 `5373cd9`(docs: ModelScope 링크). 얕은 clone을 `repo/`에 두고 정적 읽기만 함 |
| 프로젝트 페이지 | https://roboflywheel-com.github.io/RoboFlywheel-Soft/ | - | - | - | 코드와 데이터 "SOON"이라 적혀 있음 (`04` D13) |
| 데이터(ModelScope) | https://modelscope.cn/datasets/RoboFlywheel/RoboFlywheel-Soft | 저장소 README 기준 별도 | - | - | 접속과 크기는 확인하지 않음. 전체 자산, 이미지, 메시, 물리 번들은 여기에만 있음 |
| RFW-A 코드 (선행 연구, arXiv:2610.40333) | https://github.com/RoboFlywheel-com/RoboFlywheel-Articulation | Apache-2.0 | 5 | 2026-11-19 | 같은 저자 그룹. 관절 물체 400K 자산. 논문 참고문헌 [39] |
| 같은 조직의 기타 저장소 | RoboFlywheel-Sim, -World, -Recipe, RoboFlywheel | 없음 | 0–2 | 2026-09 | 이름만 확인. 내용은 읽지 않음 |
| 논문 PDF v1 / v2 | arxiv.org/pdf/2611.23056v1, v2 | non-exclusive | - | - | v2 본문이 v1과 같음 (`01` §6) |
| 라이선스 | arxiv.org/licenses/nonexclusive-distrib/1.0/ | - | - | - | abs 페이지의 "view license" 링크 |

- 저장소 라이선스 주의: 코드는 Apache-2.0이지만 **Qwen, TRELLIS.2, Genesis 등 제3자 가중치와 데이터는 각자 라이선스**를 따릅니다(`docs/THIRD_PARTY.md`).
- DiagGen (arXiv:2609.23103)은 웹 검색에서 찾은 "시뮬레이션 진단으로 변형 자산을 만드는" 연구입니다. 제목은 RFW-S와 다르지만 같은 문제 영역입니다. 검증하지 않았습니다.
