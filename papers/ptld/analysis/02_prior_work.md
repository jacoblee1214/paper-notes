# 02. 선행연구, 코드 공개 상태, 관련 분석

## 1. 계보

PTLD는 세 줄기를 이어 붙입니다. 인용 번호는 논문 참고문헌 번호입니다.

| 줄기 | 선행 | PTLD가 가져온 것 | 달라진 점 |
|---|---|---|---|
| 특권 정보 증류 | Learning by Cheating [26], DAgger [50], RMA [28], HORA [11] | "오라클 → 부분 관측 학생" 틀, 잠재 모방(식 L_latent), 학생 경험으로 모으는 DAgger | 학생의 입력을 **시뮬레이션에서 흉내 낼 수 없는 센서(촉각)**로 두고, 증류를 **실제 셀에서** 한 번 더 함(§4.2) |
| 비대칭 액터-크리틱 | Asymmetric Actor Critic [29] | 크리틱만 특권 상태를 봄. 2단계 증류를 1단계로 줄임 | 액터와 크리틱 인코더 사이에 자기 증류 손실(식 1, BYOL [51], data2vec [52] 계열)을 추가 |
| 촉각 표현·적응 | Sparsh [47], Sparsh-skin [43], 촉각 적응 [17], 펜 돌리기 [49] | Sparsh-skin 표현을 기준선으로 사용 | 교사를 **고유수용 정책이 아닌 특권 센서 정책**으로 바꿈(§2) |
| 촉각 시뮬레이션 | TACTO [23], TacSL [24], [25] | 비교 대상(시뮬레이션 비용이 동기) | 촉각을 시뮬레이션하지 않음 |
| 제로샷 손 안 조작 | Dextreme [13], Rubik's cube [15], AnyRotate [18], 촉각 회전 [20], 일반 회전 [21], 손 안 이동 [22] | 과제와 지표(RotR, TTF, RotP) 설정 | 제로샷이 아니라 실제 증류를 추가 |

## 2. 가까운 대안과 비교

| 대안 | 방식 | PTLD와의 차이 |
|---|---|---|
| 촉각 시뮬레이션 + 제로샷 (AnyRotate [18], [25]) | 촉각을 시뮬레이션해 RL | 시뮬레이터 정확도에 의존. PTLD는 시뮬레이션이 필요 없음 |
| 촉각 적응 [17] | 고유수용 RL 정책을 교사로 촉각 학생을 학습 | 교사 성능이 상한. PTLD는 포즈를 보는 교사를 씀 (논문 비교: 그림 6) |
| 시연 기반 촉각 모방 (§1: [1, 2, 3, 4]) | 텔레옵 시연 + BC | 다지 손에서 시연 수집이 어려움 (§1) |
| 시각 증류 (Dextreme류) | 렌더링 기반 증류 | 시각 렌더링 비용 (§1). PTLD는 결론에서 시각으로 일반화 가능하다고만 언급 (§7, 실험 없음) |

## 4. 코드와 공개 상태 (2026-11-23 확인)

| 자원 | 주소 | 상태 | 비고 |
|---|---|---|---|
| 공식 코드 | https://github.com/akashsharma02/ptld | **HTTP 404** (`gh api` 404, 웹 404) | 프로젝트 페이지의 "Code" 버튼이 이 주소를 가리킴. 비공개이거나 아직 공개 전 |
| 프로젝트 페이지 | https://akashsharma02.github.io/ptld-website/ | 200 | 초록, 영상(https://youtu.be/YVFZt2YAS3Y), PDF(ptld_2026.pdf), arXiv 링크. 코드 링크는 위 404 주소 |
| 저자 계정의 관련 저장소 | akashsharma02/hora | HaozhiQi/hora의 포크, 마지막 push 2025-07-22 | 논문의 시뮬 과제 [11]과 같은 계열이나 PTLD 코드라는 표시는 없음 |
| akashsharma02/xela_server_ros2 | | MIT, 별 0, 2024-07-17 | Xela 센서 ROS2 서버. 하드웨어 연동 단서 |
| akashsharma02/tactile-dexterity | | 라이선스 표시 없음, 2024-01-29 | 이전 연구(Dexterity from Touch) 코드 |
| HaozhiQi/hora | https://github.com/HaozhiQi/hora | LICENSE는 MIT 형식 본문(저작권 Haozhi Qi 2022), 별 262, push 2026-07-08 | PTLD 시뮬 과제의 원형 [11] |
| facebookresearch/sparsh-multisensory-touch | | LICENSE.md는 CC BY-NC 4.0, 별 56, push 2025-08-20 | Sparsh-skin·Sparsh-X, 촉각 적응 기준선 [17][43] |
| isaac-sim/IsaacGymEnvs | | LICENSE.txt, 보관(archived), push 2024-10-26, 별 2949 | 논문이 쓴 시뮬레이터 [57]. 신규 GPU 시뮬레이터로의 이전은 논문에 없음 |
| borglab/gtsam | | LICENSE(API 분류 Other), 별 3805 | 포즈 그래프 최적화 [56] |
| NVlabs/FoundationPose | | NVIDIA 라이선스(비상업 계열 문구), 별 3712, push 2026-04-29 | 정육면체 포즈 추정 [55] |

- `gh search repos`로 "ptld", "privileged tactile latent distillation"을 검색했을 때 이 논문의 구현은 나오지 않았습니다(다른 무관 저장소만 나옴).
- 프로젝트 페이지 HTML에는 주석 처리된 "Accepted as Oral at CoRL 2025" 문구가 남아 있습니다(표시되지 않음). 학회 판단에는 쓰지 않았습니다.
- 라이선스 열은 GitHub API 분류 또는 LICENSE 파일 첫머리를 읽은 것이며, 법적 판단이 아닙니다.

## 5. 학회

**CoRL 2026 포스터.** corl.org 공식 채택 논문 표(https://www.corl.org/program/accepted-papers)에서 직접 확인했습니다(포스터 번호 153, 세션 Poster post-p2, 11월 10일 화요일 오후 5:00–6:30, 제목 "PTLD: Sim-to-Real Privileged Tactile Latent Distillation for dexterous manipulation"). arXiv comments와 journal-ref는 비어 있고, PDF 본문에도 학회 표기는 없습니다(v3는 CoRL 형식).
