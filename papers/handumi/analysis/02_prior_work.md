# 02. 선행연구와 관련 저장소

## 1. 계보

HandUMI는 로봇 없이 시연을 모으는 UMI 계열의 한 갈래입니다. 계열의 핵심 문제는 "사람이 든 기록 장치의 그리퍼가 로봇 그리퍼와 다르면 정책이 전이되지 않는다"입니다.

| 연구 | 장치 방식 | 그리퍼 대응 | HandUMI와의 관계 | 출처 |
|---|---|---|---|---|
| UMI [5] | 손에 든 그리퍼, 단안 관성 SLAM | 로봇 그리퍼를 본뜬 소프트 핑거 | 원형. HandUMI는 소프트 핑거를 팁 중 하나로 씀 | RSS 2024 (공식 proceedings) |
| UMI-3D [12] | LiDAR 관성 오도메트리 | UMI 그리퍼 | 3D 추정 확장 | arXiv:2604.14089 (논문 참고문헌) |
| FastUMI [9] | T265 VIO, 팁 교체 | 하드웨어 독립 | 팁 교체와 함께 비교 | CoRL 2025, PMLR 305 (공식) |
| DexUMI [13] | 사람 손 + iPhone ARKit, 엔코더 | 손 대응 | 손 인터페이스 계열 | CoRL 2025, PMLR 305 (공식) |
| YUBI [6] | Quest 컨트롤러, 엔코더, 회전 턱 | 로봇마다 YUBI 그리퍼로 교체 | 가장 가까운 비교 대상. 논문은 직접 비교하지 않는다고 함 | arXiv:2606.10244, CoRL 2026 포스터 (공식 목록: `conferences/corl2026/papers.json`) |
| KIWI [8] | 360° 카메라와 SLAM, 교체 가능한 도구 모듈 | 도구 모듈 | 턱 폭을 영상으로 추정. 소프트웨어 공개는 예고 단계 | arXiv:2609.22809 |
| ActiveUMI [10] | Quest 컨트롤러 | 그리퍼 복사 | 그리퍼 대응 방식 비교 | ICRA 2026 (논문 참고문헌, IEEE 링크) |
| RealDexUMI | 손 착용 장치 | 손 | 손 계열 | CoRL 2026 포스터 (공식 목록) |
| EgoMI, XRZero-G0, RDT2, HiFi-UMI | Quest 3S, PICO 4, Vive, 헤드 SLAM | 다양 | 표 1 비교 대상 | 논문 표 1의 참고문헌 |

**정책 학습 측면.** 논문이 인용하는 정책 기준은 ACT [2]와 π0.5 [11]입니다. π0.5는 PMLR 305 (CoRL 2025)에 실려 있습니다. 논문 §3.2는 π0.5 미세 조정에 "5–100+ 시간 과제 데이터"가 쓰인다는 RaC [18]와 π0 [17]의 수치를 근거로 들어 HandUMI의 1.9시간을 상대화합니다(프로젝트 페이지 §3).

## 2. 공식 코드와 프로젝트 자료

논문 본문과 프로젝트 페이지가 가리키는 저장소는 세 개이고, GitHub API로 2026-11-22에 확인했습니다.

| 저장소 | 역할 | 라이선스 (GitHub API) | 별 / 포크 | 마지막 푸시 | 확인한 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [murobotics-ai/handumi-sw](https://github.com/murobotics-ai/handumi-sw) | 기록, 보정, 검사, 재생, 재타깃팅 소프트웨어 | Apache-2.0 | 85 / 10 | 2026-11-08 | `b9258f07` (2026-11-08, "fix(calibration): adjust four simulation table placements") | 얕은 복제 완료 (`repo/handumi-sw/`) |
| [murobotics-ai/handumi-hw](https://github.com/murobotics-ai/handumi-hw) | 3D 인쇄 STL, STEP, BOM, 조립 안내 | Apache-2.0 | 120 / 9 | 2026-11-05 | `ad9e4119` (2026-11-05, "pico controller support stl files") | 얕은 복제 완료 (`repo/handumi-hw/`) |
| [murobotics-ai/handumi-quest-app](https://github.com/murobotics-ai/handumi-quest-app) | Quest 앱, APK 역공학 재구성 | Apache-2.0 (NOTICE에 예외 조항) | 4 / 1 | 2026-10-19 | `1ed5935f` (2026-10-19, "Add Apache-2.0 license, NOTICE and HandUMI links") | 얕은 복제 완료 (`repo/handumi-quest-app/`) |
| [murobotics project page](https://murobotics.ai/research/handumi) | 논문 판, 영상, 링크 | 해당 없음 | 해당 없음 | 페이지 "Updated 19 October 2026" | 해당 없음 | 읽음 (`project.txt`) |
| Hugging Face collection `murobotics/handumi-towel-folding` | 데이터셋 3종 이상, 체크포인트 3종 | 페이지 확인 필요 | 해당 없음 | 해당 없음 | 해당 없음 | **받지 않음** (규칙: 가중치와 데이터 다운로드 금지) |
| YUBI `airoa-org/yubi-sw` | 원 ROS 브리지, Quest 프로토콜 원천 | 논문은 Apache-2.0이라고 함 | 해당 없음 | 해당 없음 | 해당 없음 | 조회만 (이 분석의 대조 대상 아님) |

- 세 저장소는 모두 **정적 읽기만** 했습니다. 설치, 실행, import 없음. 복제본은 `repo/`에 있고 `.gitignore`에 들어 있습니다.
- `handumi-sw`에는 `axol`과 `franka_emika_panda` 시뮬 설정이 있는데, 논문 표 3에는 없습니다. 저장소에 있다고 논문 범위가 늘어난 것은 아닙니다.

## 4. 학회 위치

HandUMI 자체는 학회에 실렸다는 근거가 없습니다(`analysis/03_analysis.md` §0). 인용하는 쪽도 "학회 표기 없음 (arXiv)"으로 적는 것이 맞습니다.
