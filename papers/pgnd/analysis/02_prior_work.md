# 02. 선행연구와 관련 자료

## 1. 계보

| 계열 | 대표 연구 (논문 참고문헌 번호) | PGND와의 관계 |
|---|---|---|
| 물리 기반 변형체 시뮬 | MPM [12, 29], PBD [31], 질량-스프링 [24, 20], FEM [10, 33, 7], DER [5, 6] | MPM은 직접 비교 기준선. 물성은 재료별로 최적화(부록 B.21) |
| 입자 기반 학습 동역학 (GNN) | GBND, 즉 Dynamic 3D Gaussian Tracking for GBND [58]; AdaptiGraph [57]; Graph network simulators [3]; RoboPack [2] | 가장 가까운 학습 기준선. 같은 저자 계열. 희소 그래프와 메시지 전달이 약점이라고 주장 |
| 밀집 입자 학습 | Whitney et al. (transformer, 밀집 입자) [48, 49] | 관련 연구로만 언급. 강체 파지와 밀기에 집중 |
| 비디오 예측 세계 모델 | Learning interactive real-world simulators [56], 비디오 기반 모델 [9] | 계산이 많고 3D 구조가 없다고 주장 |
| 3DGS 기반 물리 결합 | PhysGaussian [53], PhysDreamer [59], 스프링-질량 3DGS [60], Physically embodied GS [1] | 렌더링 연결. PGND는 학습된 입자 동역학에 LBS로 가우시안을 얹음 |
| 시각 기반 분할과 추적 | Grounded-SAM-2 [38, 39], SAM [18], CoTracker [15] | 데이터 파이프라인의 구성 요소. 재현 시 필요 |
| 점군 인코더와 보조 도구 | PointNet [36], PTv3 [52], MPPI [50], GELLO [51], NVIDIA Warp [30], Sulsky MPM 원문 [44], LBS [45] | 구성 요소 출처 |

PGND가 주장하는 차이는 "입자와 격자의 하이브리드 표현 + 전역 점 인코더"입니다. GBND처럼 그래프 연결을 정하지 않고, MPM처럼 격자 위에서 속도를 정하지만 물성 법칙을 쓰지 않습니다.

## 2. 가까운 대안

- **GBND [58]**: 같은 저자 그룹이 낸 입자 그래프 학습. 논문에서 같은 데이터와 같은 평가 지표로 비교하는 유일한 학습 기준선입니다. 공개 코드가 없습니다(`04` D9).
- **MPM [12, 29]**: 물리 기준선. 파라미터는 전역 Young 계수와 마찰로 두 개만 최적화합니다. 정지 상태에서 시작한다고 가정합니다(부록 B.21).
- **입자만 쓴 모델**: 본 논문의 절제(격자 제거). 같은 구조에서 격자 효과만 분리합니다.

## 4. 공식 자료와 저장소

| 자료 | 주소 | 라이선스 | 크기 지표 | 상태 |
|---|---|---|---|---|
| 공식 코드 (kywind/pgnd) | https://github.com/kywind/pgnd | MIT (LICENSE 파일과 GitHub API 모두 MIT) | 별 95, 포크 9, 생성 2025-06-19, 마지막 push 2026-09-03 | 얕은 복제 `repo/`, 커밋 `06ef5dbbe9612b56aaee317c938dcf953074c05f` (2026-09-01). 정적 읽기만 |
| 프로젝트 페이지 | https://kywind.github.io/pgnd | 페이지 라이선스 미확인 | — | 주소만 확인. 논문의 Paper 링크는 RSS 공식 proceedings를 가리킴 |
| 데이터셋 (HF kaifz/pgnd-dataset, release-20260831) | https://huggingface.co/datasets/kaifz/pgnd-dataset | HF 카드 메타데이터에 cc-by-4.0 | 압축 해제 350 GB, 중복 제거 101 GB, 궤적 2180 (학습 2020, 평가 160) | **다운로드하지 않음.** 매니페스트 JSON만 읽음 |
| 체크포인트 (HF kaifz/pgnd-checkpoints, release-20260831) | https://huggingface.co/kaifz/pgnd-checkpoints | 라이선스 필드 없음 | 12개 파일, 합계 20.3 MB, 범주당 약 3.4 MB | **받지 않음.** 매니페스트 JSON만 읽음 |
| 데모 (HF Space kaifz/pgnd) | https://huggingface.co/spaces/kaifz/pgnd | 미확인 | — | 주소만 확인 |
| 제3자 렌더러 `diff-gaussian-rasterization-w-depth` | 저장소 `third-party/` | **Gaussian-Splatting License** (비상업 연구용 조건) | — | 정적으로만 확인. 라이선스가 MIT와 다름 |

- 공식 저장소의 README에 따르면 데이터셋은 2026-08-31에 "버전 있는 릴리스"로 공개되었습니다. 2026-01-05에 "데이터 정리"가 있었습니다. **논문 표의 수치가 어느 데이터 버전에서 나왔는지는 확인할 수 없습니다.**
- 기준선 MPM, GBND, Particle의 코드는 저장소에 없습니다.

## 5. 학회와 판본

- **학회:** RSS 2025 (RSS XXI 온라인 proceedings, 논문 번호 p036). 프로젝트 페이지의 "Robotics: Science and Systems (RSS), 2025"와 proceedings 페이지가 일치합니다(`https://www.roboticsproceedings.org/rss21/p036.html`). 발표 형식(구두 또는 포스터)은 확인하지 못했습니다.
- arXiv 댓글란에는 학회 정보가 없고 "Project page"만 있습니다.
