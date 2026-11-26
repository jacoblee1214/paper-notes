# 02. 선행 연구와 공식 저장소

## 0. 방법과 venue 판정

- 논문 본문(v1–v3), arXiv abs 페이지와 API(2026-11-26 조회), 프로젝트 페이지 `fingervip.github.io`, 공식 저장소 `zhzhen0626/FingerViP`의 README와 GitHub API 메타데이터를 봤습니다.
- **venue:** `CoRL 2026`, 발표 형식은 포스터입니다. 근거는 세 가지입니다. (1) arXiv 주석 "Accepted to CoRL 2026". (2) corl.org의 공식 채택 목록(`corl.org/program/accepted-papers`)에 같은 제목이 있습니다. 목록 데이터 corl.org 공식 채택 목록 데이터에서는 포스터 133번, 세션 "Poster post-p2", 11월 10일 화요일 17:00–18:30입니다. (3) 논문 v3 첫 쪽 하단과 프로젝트 페이지 모두 CoRL 2026을 적습니다. 공식 목록에서 확인했으므로 `공식 목록 미확인` 표기는 붙이지 않습니다.
- **라이선스:** arXiv "view license" 링크는 `arxiv.org/licenses/nonexclusive-distrib/1.0/`입니다(CC 아님). 논문 그림은 공개 출력에 쓰지 않습니다.
- **코드 공개 시점:** v1(2026-04)은 "코드는 공개될 것(will be open-sourced)"이라고 썼고, 공식 저장소의 유일한 커밋은 2026-09-24(v3와 같은 날)입니다. 프로젝트 페이지 저장소도 같은 날 마지막으로 갱신됐습니다.

## 1. 계보: 손끝 시각 센서에서 손끝 카메라 정책까지

- **광학 촉각 센서:** GelSight [24], DIGIT [23], 9DTact [29] 같은 젤 기반 센서는 접촉면의 변형을 고해상도로 보지만 접촉 전의 장면은 보지 못하고, 일반 RGB 영상을 주지 않습니다(§2). FingerViP은 이 계열 대신 일반 RGB 카메라를 손끝에 둡니다.
- **시각-촉각 융합 손끝 센서:** FingerVision [38, 39]과 Finger-STS [40, 41](투명 피부 + 마커), 선택 투과 막 [42], 기계식 전환 [43], 편광 [44], StereoTac [45], VisTac [46], HydroPalm [47], 다중모달 손끝 센서 [48]. 저자는 이들이 광학 구조가 복잡하거나 시각 선명도가 낮다고 평가합니다(§2). 접촉과 근접을 한 센서로 푸는 대안이고, FingerViP은 접촉 정보를 관절 전류로 따로 받습니다.
- **손목·외부 카메라 정책:** UMI [14](손목 카메라 그리퍼), DexUMI [20], CordViP [9], RAPID Hand [12](손목 카메라가 붙은 손), DexOP [35] 등 현대 다지 손 모방학습이 쓰는 입력 구성입니다. FingerViP의 기준선(TVC, WC, TVC w/ WC)이 이 구성에 해당합니다.
- **확산 정책 기반:** Diffusion Policy [25]가 정책의 뼈대이고, 카메라 자세 위치 인코딩은 RoboPanoptes [67]에서 영감을 받았다고 밝힙니다(§3.3). 시각 특징은 CLIP ViT-B/16 [71, 72]를 얼려 씁니다. 회전은 6D 연속 표현 [69]입니다.
- **원격조작:** Apple Vision Pro 기반 인터페이스 [68]와 RAPID Hand의 재타겟팅 최적화(부록 B).
- **동시기 연구:** 같은 CoRL 2026에 FingerEye(arXiv:2604.20689, 하루 앞선 2026-04-22 제출)가 있습니다. 논문 참고문헌에는 없습니다. FingerEye는 연속 시각-촉각을 한 RGB 스트림으로 주는 센서를 쓰고, FingerViP은 일반 RGB 카메라와 관절 전류를 따로 씁니다.

## 2. 구성 요소별 출처

| 구성 요소 | 논문 근거 | 출처와 상태 |
|---|---|---|
| 20 DoF 손과 구동 | §3, §4.1 | RAPID Hand [12]를 바탕으로 한 3D 프린트 손(SYSU-RoboticsLab/RAPID-Hand, 라이선스 표기 NOASSERTION). 공식 저장소의 손 모델·보정 파일은 RAPID Hand 구조에서 가져온 것 |
| 손끝 카메라 모듈 | §3.1, 그림 2 | 자체 설계. 공식 저장소 `Hardware/`에 STEP·STL과 조립 안내(CC BY-NC-SA 4.0) |
| Vision Pro 원격조작과 재타겟팅 | §3.2, 부록 B | [68]과 RAPID Hand [12]의 최적화. 저장소 `Teleop/`에 포함 |
| 확산 정책 학습 코드 | §3.3 | 저장소는 RoboPanoptes와 UMI 코드를 바탕으로 했다고 README에 적음. `DiffusionPolicy/diffusion_policy/`에 Diffusion Policy 원본 파일(Columbia 라이선스 표기 포함)이 함께 있음 |
| 카메라 자세 인코딩 | §3.3 | RoboPanoptes [67]의 위치 인코딩 발상. 구현은 저장소 `camera_pose_embed.py`와 `transformer_obs_encoder.py` |
| 손가락 관절 전류 인코딩 | §3.3, 부록 A.6 | 자체. 전류를 접촉 단서로 쓴다는 발상 |
| CLIP 시각 인코더 | §3.3, 표 3 | OpenAI CLIP ViT-B/16, 저장소는 timm `vit_base_patch16_clip_224.openai` |
| 기구학 | §3.3 | Pinocchio와 손 URDF(저장소 `dexterous_hand/urdf/`) |

## 4. 공식 코드와 저장소

| 저장소 | 역할 | 라이선스 | 스타 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [`zhzhen0626/FingerViP`](https://github.com/zhzhen0626/FingerViP) | **공식 코드** (프로젝트 페이지 링크, README에 CoRL 2026과 논문 링크) | MIT (루트 `LICENSE`). `Hardware/`는 CC BY-NC-SA 4.0, `Teleop/`는 MIT(REAL 연구실), `DiffusionPolicy/diffusion_policy/`는 MIT(REAL, Columbia 표기), 로봇 팔·손 자산에 별도 `LICENSE` | 6 (이슈 1건) | 2026-09-24 (커밋 `aba80110bf5777cd926f394ac6e870c877e21c6c`, "init FingerViP", 단일 커밋) | `repo/`에 얕은 클론 (`--depth 1`). 정적 읽기만 함 (실행, 설치, 다운로드 없음) |
| [`fingervip/fingervip.github.io`](https://github.com/fingervip/fingervip.github.io) | 프로젝트 페이지 저장소 ("Official website of FingerViP") | 없음 | 1 | 2026-09-24 | 프로젝트 페이지 HTML만 조회 |
| [`SYSU-RoboticsLab/RAPID-Hand`](https://github.com/SYSU-RoboticsLab/RAPID-Hand) | 손 하드웨어와 제어의 원본 (논문 [12]) | NOASSERTION | 59 | 2026-04-21 | 조회만 |
| [`real-stanford/RoboPanoptes`](https://github.com/real-stanford/RoboPanoptes) | 정책 구현과 카메라 자세 인코딩 발상의 출처 (README 감사 문구) | MIT | 28 | 2025-04-17 | 조회만 |
| [`real-stanford/universal_manipulation_interface`](https://github.com/real-stanford/universal_manipulation_interface) | 정책 구현의 또 다른 출처 (README 감사 문구) | MIT | 1,647 | 2026-05-26 | 조회만 |
| [`real-stanford/diffusion_policy`](https://github.com/real-stanford/diffusion_policy) | 정책 틀의 원본 | MIT | 4,747 | 2024-12-24 | 조회만 |
| `David0412-sun/FingerVip` | 이름이 비슷한 저장소. 설명 없음, 크기가 공식 저장소와 비슷함(28,930 대 28,871 KB), 포크 표기 없음 | MIT | 0 | 2026-09-29 | **저자 링크가 아님**. 프로젝트 페이지와 README 어디에도 연결되지 않음. 내용은 보지 않음 |

- **데이터와 가중치:** 저장소 README와 프로젝트 페이지 어디에도 시연 데이터(1,012개)나 학습된 체크포인트 링크가 없습니다. 이 분석은 따로 확인하지 않았습니다(다운로드 금지).
- **저장소에 있는 것:** 손끝 모듈 CAD(STEP, STL)와 BOM·조립 안내 PDF, 모터 보정과 구동 코드, Vision Pro 원격조작·데이터 수집·배치용 ROS 노드, 데이터 처리(`process_data.py`), 학습·추론 코드와 설정.
- **저장소에 없는 것:** 기준선 정책 설정 중 손목 카메라를 쓰는 것(TVC w/ WC, WC, FTC w/ WC), 표 2의 FLOPs·지연 측정 스크립트, 표 3·5의 오프라인 MAE 평가 코드(`04` D12, D13).

## 5. 사용한 자료

- 논문 PDF v1, v2, v3와 v3 텍스트(`paper/`), arXiv abs 페이지와 API(2026-11-26), 프로젝트 페이지 HTML, corl.org 채택 목록 페이지, 저장소 README·설정·코드 전체와 GitHub API 메타데이터.
