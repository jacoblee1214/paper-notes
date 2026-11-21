# 02. 선행 연구와 공식 저장소

## 0. 방법과 venue 판정

- 논문 본문, arXiv abs 페이지 (2026-11-21 조회), 프로젝트 페이지 `dex-umi.github.io`, 공식 저장소 `real-stanford/DexUMI` README와 API 메타데이터를 봤습니다.
- **venue:** PMLR 공식 프로시딩 `proceedings.mlr.press/v305/`에서 "Proceedings of The 9th Conference on Robot Learning, PMLR 305:437-459"의 "DexUMI" 항목을 확인했습니다. 따라서 `CoRL 2025`로 둡니다.
- **최우수 논문 표기:** 프로젝트 페이지는 "CoRL 2025 Best Paper Final List"라고 씁니다. 공식 CoRL 2025 수상 목록은 찾지 못했습니다. 웹 검색으로 본 Stanford SAIL CoRL 2025 블로그와 제3자 목록은 "Best Paper Finalist"로 적지만 이는 2차 자료입니다. 그래서 "최우수 논문 후보 표기는 공식 목록 미확인"으로 남깁니다.
- corl.org의 `accepted-papers` 페이지는 CoRL 2026 목록입니다. 이 목록에는 DexUMI가 없습니다.

## 1. 계보: UMI에서 손 모듈까지

- **UMI** (Chi 외, arXiv:2402.10329): 손목 카메라가 붙은 핸드헬드 그리퍼로 로봇 없이 시연을 모읍니다. DexUMI의 상대 행동 표현과 시각 지연 보정(QR 코드)은 여기서 가져왔습니다 (부록 E.2, 논문 [1]).
- **병렬 그리퍼 계열** [1–3, 48–57]: 단순 집게 그리퍼에 한정되어 다지 손에 적용하기 어렵습니다 (§2).
- **모션 캡처 장갑:** DexCap [58]은 재타겟팅과 로봇 데이터가 필요합니다. DexUMI는 이 둘을 없앤다고 주장합니다 (§2).
- **손 위 손(hand-over-hand) 시스템** [59, 60]: 로봇 손이 실제로 있어야 합니다.
- **원격조작 대 외골격:** 손가락 매핑을 직접 맞추는 외골격 연구 [37]는 기계 구조 차이 때문에 정확도가 제한됩니다. DexUMI는 설계 최적화로 이 간극을 줄입니다 (§3.1).

## 2. 구성 요소별 출처

| 구성 요소 | 논문 근거 | 출처와 상태 |
|---|---|---|
| 외골격 설계 최적화, 식 (1) | §3.1, 부록 A·E.2 | 자체 기여. 코드 대조는 `04` D1 |
| 관절 엔코더 + 회귀 | S.1 | 자체. 저장소에는 일부 (`linkage_optimization/simple_regression.py`) |
| 손목 자세 (ARKit) | S.2 | iPhone + Record3D 포크 (README 링크) |
| 손목 광각 카메라 | S.3, 부록 A | OAK-1 (논문 [63]) |
| 촉각 (FSR, 전자기) | S.4 | 외부 센서. XHand 내장, Inspire는 FSR [64] |
| 손 분할 | V.1 | SAM2 (논문 [65]) |
| 배경 복원 | V.2 | ProPainter (논문 [66]) |
| 정책 백본 | 부록 E.3 | DINO 비전 특징 (논문 [76, 77]), diffusion policy [78, 79] |
| 상대 손가락 행동 | §5, Table 1 | UMI의 상대 궤적 표현 [1]에서 가져옴 |

## 4. 공식 코드와 저장소

| 저장소 | 역할 | 라이선스 | 스타 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| `real-stanford/DexUMI` | **공식 코드** (프로젝트 페이지 "Code" 링크, README 저자 표기) | MIT | 280 | 2026-04-18 (커밋 `acddb8f`, "Update bibtex.") | `dexumi/repo/`에 얕은 클론 (`--depth 1`, LFS 제외). 정적 읽기만 함 |
| `dex-umi.github.io` | 프로젝트 페이지 (Hardware·Deployment 가이드 링크) | — | — | — | 조회함 |
| `facebookresearch/sam2` | 분할 모델 (README 명시) | — | 미조회 | 미조회 | 가중치 `sam2.1_hiera_large.pt` 다운로드 안 함 |
| `sczhou/ProPainter` | 배경 복원 (README 명시) | — | 미조회 | 미조회 | 설치·실행 안 함 |
| `mengdaxu/record3d` | Record3D 포크 (README 명시, Python 3.10 호환) | — | 미조회 | 미조회 | 설치 안 함 |
| `gray-wei/DexUMI` | 동명 저장소. 저자 README 링크가 아님 | MIT | 0 | 2025-11-10 | **비공식** (README와 프로젝트 페이지에 없음) |
| `DexUMI/DexUMI` | 동명 저장소 | 없음 | 0 | 2025-05-11 | 비공식 |
| `roboterax/xhand1-pro-u` | XHAND1 Pro-U URDF. 설명에 "DexUMI solution initiative" 문구 | MIT | 0 | 2026-11-13 | 제3자 로봇 손 자료 (논문 코드 아님) |

- 데이터: 프로젝트 README는 샘플 데이터 `real.stanford.edu/dexumi/sample_data.zip`을 링크합니다. 다운로드하지 않았습니다 (brief: 데이터 다운로드 금지).
- 학습 가중치: 저장소에 없습니다. DINO와 SAM2 가중치는 외부에서 받아야 합니다.

## 5. 사용한 자료

- 논문 PDF v3 (`paper/paper.pdf`, 22.4 MB), v1·v2 PDF와 텍스트 (gitignore).
- arXiv abs 페이지 (2026-11-21), 프로젝트 페이지 HTML, 저장소 README와 커밋 목록 (GitHub API), PMLR v305 목록.
