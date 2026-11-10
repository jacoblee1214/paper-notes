# 02. 선행연구와 관련 저장소

조회일 2026-11-10 (GitHub API, 프로젝트 페이지, arXiv abs). 논문의 참고문헌 번호는 PDF 기준 [n]이다.

## 1. 계보

| 줄기 | 대표 연구 (논문 인용) | 이 논문과의 관계 |
|---|---|---|
| 시각 촉각 시뮬레이션 | Taxim [12], FOTS [13], TacSL [18], Taccel [19] | 원시 신호를 센서별로 맞추는 방향. 이 논문은 이를 피한다고 주장 |
| 물리 기반 촉각 시뮬 | Xu et al. penalty 모델 [34], elastic 시뮬 [21] | Isaac Lab용 Warp 구현은 [34]의 penalty 방식을 따른다(Sec. III-D). 이 논문의 시뮬 기여 |
| 도메인 적응·번역 | Church et al. [22], TacGen [23], 실측→시뮬 변환 | 원시 신호를 한쪽 영역으로 옮긴다. 이 논문은 공유 잠재를 쓴다 |
| 센서 간 정렬 | Narang et al. [32](FEA와 압저항 센서의 잠재 투영), Yang et al. [24], Rodriguez et al. [27] | [32]이 가장 가까운 선행연구. 이 논문은 "비대칭 투영"이 아니라 "양방향 재구성 + 대조"라고 구분한다(Sec. II-B) |
| 촉각 표현 학습 | M2CURL [29], ViTacFormer [31], UPVital [30], Binding touch [24] | 촉각–시각–언어 공간 정렬. 대부분 시뮬 안에서만 정렬하므로 sim-to-real을 다루지 않는다고 지적(Sec. II-B) |
| 대조 학습 | CPC / InfoNCE [14] | 정렬 손실 원형 |

**이 논문이 비교하지 않은 것.** [32] Narang et al.은 가장 가까운 대안인데 정량 비교가 없다. [18] TacSL, [19] Taccel, [20] UniVTAC은 시뮬레이터 대안인데 이 논문은 시뮬 품질을 직접 비교하지 않는다. 이 빈칸은 `01` §6에 적었다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 재사용 여부 |
|---|---|---|
| ViT 인코더 | Dosovitskiy et al. [33] | 표준 |
| t-SNE 시각화 | van der Maaten & Hinton [35] | 표준 |
| 촉각 시뮬 penalty 모델 | Xu et al. [34] | 논문 Sec. III-D가 명시 |
| Isaac Lab | Mittal et al. [15] (arXiv:2511.04831) | 시뮬 플랫폼 |
| ABAQUS | Dassault Systèmes [16] | 상용 FEM, 라이선스 필요 |
| 정전용량 촉각 지문 | Weichart et al. [4] | 실측 센서 |

## 3. 이 저장소의 관련 분석 (cross-reference)

| 폴더 | 무엇 | TactSpace와의 관계 |
|---|---|---|
| `gpt6_astra_tactile/` | UniVTAC 촉각 시뮬레이션(Isaac Sim 기반) | 시뮬 데이터 생성 경로. 이 저장소에서 재구현·실험을 했다 |
| `contact_disassembly/` | 촉각 힘장(TacFF)과 TacRGB, Isaac Gym 계열 | 힘장을 시뮬 힘과 실측 영상에서 얻는 방식 비교 |
| `restacvla/` | 잔차 촉각 VQ와 σ 게이트, 실측 촉각 | 촉각 잠재를 정책에 넣는 방법 비교. 정렬 방식이 아니라 잔차 방식 |
| `kinefuse/` | 고유감각·F/T·접촉을 시각 추정에 융합 | 다중 모달 융합. 쌍 데이터 문제는 다루지 않음 |
| `graspsense/` | 힘 지도와 파지 후보 재순위화 | 힘 라벨을 쓰는 하위 과제 예시 |
| `robosimstudio/` | 영상 기반 물성 식별로 Real2Sim 보정 | 시뮬 물성을 실측에 맞추는 대안. 이 논문은 물성 맞춤 없이 잠재 정렬로 간다 |

## 4. GitHub 저장소 표

| 저장소 | 무엇 | 공식 여부 | 라이선스 | 별 | 최근 커밋 (확인 방법) |
|---|---|---|---|---|---|
| `leggedrobotics/TactSpace` | 메타 저장소. 세 하위 저장소를 서브모듈로 묶음. 공개, Python | 공식 (leggedrobotics 조직) | 없음 (licenseInfo null) | 2 | `bfe39a7` "first commit", 2026-05-29 (`git clone --depth 1`) |
| `leggedrobotics/TaxSimIsaacLab` (서브모듈 `81ace87`) | Isaac Lab 촉각 시뮬(Warp). 논문이 공개한다고 밝힌 시뮬레이터 | 공식 (서브모듈 포인터) | 확인 불가 | — | **접근 불가 (GitHub API 404)**. 비공개이거나 없음 |
| `leggedrobotics/tactile_encoding` (서브모듈 `93c3f7f`) | 정렬과 TactSpace 본체 구현 | 공식 (서브모듈 포인터) | 확인 불가 | — | **접근 불가 (404)** |
| `JohannesWei/1_GitHub_TestBench-RoSkin` (서브모듈 `34362435`) | 촉각 데이터 수집(5축 로봇) | 제3자 저장소를 서브모듈로 씀 | 확인 불가 | — | **접근 불가 (404)** |
| 프로젝트 페이지 `leggedrobotics.github.io/tactspace-web` | 영상과 보충 자료. 페이지에 "Paper (Coming Soon)", "Code (Coming Soon)"이라 적힘 (2026-11-10 확인) | 공식 | 페이지 템플릿 (Nerfies 계열) | — | 저장소 상태와 불일치 |
| 동명 저장소 `Tarun2805/TactSpaces-SIte`, `tactspaces-cpu/…` | 무관한 개인 사이트·파일 | 비공식 (무관) | — | 0 | 사용하지 않음 |

**판정.** 공식 저장소 세 개 가운데 **접근 가능한 것은 메타 저장소 하나뿐이다.** 그 안의 실제 코드는 ABAQUS 스크립트와 예시 설정이며, 시뮬레이터와 정렬 구현은 확인할 수 없다. 메타 저장소 README는 "이 저장소가 TactSpace 프로젝트에 필요한 모든 저장소를 담는다"고 하지만, 두 핵심 저장소는 404다. 두 저장소가 공개 전인지, 접근 권한이 필요한지는 확인할 수 없다.

## 5. 선행 쟁점

- **센서 간 전이.** [32] Narang et al.은 시뮬 전기 신호를 실측 압저항 신호로 옮기는 잠재 투영을 학습했다. 이 논문의 정렬 방식은 비대칭 투영과 다르다고 하지만(Sec. II-B), 두 방식의 정량 비교는 없다.
- **쌍 요구.** 시각–촉각 정렬 연구 [24][29][31]은 쌍 데이터를 쓰지만 대부분 시뮬에서만 쌍을 만든다. 이 논문은 실측 쌍을 쓴다는 점에서 다르다.
