# 02. 선행연구, 학회 확인, 코드 탐색

## 1. 학회와 발표 상태

- **판정: 학회 표기 없음 (arXiv).** 제출 2026-11-18 (v1 하나), 저널 참조 없음, 프로젝트 페이지 링크 없음.
- 확인한 목록: CoRL 2026과 IROS 2026 공식 채택 목록 (제목과 초록 검색, 일치 없음), 웹 검색 (제목 검색, 일치 없음). RSS, PMLR, IEEE 색인은 제목으로 직접 찾지 못했습니다. 발표일이 최근이라 목록이 아직 나오지 않았을 수 있습니다.
- 본문에 CoRL, RSS 등 학회 이름은 없습니다. 참고문헌의 `ECCV 2026` (TouchAnything [14])은 인용 대상이지 이 논문의 학회가 아닙니다.

## 2. 선행연구 계보

### 2.1 가우시안 과정 암묵 곡면과 능동 촉각 탐색

- **GPIS 원전**: Williams & Fitzgibbon (2006, [15]), Dragiev, Toussaint, Gienger (ICRA 2011, [7]). 촉각 접촉을 부호 거리 제약으로 쓰는 방법의 출발점입니다. TapTacTouch는 이 틀을 그대로 씁니다.
- **능동 촉각 탐색**: Yi 외 (IROS 2016, [9,11]), Matsubara & Shibata (RAS 2017, [10]), Luan 외 (Measurement 2025, [6]). 불확실성과 이동 비용으로 다음 접촉을 고릅니다. TapTacTouch는 **규칙 기반 고정 순서**를 씁니다. 논문 §V가 적응 탐색을 향후 과제로 밝힙니다.
- **다지 손 탐색과 형상 추정**: Chen 외 (IROS 2023, [8]), TactoFind (Pai 외, ICRA 2023, [12]). 여러 손가락으로 접촉을 모으는 설계와 가장 가깝습니다.

### 2.2 시각-촉각 및 학습 기반 형상 복원

- **TouchSDF** (Comi 외, RAL 2024, [5]): 촉각 이미지 CNN과 DeepSDF. 고밀도 촉각 이미지가 전제입니다.
- **Touch2Shape** (Wang 외, CVPR 2025, [13]): 촉각 조건부 3D 확산.
- **TouchAnything** (Gu 외, ECCV 2026, [14]): 사전학습 2D 확산과 학급 설명을 씁니다. TapTacTouch가 "사전 모델 없이"라고 강조하는 대조 대상입니다.
- **NeuralFeels** (Suresh 외, Science Robotics 2024, [4]), **ShapeMap 3D** (Suresh 외, ICRA 2022, [2]): 시각과 촉각을 함께 쓰는 형상 지도.
- 능동 시각-촉각 형상 완성 (Rustler 외, RAL 2022, [3]).

### 2.3 TapTacTouch의 자리

- 시각 없음, 학습 없음, 범주 무관. 학습 기반 연구와 달리 **접촉 기하를 그대로 씁니다.**
- 기여는 세 가지입니다. (a) 다지 손 FK 접촉을 GPIS로 옮기는 파이프라인, (b) 접촉 신뢰도와 가상 접촉, (c) 층별 단면 사전(CSP). (c)가 가장 새로운 부분입니다.
- 비교 대상에 TouchAnything, Touch2Shape, TouchSDF 같은 학습 방법이 **없습니다.** 결과 비교가 "단일 척도 GPIS 대 제안법"으로만 이루어집니다. 학습 방법과의 정량 비교가 없다는 것이 이 논문의 큰 빈칸입니다(`04` C10).

## 4. GitHub, 프로젝트 페이지, 관련 코드

| 구분 | 확인 결과 | 방법 | 라이선스 / 별 / 마지막 푸시 | 복제 |
|---|---|---|---|---|
| 공식 코드 | **찾지 못함** | arXiv abs 링크 확인 (코드 링크 없음), GitHub 저장소 검색 (제목 "TapTacTouch", "tactile GPIS shape" 등, 결과 0) | - | 없음 |
| 프로젝트 페이지 | **찾지 못함** | 논문 본문과 abs에 URL 없음, 웹 검색 결과 없음 | - | - |
| 책임 저자(Andrew Jaeyong Choi) GitHub 계정 | 본인 계정인지 확정하지 못함 | 이름에서 추정한 계정에 공개 저장소 없음 | - | - |
| 연구실 조직 계정 | 공개 저장소 4개 | 저장소 목록 | 촉각 관련 저장소 없음 (automated-feature-engineering, AutoRace-ImageSet, Gazebo_Navigation_Map, Robotics_vnc) | 복제 안 함 |
| 관련 코드 | 없음 | 위와 같음 | - | - |

- 공식 코드가 없으므로 `04_claim_verification.md`(주장 C1–Cn)로 작성합니다. `04_code_verification.md`는 만들지 않습니다.
- 저장소 복제(`repo/`)는 하지 않았습니다. 확인한 GitHub 자료는 검색 응답뿐입니다.
- 연구실 계정 확인은 GitHub API 응답으로 했고, API 한도에 걸린 뒤 `gh` 명령으로 다시 확인했습니다.
