# 02. 선행연구, 코드, 구성 요소별 출처

## 1. 선행연구 계보 (논문의 §2와 참고문헌 기준)

VISTA는 세 갈래를 합칩니다. (a) 동변 시각운동 정책, (b) 시각-촉각 모방학습, (c) 촉각 동변성. 아래 번호는 논문 참고문헌 번호입니다.

### 1.1 동변 조작 정책 (논문 §2.1)

| 갈래 | 대표 | VISTA와의 관계 |
|---|---|---|
| 점군·3D 기하 기반 | Neural Descriptor Fields [19], SE(3)-Transformers [20], Vector Neurons [9], EquiBot [21], ET-SEED [22], Canonical Policy [23] | 깊이, 보정, 점군 추출이 필요하다는 점이 약점으로 꼽힙니다(§2.1). VISTA는 이를 쓰지 않습니다 |
| RGB·손목 카메라 기반 | Fourier Transporter [24], **Equivariant Diffusion Policy (EquiDiff)** [25], **Image-to-Sphere Policy (ISP)** [14], Image to Sphere [26], Image-to-Icosahedral [40] | VISTA는 ISP의 "센서 국소 구면 표현 + 엔드이펙터 자세로 회전 보정" 틀을 그대로 가져오고 촉각을 더합니다. EquiDiff+Tac이 기준선입니다 |
| 일반 이론 | Group CNN [16], Steerable CNN [18], Cohen 외 [7], Wang 외 [6, 39] | 등변 ResNet(escnn)과 유한군 가중치 공유의 근거 |

### 1.2 시각-촉각 모방학습과 표현 (§2.2)

- 정책: 3D-ViTac [30], Reactive Diffusion Policy [3], MimicTouch [1], Tian 외 [32], Hogan 외 [33].
- 표현·벤치마크: VITAL [2] (시각-촉각 사전학습, ACT 계열), UniVTAC [28] (시뮬 데이터 생성, 벤치마크, 촉각 사전학습 인코더), ManiFeel [29] (시각-촉각 정책 벤치마크), Binding Touch [27], Touch and Go [31].
- 논문의 진단: 이 방법들은 촉각을 CNN이나 사전학습 인코더로 인코딩해 시각·고유감각과 이어 붙이고, 물체·손·작업 프레임 변화에서 시각, 촉각, 로봇 상태, 6-DoF 행동이 같이 변환되어야 한다는 제약을 두지 않습니다(§2.2).

### 1.3 촉각 동변성과 기하 접촉 정책 (§2.3)

| 방법 | 동변성의 위치 | VISTA와의 차이 (논문 주장) |
|---|---|---|
| EquiTac [34] | 손 안 물체 회전용. 촉각 법선 지도를 SO(2) 벡터로 다루어 각도 잔차 보정 | 지역(손끝 프레임) 대칭. VISTA는 작업 공간 전체 정책 |
| 기하 임피던스 SE(3) 정책 [35] | 몸체 프레임 오차와 렌치 표현 + 임피던스 제어 | 제어 이론 관점. VISTA는 종단 확산 정책 |
| EquiContact [36] | 전역 기준 프레임 추정 + 국소 순응 정책의 계층 | 같음 |

논문은 "VISTA는 이들을 보완한다"고 씁니다. 직접 비교 실험은 없습니다(EquiTac, EquiContact 코드와 수치 비교 없음).

## 2. 공식 코드와 관련 저장소

확인일 2026-11-23. 스타와 푸시 시각은 `gh api`로 읽은 값입니다.

| 저장소 | 관계 | 라이선스 | 스타 | 마지막 푸시 | 커밋 | 상태 |
|---|---|---|---|---|---|---|
| [Kenn3o3/Vista](https://github.com/Kenn3o3/Vista) | **공식 구현** (시뮬 학습, 평가, 기준선, 데이터 수집) | 루트 Apache-2.0. 단 ISP에서 가져온 부분은 학술·교육 한정 라이선스(`policy/ISP/LICENSE`, `policy/VISTA/LICENSE`) | 3 | 2026-10-05 | `a0381fb1f95498dda9b006344bbbe923d42b5ccc` (2026-10-05 16:03Z, "Link the Hugging Face release of the paper dataset.") | 얕은 복제 127 MB, 정적 읽기만, 실행·설치 안 함 |
| [Kenn3o3/vt_franka](https://github.com/Kenn3o3/vt_franka) | 실로봇 수집·배포 (컨트롤러 PC, 작업 PC) | 없음 (API `license: null`, `THIRD_PARTY_NOTICES.md`만 있음) | 3 | 2026-09-15 | 읽은 파일만 `gh api`로 확인 | 복제하지 않음, 설정 파일만 읽음 |
| [vista-paper.github.io](https://vista-paper.github.io/) | 프로젝트 페이지 | - | - | - | - | 내용 확인 (결과 표, 영상) |
| [KLHWONG/MIDFOV_EXPERIMENT](https://huggingface.co/datasets/KLHWONG/MIDFOV_EXPERIMENT) (Hugging Face, ModelScope 거울) | 시뮬 시연 데이터 | 태그상 apache-2.0 | - | 2026-10-05 | - | **124.8 GB, 파일 1611개**. 받지 않음 |
| [univtac/UniVTAC](https://github.com/univtac/UniVTAC) | 벤치마크 원본 (과제 8개) | Apache-2.0 | 197 | 2026-09-07 | - | 참조만 |
| [pointW/equidiff](https://github.com/pointW/equidiff) | EquiDiff 원본 | MIT | 143 | 2025-02-13 | - | 참조만 |
| [purdue-mars/manifeel](https://github.com/purdue-mars/manifeel) | ManiFeel 원본 | MIT | 68 | 2026-04-25 | - | 참조만 |
| ISP 원본 | 선행 방법 (Hu 외, NeurIPS 2025) | 학술 한정 라이선스 문구는 Vista 저장소에 사본이 있음 | - | - | - | GitHub 검색에서 별도 공식 저장소를 찾지 못함 (미확인) |

- 공개 체크포인트는 찾지 못했습니다(README에 링크 없음). 학습은 데이터를 받아 직접 해야 합니다.
- 실로봇 데이터와 실로봇 체크포인트는 어느 저장소에도 올라와 있지 않습니다.

## 4. 구성 요소별 출처와 가까운 대안

| 구성 요소 | 출처 | VISTA에서의 변화 |
|---|---|---|
| 확산 정책 (DDPM, 노이즈 예측, 행동 청크) | Diffusion Policy [42], DDPM [43] | 같음 (식 5) |
| 등변 U-Net, 군 위 가중치 공유 | EquiDiff [25] | 군을 SO(2)(C8)에 더해 SO(3)(I60)로 확장해 과제별로 고름 |
| 이미지 → 구 올림, 조화 투영, 엔드이펙터 보정 | ISP [14], Image to Sphere [26] | 시각 쪽은 그대로. 반구 HEALPix 격자, 별도 학습 촉각 투영(A.2)이 추가 |
| 6D 회전 표현 | Zhou 외 [37] | 같음 (처음 두 축을 벡터 특징으로) |
| 등변 ResNet (escnn) | Wang 외 [39] | 시각과 촉각 모두에 같은 구조 |
| 교차 주의 | Vaswani 외 [41] | 질의 = 시각 방향, 키·값 = 촉각 토큰. 이 논문의 새 부분 |
| 촉각 센서 | GelSight Mini (실로봇), 시뮬 GelSight 계열(TacEx/UniVTAC) | 촉각 RGB 이미지를 그대로 입력 |

**가까운 대안.** (i) EquiDiff+Tac(촉각을 자명 표현으로 연결, 논문의 기준선). (ii) 촉각을 구 밖에서 연결하는 ISP+Concat-Tac, Early Fusion, global-ℓ0 (논문의 절제 변형). (iii) 촉각 전용 지역 동변성(EquiTac, 논문은 직접 비교하지 않음). (iv) 비동변 촉각 정책(ManiFeel, VITAL, UniVTAC, RDP).

**새로움의 범위.** 논문이 새롭게 주장하는 부분은 "촉각 구면 토큰 + 시각 방향 질의 교차 주의"입니다. 동변 확산, 구면 올림, 엔드이펙터 보정은 선행 연구의 구성 요소입니다. 동변성 자체는 융합 위치(회전 보정 앞)와 센서 국소 불변 가정에서 나오므로, 새 융합의 효과는 표현력 쪽에서 평가해야 합니다 (`01` §2.4, `03` §6).
