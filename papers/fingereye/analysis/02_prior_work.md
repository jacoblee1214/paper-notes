# 02. 선행연구, 구성 요소 출처, 관련 자료 (FingerEye)

논문 참고문헌 번호는 arXiv v4 기준입니다.

## 1. 계보

### 1.1 See-Through-Skin(STS)과 시각-촉각 센서

| 계열 | 대표 | FingerEye와의 관계 |
|---|---|---|
| 투명 막 위의 시각+접촉 | FingerVision [8, 18] (Yamaguchi·Atkeson), VisTac [9], Finger-STS [12] | 카메라가 접촉면을 통과해 물체를 봄. FingerEye는 반투명 막 대신 **투명 아크릴 + 주변 고리**로 시야를 비움(DH2) |
| 조명 전환·특수 조명 | StereoTac [14], SpecTac [13], 다중 모드 [15] | 모드 전환·능동 조명이 없다는 점을 차별점으로 내세움(DH1) |
| 마커/키라인 추적 | TacThru류 [19] | AprilTag 배열 + 다중 태그 PnP로 대체(DH3). 비교는 정성 그림 |
| 겔 기반 촉각 | GelSight [25], 9DTact [26], TacTip [27], MiniTac [28] | 접촉 후만 감지. 비교는 실로봇 GelSight 열(표 2) |
| 압전·압저항 | [20–23], BioTac [24] | 언급만 |
| 6축 힘의 마커 기반 대용 | Ouyang·Howe [49], ShapeForce [50] | 감도 분석 방식의 출처(CH3, A.2.2) |

### 1.2 다중 센서 정책 학습

- **플랫 융합:** 토큰을 이어 붙여 하나의 인코더에 넣는 방식(ViTacFormer [38], ACT [41], PolyTouch [42] 등), 또는 행동 쿼리가 섞인 컨텍스트에 어텐션(Diffusion Policy [43], RoboPanoptes [44], UMI [53]). 논문은 이들이 쉬운 전역 신호(손목 영상·고유감각)로 지름길을 만들 수 있다고 봅니다 [46–48].
- **FingerEye의 위치:** 인코더를 모달 그룹 안으로 제한(GroupEnc)하고, 디코더에서 그룹별 교차 어텐션의 잔차를 평균(GroupDec). 관련 아이디어는 FACTR [47](힘 주의 커리큘럼), 정책 합의 [46](multi-modal policy consensus)이며, 구조적으로는 ACT 디코더의 교차 어텐션 분할입니다.
- **표현 사전학습 쪽:** Sparsh [36, 37], ViTaMin [39], Touch in the Wild [40]는 촉각 표현을 따로 학습. FingerEye는 동결 범용 비전 백본(RADIO [54])의 요약 벡터를 쓰고 촉각 전용 사전학습은 없습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문에서의 위치 |
|---|---|---|
| 행동 청킹 + 인코더-디코더 트랜스포머 | ACT [41] (공식 코드에 ACT 블록 복사, Apache-2.0 헤더) | 4.2절, A.3.1 |
| 이미지 백본 | AM-RADIO [54] (코드는 `radio_v2.5-b`를 torch.hub로 로드) | PD3, A.3.1 식 8 |
| 손 | LEAP Hand [51] | 4.1절 |
| 원격조작 | 리더-팔로워(일치 기구학). 리타게팅(AnyTeleop [52])은 예비 시도에서 격차가 커 포기 | 4.1절 |
| 감도 분석 | Ouyang·Howe [49], ShapeForce [50] | A.2.2 |
| 섬세 파지 실험 | SpikeATac [55] | A.2.3 |
| 태그 | tag25h9 계열 AprilTag, pupil_apriltags 검출기 (코드) | A.1.1, A.1.3 |
| 시뮬레이터 | Isaac Lab 2.3.0 / Isaac Sim 5.1.0.0 (공식 Lab 저장소 README) | 4.1절 |

## 4. GitHub와 코드 자료 (확인 2026-11-23)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (FingerEye 모노레포)** | https://github.com/ariszxxu/FingerEye | **없음** (저장소에 LICENSE 파일 없음, GitHub 표시도 없음) | 별 24, 포크 3, 커밋 5, 기본 브랜치 master, 마지막 푸시 2026-06-08, 최신 `2466c11` ("500 default epochs") | 개인 계정. `repo/`에 희소 clone(자산 제외, 1.5 MB)해 정적으로만 읽음. 전체 크기 약 255 MB(Isaac 자산·HDRI·텍스처, 티저 PNG 13.7 MB) |
| 하위 `FingerEyePolicy/` | 위 저장소 | 위와 같음 | - | PyTorch 정책, 전처리, 학습. 데모 데이터 미공개(`zarr_path: your_zarr_path.zarr`) |
| 하위 `FingerEyeRW/` | 위 저장소 | 위와 같음 | - | 실세계 수집·배치·섬세 파지·카메라 설정. xArm7, LEAP, RealSense D435, FingerEye 하드웨어 필요 |
| 하위 `FingerEyeLab/` | 위 저장소 | 위와 같음 | - | Isaac Lab 2.3.0 / Isaac Sim 5.1 디지털 트윈, 시뮬 평가(`scripts/record.py`). NVIDIA GPU 필요 |
| 프로젝트 페이지 | https://nus-lins-lab.github.io/FingerEyeWeb/ | 없음(repo: NUS-LinS-Lab/FingerEyeWeb) | 별 2, 마지막 푸시 2026-10-09 | Nerfies 템플릿. 논문·코드·하드웨어 가이드 링크, "CoRL 2026 Spotlight" 표기, 요약 영상, 관찰 O1–O4 |
| 하드웨어 가이드 | Google Docs 링크(프로젝트 페이지) | 미확인 | - | 열람하지 않음(로그인 문서일 수 있음). 제작·BoM 상세가 있다고 서술 |
| RobotCamCalib | https://github.com/ariszxxu/RobotCamCalib | 없음 | 별 12, 푸시 2026-10-20 | 카메라 보정(실세계 README가 참조). 받지 않음 |
| 포크 | wu467/FingerEye 외 2개 | - | - | 단순 포크로 추정 |
| RADIO (백본) | https://github.com/NVlabs/RADIO | NOASSERTION(라이선스 파일 별도 확인 필요) | 별 약 2,010 | 가중치는 받지 않음. 코드는 `radio_v2.5-b`를 torch.hub로 내려받음 |
| LEAP Hand API | https://github.com/leap-hand/LEAP_Hand_API | NOASSERTION | 별 179 | 하드웨어 드라이버 |
| ACT | https://github.com/tonyzhaozh/act | MIT | 별 2,352 | 정책 블록의 원본 |

공식 저장소에 라이선스가 없으므로 기본 설정상 모든 권리가 저작자에게 있습니다. 이 분석은 정적으로만 읽었고 실행·복사·재배포는 하지 않았습니다.
