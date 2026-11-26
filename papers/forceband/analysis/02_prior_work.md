# 02. 선행연구와 관련 저장소

번호는 논문 참고문헌 번호(`paper/paper.txt` References)입니다. 선행연구의 내용은 논문 본문이 적은 범위(§1, §5, §6)에서만 정리하고, 논문을 직접 읽지 않은 것은 "논문 기술" 또는 제목 수준으로 표시합니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| ForceBand의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 사람 시연을 정책 학습 자료로 사용 | Human-to-robot imitation in the wild [1], Point Policy [2], P3-PO [3], HOP [4], HumanEgo [5], Dexterity from smart lenses [6], Chain-of-Modality [7] | 동작과 외형은 얻지만 접촉 힘이 빠진다는 문제의 출발점(§1). ForceBand는 이 계열에 힘 채널을 더합니다 |
| 영상 기반 접촉·상호작용 추정 | 손-물체 접촉 [8, 9], 3D 접촉 지도·주석 자료 [10, 11], 의사 접촉 라벨 [12], 영상에서 촉각 예측 [13] | 비교 대상(VISOR-HOS [26], FEEL [27])이 속한 계열. 논문은 가림과 모호함 때문에 힘 크기는 추정이 어렵다고 봅니다(§1, §6) |
| 직접 촉각·힘 센싱 장갑 | OpenTouch [14], OSMO [15], Feel the Force [16] | 힘 정답은 더 신뢰할 수 있지만 시각 가림과 수집 부담이 있다는 점이 동기(§1). 표 2 기준선 설계는 [16]을 따랐다고 적습니다(§5.3) |
| sEMG 기반 힘·제스처 | 단일 채널 EMG 분해 [17], Wrist2Finger [18], DexEMG [19], emg2pose [28], 제스처 인식 [29–32], 채널 선택 [33, 34], EMG + 관성 [18, 35] | 힘 추정의 직접 선행. 논문은 기존 연구가 이진 동작이나 단일·균등 전극에 머문다고 서술합니다(§6) |
| 전극 배치 | 단일 배측 [18], 굴곡·신전 두 채널 [22], 손목 4채널 [23], 8채널 균등 암밴드 [24, 25] | 표 1의 비교 대상 |
| 스펙트로그램 + 사전학습 비전 인코더 | DINOv3 [20] | STFT를 이미지처럼 다뤄 사전학습 비전 인코더를 재사용(§3). 공식 코드는 DINOv2 ViT-S/14로 출하(`04` D2) |
| 정책 | 조건부 흐름 매칭 [21] | 행동에 힘 성분을 더해 동작과 힘을 함께 예측(§4) |
| 사람 영상 → 로봇 관측 | Aria [41], Grounding DINO [42], SAM 2 [43], CoTracker [44], Orient-Anything [45], LaMa [46] | 부록 C의 변환 파이프라인. 모두 외부 도구입니다 |
| 사용자 보정 아이디어 | 상용 장갑의 지침 자세 보정 [40], 대규모 sEMG의 사용자 간 일반화 [28, 39] | 한계 절에서 보정 센서 제거 방향으로 인용 |

**가까운 대안과의 차이.** 같은 질문(사람 시연에 힘 라벨 붙이기)에 대한 가까운 대안은 촉각 장갑(OSMO [15], OpenTouch [14])과 영상 기반 힘 추정(FEEL [27])입니다. ForceBand는 손끝을 비워 두는 대신 손목 sEMG와 사용자별 15분 보정(손끝 힘 센서 필요)을 요구합니다. 논문 비교는 FEEL(영상)까지이고, 촉각 장갑이나 손끝 센서를 단 정책과의 직접 비교는 없습니다.

## 2. 코드와 저장소 (GitHub 표)

2026-11-26에 `gh api`와 Hugging Face API 메타데이터로 확인했습니다.

| 저장소 | 관계와 범위 | 라이선스 · 별 · 마지막 push | 대조 기준과 상태 |
|---|---|---|---|
| [Bottle101/ForceBand](https://github.com/Bottle101/ForceBand) | **공식 코드**. EMG2Force 모델(평가, 보정 미세조정, 전체 학습 스크립트), 검증·데모 자료 일부, 체크포인트. 정책 학습, 로봇 제어, 영상 변환 코드는 없음 | Apache-2.0 · 96 · 2026-10-28 (생성 2026-06-26, 포크 12, 이슈 4건 열림) | `4e02176289acb3b70121ead16f3d1272a0b0d67d` (2026-10-28, 얕은 복제). 정적 읽기만, 실행·설치·import 안 함 |
| [forceband-emg/forceband-emg.github.io](https://github.com/forceband-emg/forceband-emg.github.io) | 프로젝트 페이지 소스 | 사이트 하단에 CC BY-SA 4.0 표기 · 0 · 2026-11-18 | 페이지 HTML만 읽음 |
| [Hugging Face boootaooo/forceband-emg2force](https://huggingface.co/datasets/boootaooo/forceband-emg2force) | 사전학습 데이터셋. README 기준 8.6 h(영상 있는 부분 6.4 h), 53개 기록 | Apache-2.0 · 내려받기 870회 · 수정 2026-10-21 | 파일 목록 메타데이터만 확인(hdf5 76개 0.556 GB, mp4 47개 7.68 GB). 내려받지 않음 |
| 체크포인트 `checkpoints/emg2force_prevforce_dinov2.ckpt` | 공개 가중치(DINOv2 백본) | Apache-2.0 | Git LFS 포인터의 크기 116,115,839 byte. 받지 않음 |
| [jessicayin/osmo_tactile_glove](https://github.com/jessicayin/osmo_tactile_glove) | 논문 [15] OSMO 관련 저장소 | 표기 없음 · 91 · 2026-10-26 | 확인만 |
| [OpenTouch-MIT/opentouch](https://github.com/OpenTouch-MIT/opentouch) | 논문 [14] OpenTouch 관련 저장소 | 표기 없음 · 84 · 2026-03-18 | 확인만 |
| [facebookresearch/emg2pose](https://github.com/facebookresearch/emg2pose) | 논문 [28] emg2pose (sEMG 손 자세 벤치마크) | LICENSE 파일 있음(GitHub 분류는 NOASSERTION) · 293 · 2024-12-30 | 확인만 |

- GitHub에서 같은 이름의 `ClangWU/forceband`(2024, "force feedback wrist band")와 `stelios191/emg2force`("forearm HD-sEMG에서 5손가락 힘 스트리밍 디코더")는 이 논문과 별개의 저장소입니다(이름이 겹칠 뿐입니다).
- 저장소 README의 "Roadmap"은 10-21 커밋에서 삭제됐고, 열린 이슈는 사용자 ID·기록-사용자 대응표 공개 요청(#6, 교차 사용자 평가용), 전체 데이터 공개 시점(#3), 카메라-EMG 동기화와 DIY 매뉴얼(#2), BOM(#1)입니다.
- 논문 §3은 데이터셋을 "공개한다(collect and release)"고 쓰고, 프로젝트 페이지 데이터 섹션에는 "Public release planned" 문구가 남아 있습니다. 같은 페이지의 버튼은 Hugging Face 데이터셋을 가리킵니다.

## 4. 이 논문의 주장이 기대는 선행 결과

- sEMG와 손가락 힘의 관계가 사용자와 전극 위치에 따라 달라진다는 점(§1)은 논문이 인용한 문헌([17, 18, 19])의 일반적 진술입니다. 사용자별 보정이 필요하다는 서술은 공식 코드 README에서도 수치로 나옵니다(`04` §3).
- "근육 기반 배치가 균등 배치보다 낫다"는 표 1의 30분 비교 한 쌍이 직접 근거이고, 인용된 [33, 34]의 결과는 각 논문의 설정(제스처 분류, 팔꿈치 각도)이 달라 같은 비교는 아닙니다(논문 기술 수준).
