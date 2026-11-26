# 02. 선행연구, 코드와 저장소, 관련 분석

## 1. 선행 계보 (논문 §1, §2의 인용 기준)

| 줄기 | 대표 | 이 논문과의 관계 |
|---|---|---|
| 지름길 학습 | Geirhos 외 [1], VQA 언어 선행 편향 [3, 25] | "단서를 따라가는" 일반 현상. 이 논문은 지시 인자 단위로 좁힌다 |
| 조합 일반화 | Lake & Baroni [2], Hupkes 외 [11], Keysers 외 [12] | 인자 곱 공간에서 일부만 보고 나머지를 맞히는 평가 틀 |
| 로봇 정책의 지름길 | 데이터 다양성·파편화 [5], 반사실 실패 [4] (arXiv:2602.17659), LIBERO-Plus [7], RoboView-Bias [10] | 모두 집계 성공률 중심. 이 논문은 "어느 인자"를 가르는 진단을 추가 |
| 데이터 스케일링과 커버리지 | Data scaling laws [6], 데이터 품질 [39], 무엇이 중요한가 [37] | 양보다 과제 커버리지가 병목이라는 주장 계열 |
| 환경 인자 조합 데이터 수집 | **Gao 외 [40]** (arXiv:2403.05110) | V 전략의 직접 출처. 환경 인자 격자에서 "열 + 대각" 형태로 수집하고 Hamming 거리로 비용을 센다 |
| 평가 보조로서의 VLM 판정 | Gemini 2.5 [42] | 롤아웃 영상 분류 |
| 정책 백본 | π0 [21], π0.5 [22], OpenVLA-oft [43], GR00T N1.x [44], X-VLA [45], Genie Envisioner [46] | 진단 대상 여섯 개 |
| 시뮬레이션 | ManiSkill [41] | 환경과 모션 플래너 |

저자가 내세우는 새로움(§2): "지시 인자 편향"이라는 실패 양식의 정량화(FDR, FDH)와, 이를 데이터 수집 배분에 연결한 점. 인자별 성능 분해 자체(예: 색, 동사, 크기별 성공률)는 기존 벤치마크에서도 보고되던 것이고, 이 논문의 차별점은 **상관 훈련 후 충돌 지시**로 "어느 인자가 이기는가"를 직접 세는 설계에 있다.

## 2. 코드와 저장소 (2026-11-26 확인)

| 대상 | 관계 | 라이선스 | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| [yqi19/CoRL2026-Scale-Up-Strategically](https://github.com/yqi19/CoRL2026-Scale-Up-Strategically) | **공식 코드** (저장소 설명 "Official Code of Scale Up Strategically", 프로젝트 페이지의 Code 링크) | 저장소 최상위에 라이선스 파일 없음. 포함된 ManiSkill 포크 폴더는 Apache-2.0 | 3 | 2026-10-13 | 분석 커밋 `b5b32418bd070b2ce55f3fcde21d742306331ce0`, 커밋 4개, 크기 약 110 MB(API 보고) |
| [프로젝트 페이지](https://yqi19.github.io/scale-up-strategically/) | 저자 페이지 (CoRL 2026 표기, 논문·코드·데이터·영상 링크) | - | - | - | 표 1·표 3 수치를 그대로 싣고 FDH 목록을 코드로 렌더링 |
| [yqi19/all_factor_maniskill_hf_dataset](https://huggingface.co/datasets/yqi19/all_factor_maniskill_hf_dataset) | 데이터 (프로젝트 페이지의 Data 링크) | 카드에 표기 없음 | - | 2026-05-07 | 저장 용량 약 106 GB(API 메타데이터, 받지 않음) |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0, π0.5 구현. 공식 코드는 커밋 `e6b0441`에 패치 적용 | Apache-2.0 | 14,772 | 2026-08-24 | 공식 코드의 `finetune/pi05` |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | GR00T-N1.7 구현. 공식 코드는 커밋 `4b1dca9`에 패치 적용 | Apache-2.0 | 8,423 | 2026-11-17 | 공식 코드의 `finetune/gr00t` |
| [mani-skill/ManiSkill](https://github.com/mani-skill/ManiSkill) (haosulab에서 이전) | 환경 | Apache-2.0 | 3,552 | 2026-08-04 | 공식 코드는 포크를 포함 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | 데이터 형식 | Apache-2.0 | 29,102 | 2026-11-26 | 변환 스크립트가 LeRobot 형식으로 씀 |

공식 코드의 범위(`04` §1): 데이터 수집(충돌 실험과 전략 수집), 미세조정 레시피(GR00T, π0.5/π0), 롤아웃 클라이언트, VLM 판정기, FDR/FDH 계산. 포함되지 않은 것: 실로봇 코드, Complete 전략, OpenVLA-oft·X-VLA·Genie-Envisioner 평가기(Genie는 "비공개 추론 패키지에 의존해서 뺐다"고 README가 적음), 사람 판정 데이터.

## 4. 가까운 대안과 구성 요소별 출처

| 구성 요소 | 이 논문의 선택 | 출처 / 대안 |
|---|---|---|
| 인자 분해 | 5인자 템플릿 | 템플릿 지시는 합성 벤치마크 전반의 관행. 자연어 변이(LIBERO-Plus 계열 섭동)와 달리 인자 값 통제 |
| 상관 후 충돌 | 대각선 훈련 + 충돌 평가 | 지름길 학습의 고전 설계(스퓨리어스 상관 도입)를 지시 인자에 적용 |
| 판정 | VLM(영상 → 라벨) | 시뮬레이터가 둘 다 알고 있으므로 환경 판정(코드의 `--env-dir`)이 대안. 저장소는 두 방식을 모두 구현하나 논문은 VLM만 보고 |
| 집계 | Copeland 다수결 | 쌍 비교 집계의 표준. 값의 크기를 버리므로 대안은 Bradley–Terry나 FDR 행렬의 직접 비교 |
| 비용 척도 | Hamming 거리 합 f | Gao 외 [40]의 환경 설정 노력 척도를 지시 변경 비용으로 옮김 |
