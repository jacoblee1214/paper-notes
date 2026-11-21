# 02. 선행연구, 구성 요소의 출처, 공식 코드와 자료

## 1. 선행 계보

GR-1은 세 흐름이 만나는 자리에 있습니다.

**(a) 언어 조건 시각 조작.** RT-1(Brohan 2022)은 합성곱과 Transformer로 행동을 직접 냅니다. CLIPort와 PerAct(Shridhar 2022, 2023)는 CLIP으로 언어와 시각을 잇고, Hiveformer(Guhur 2023)는 여러 시점과 기록을 한 Transformer로 묶습니다. MCIL과 HULC(Lynch·Mees 2020, 2022)는 언어 없는 play 데이터를 함께 써서 잠재 계획을 학습합니다. RT-2(Brohan 2023)는 웹 시각-언어 데이터를 함께 학습합니다. 이 계열은 대부분 키포인트와 모션 플래너, 또는 잠재 계획을 씁니다. GR-1은 행동을 연속으로 내는 단일 GPT 스타일 모델이라는 점이 다릅니다(§2.1).

**(b) 시퀀스 모델링.** Decision Transformer(Chen 2021), VIMA(Jiang 2022), GATO(Reed 2022), RoboCat(Bousmalis 2023)이 있습니다. RoboCat은 행동과 미래 이미지를 함께 예측하지만, 비디오 사전학습이 없고 목표 이미지 조건이라는 점에서 GR-1과 다르다고 논문이 밝힙니다(§2.2).

**(c) 로봇 사전학습과 비디오 예측 정책.** R3M(Nair 2022, Ego4D로 학습), MVP(Xiao 2022), VIP(Ma 2022), RPT(Radosavovic 2023), PLEX(Thomas 2023), SMART(Sun 2023) 등 표현 학습 계열이 있습니다. 비디오 예측을 정책에 쓰는 흐름은 UniPi(Du 2023)와 VIPER(Escontrela 2023)가 있습니다. VPT(Baker 2022)는 역동역학 모델로 비디오에 행동을 붙입니다. 논문은 VPT와 VIPER의 비디오가 과제와 같은 도메인이라 "도메인 내", GR-1의 Ego4D는 "도메인 외"라고 구분합니다(§2.3).

**후속 계보 (논문 밖, 본문에서 확인한 것만).** GR-2(Cheang 2024, 웹 규모 지식을 쓴 생성형 시각-언어-행동 모델)는 본 분석 저장소의 다른 논문 분석에서 GR-1의 후속으로 인용됩니다. 여기서 GR-2 원문은 확인하지 않았습니다.

## 2. 구성 요소별 출처와 GR-1의 기여

| 구성 요소 | 출처 (논문 절) | GR-1에서의 역할 | GR-1의 새로운 점 |
|---|---|---|---|
| 언어 인코더 CLIP | Radford 2021 (§3.2.1) | 동결 | 매 시점 반복 토큰 |
| 영상 인코더 MAE ViT | He 2022 (§3.2.1) | 동결 | 패치 토큰을 Perceiver로 줄임 |
| Perceiver Resampler | Jaegle 2021 (§3.2.1) | 패치 토큰 압축 | (기존 구성 요소) |
| 인과 GPT | Radford 2018 (§3.2) | 메인 백본 | 시점 임베딩 공유, [ACT]·[OBS] 마스크 |
| 비디오 예측 디코더 | He 2022 (MAE 디코더 방식, §3.2.3) | 미래 패치 복원 | 디코더와 마스크 토큰을 행동 예측과 함께 학습 |
| 시각 데이터 | Ego4D, Grauman 2022 (§3.3) | 사전학습 데이터 | 로봇 아닌 사람 영상 |
| 평가 벤치마크 | CALVIN, Mees 2022 (§4.1) | 장기 다중 과제 | (기존 벤치마크) |
| 기준선 | RT-1, HULC, MCIL, MT-R3M (§4.1) | 비교 | MT-R3M은 같은 Ego4D R3M을 같은 GPT 헤드에 붙인 것 |

## 4. 공식 코드와 자료 표

| 항목 | 주소 | 공식 여부 | 라이선스 | 별 수 | 마지막 push | 상태 |
|---|---|---|---|---|---|---|
| 공식 코드 | https://github.com/bytedance/GR-1 | 공식 (논문 프로젝트 페이지의 Code 링크, 저장소 README가 논문과 ICLR 2024를 인용) | Apache-2.0 (LICENSE.txt, GitHub API) | 309 | 2024-04-22 (커밋 4a2e956) | 얕은 clone을 `repo/`에 받아 정적으로만 읽음. 평가 코드만 있음 |
| 같은 이름의 저장소 | https://github.com/GR1-Manipulation/GR-1 | 비공식 (README 한 장, "GR-1 code is available in bytedance/GR-1" 안내) | 없음 | 45 | 2024-04-19 | 안내만 확인. clone 안 함 |
| 프로젝트 페이지 | https://GR1-Manipulation.github.io | 저자 페이지 (arXiv 댓글과 abs 페이지 링크) | 표시 없음 (템플릿은 nerfies 기반) | - | - | 저작권·라이선스 표시 없음. BibTeX는 `@misc`, arXiv 인용 |
| 체크포인트 ABCD→D | https://lf-robot-opensource.bytetos.com/obj/lab-robot-public/gr1_code_release/snapshot_ABCD.pt | 공식 README 링크 | 명시 없음 | - | - | 다운로드 안 함. HEAD 요청으로 크기만 확인 (약 197 MB) |
| 체크포인트 ABC→D | 같은 경로의 snapshot_ABC.pt | 공식 README 링크 | 명시 없음 | - | - | 다운로드 안 함. 약 197 MB |
| MAE ViT-B 가중치 | https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth | 외부 (MAE 저장소) | MAE 저장소 기준 | - | - | 다운로드 안 함. 약 343 MB |
| 학술 평가 벤치마크 | CALVIN, https://github.com/mees/calvin | 외부 (환경과 데이터) | (외부) | - | - | 평가 환경 설치 필요. 여기서는 설치하지 않음 |

**왜 공식 코드는 평가 전용인가.** `bytedance/GR-1`에는 `evaluate_calvin.py`, `evaluation/`, `models/`, `utils/`, `logs/configs.json`, `install.sh`만 있습니다. 사전학습 스크립트, 데이터 로더, 손실 함수, 학습 루프는 없습니다(`04` D1–D4). 실로봇 코드도 없습니다.

## 5. 열린 질문

1. 같은 저자 팀이 GR-2에서 같은 비디오 사전학습 방식을 어떻게 바꿨는지는 GR-2 원문을 봐야 합니다.
2. 사람 영상(Ego4D)과 로봇 영상의 사전학습 효과를 나누는 대조 실험은 공개 자료에 없습니다(`01` §6.4).
3. 공식 코드 저장소가 평가만 공개한 이유는 공개 자료에서 확인할 수 없습니다.
