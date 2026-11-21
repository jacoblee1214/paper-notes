# 02. 선행연구와 공식 코드 (LASR, arXiv:2611.17775)

분석일 2026-11-21. 출처 표기는 논문 참고문헌 번호 [n]을 씁니다(`paper/paper.txt`의 참고문헌 목록 기준).

## 1. 선행연구 계보

| 계보 | 논문 | LASR와의 관계 |
|---|---|---|
| 행동 없는 사전학습 | Seo 외 2022 [39] (RL with action-free pre-training), Schmidt & Jiang 2024 [38] (LAPO), Bruce 외 2024 [4] (Genie) | 잠재 행동이 제어에 쓰인다는 초기 근거. LASR은 "쓸모"를 평가 기준으로 바꿈 |
| 로봇용 잠재 행동 | Ye 외 2025 [44] (LAPA), Bu 외 2025 [5] (UniVLA), Losey 외 2022 [32], Chen 외 2025 [8] (villa-X), Chen 외 2025 [9] (Moto) | 평가 대상 중 LAPA, UniVLA. 성공률 중심 평가 |
| 잠재 행동 표현의 수정 | Zhang 외 2025 [45] (What do latent action models learn?), Nikulin 외 2025 [33] (distractors), Lachapelle 2025 [26] (identifiability), Lee 외 2026 [27] (why latent actions fail), Bu 외 2026 [6] (LAOF), Klepach 외 2026 [25] (object-centric), Jiang 외 2026 [23] (OLAF-World), Wei 외 2026 [43] (causally debiased LAM) | LASR이 직접 겨냥하는 교란(지름길, 정보 누출)을 다룹니다 |
| 세계모델 잠재 평가 | Vafa 외 2024 [41] (implicit world model), Nilaksh 외 2026 [34] (reconstruction or semantics?), Hafner 외 2019 [17] (PlaNet 계열) | 구조 검사와 표현 평가. 예측 정확도가 구조 오류를 가린다는 주장 |
| 잠재 행동 세계모델 (최신) | Garrido 외 2026 [13] (ICML 2026, latent action WM in the wild), Gao 외 2025 [12] (AdaWorld), Liang 외 2025 [29] (CLAM), Zhang 외 2026 [50] (DiLA), Lee 외 2026 [28] (MVP-LAM), Chen 외 2026 [7] (LaWAM), Gao 외 2025 [12] | 평가 대상 다섯 표현 중 DiLA, MVP-LAM, LaWAM |
| 조건부 프로빙 | Hewitt & Liang 2019 [20] (control tasks), Hewitt 외 2021 [21] (conditional probing), Alain & Bengio 2016 [1] (linear probes) | 상태 기준선 (조건부 정보)과 비행동 대안 (control task)의 근거 |
| 시뮬 상수 | Tao 외 2025 [40] (ManiSkill3), Zhao 외 2023 [51] (ALOHA), Khazatsky 외 2024 [24] (DROID), Liu 외 2023 [31] (LIBERO) | 데이터와 시뮬 출처 |
| 생성 품질 지표 | Wang 외 2004 [42] (SSIM), Zhang 외 2018 [48] (LPIPS), Heusel 외 2017 [19] (FID), Oquab 외 2024 [35] (DINOv2), Huang 외 2024 [22] (VBench), Ghosh 외 2023 [15] (GenEval) | 세계모델 예측 지표(표 3c) 근거 |
| 정보·의사결정 이론 | Blackwell 1953 [3] (실험 비교), Gneiting & Raftery 2007 [16] (엄격 적합 점수 규칙), Banerjee 외 2005 [2] (Bregman 위험), Gelada 외 2019 [14] (DeepMDP) | 정리 4, 따름정리 5, 식 (5)의 근거 |

**차별점 (§2, 2쪽):** 선행 평가는 성공률, 복원 손실, 행동 복원 정확도, 계획 성공 중 하나로 잠재를 평가하며, 이 중 어느 것도 상태와 비행동 대안을 넘어서는 기여를 분리하지 못한다고 주장합니다. LASR은 "상태만 쓰는 경로", "잠재 경로", "검증으로 고른 비행동 경로"를 같은 예측기로 비교하는 설계를 제안합니다.

**근접 대안:** 조건부 프로빙 [21]은 한 기준선에 대한 개선을 봅니다. LASR은 기준선 두 개(상태, 비행동 대안)의 최소와 비교하고, 같은 예측기 계열을 강제합니다 (§2, 3쪽). 이 차이는 식 (1)의 분모입니다.

## 2. 공식 코드, 프로젝트 페이지, 관련 저장소

### 2.1 검색 결과 (2026-11-21)

| 항목 | 상태 | 확인 방법 |
|---|---|---|
| 논문 공식 코드 저장소 | **찾지 못함** | GitHub 저장소 검색: "LASR latent action support ratio", "GeoLASR", "WorstLASR", "2611.17775", "Beyond Prediction Gain" 모두 0건. 저자 계정 이름(Changsong Lei, Yong-Jin Liu, Yeying Fan) 사용자 검색 0건 |
| 프로젝트 페이지 | **찾지 못함** | arXiv abs와 HTML(v1) 외부 링크에 없음. 웹 검색 결과에 없음. alphaXiv 페이지에도 코드 링크 없음 |
| 동반 코드 산출물 | 논문 부록 I에 "companion code artifact"가 있다고 적었지만 공개 위치는 없음 | `paper/paper.txt` 부록 I |
| Hugging Face 논문 페이지 | 404 | `huggingface.co/papers/2611.17775` |

**판정:** 공식 코드가 없습니다. 그래서 `repo/` 복제는 하지 않았고 `analysis/04_claim_verification.md`로 대신합니다. 공식 목록 제도상 "학회 표기 없음"입니다.

### 2.2 기준선 표현의 공식 저장소 (논문이 재사용한 공개 체크포인트, 복제하지 않음)

이 표는 표현 추출에 쓰인 공개 저장소의 메타데이터만 확인한 것입니다. 실행, 설치, 복제를 하지 않았습니다.

| 방법 | 저장소 | 라이선스 (GitHub API) | 별 | 마지막 push | 비고 |
|---|---|---|---|---|---|
| UniVLA [5] | OpenDriveLab/UniVLA | Apache-2.0 | 1,145 | 2025-11-19 | RSS 2025 |
| LAPA [44] | LatentActionPretraining/LAPA | MIT | 588 | 2025-01-22 | ICLR 2025 |
| DiLA [50] | senngadaisuki/disentangled-latent-action-world-models | 없음 (license null) | 22 | 2026-09-15 | README는 "ICML 2026 공식 구현"이라고 적음. 논문 참고문헌 [50]은 arXiv:2605.15725로만 적혀 있어 대응은 미확인 |
| MVP-LAM [28] | jmSNU/mvp_lam | 없음 | 7 | 2026-05-28 | ICML 2026 |
| LaWAM [7] | 찾지 못함 | — | — | — | 논문의 arXiv 2606.15768 주소를 직접 확인하지 않음 |

- 세 가지 저장소의 라이선스가 없거나 불명입니다(DiLA, MVP-LAM). 이 폴더의 공개 산출물에는 코드를 넣지 않았습니다.
- LaWAM 저장소는 검색으로 찾지 못했습니다. 논문 부록 D의 "공개 체크포인트" 설명과 맞춰 확인해야 합니다.

## 4. 구성 요소별 출처 요약

| 구성 | 근거 |
|---|---|
| 세 경로 비교, 분모의 최소 | 저자 설계 (§3.2–3.3). 상태 기준선은 [21]의 조건부 프로빙에서 옴 |
| 비행동 대안 선택 (진행도, 에피소드 정체성) | 저자 설계 (부록 C). 진행도 선택은 10쌍 모두 |
| 기하평균과 최댓값 집계 | 저자 설계 (식 (2)–(3)), 명제 6 |
| 정보 이론 해석 | [3], [16], [2] 및 저자 증명 (부록 B) |
| 잔차 어댑터 | 저자 설계 (§3.7, 부록 F). 학습 신호는 LASR 유사 대리 |
| 세계모델 예측 지표 | [42] SSIM, [48] LPIPS, [35] DINOv2 |
| 시뮬과 데이터 | [51] ALOHA, [40] ManiSkill3, [31] LIBERO |
