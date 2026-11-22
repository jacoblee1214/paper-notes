# AnyTouch: 선행 연구, 가까운 대안, 공식 저장소

## 1. 계보

AnyTouch는 세 흐름이 만나는 자리에 있습니다.

1. **마스크 기반 촉각 사전학습.** MAE (He 외, 2022)와 VideoMAE (Tong 외, 2022)의 마스크 재구성을 촉각 이미지와 비디오에 적용합니다. 촉각 MAE의 선행 사례로 Cao 외 (IROS 2023)와 Higuera 외 (Sparsh, CoRL 2024)가 인용됩니다.
2. **시각·언어·촉각 정렬.** CLIP (Radford 외, 2021)과 OpenCLIP (Cherti 외, 2023)이 백본입니다. 촉각 정렬 계열로 TVL (Fu 외, 2024, Touch-Vision-Language), Touch100k와 TLV-Link (Cheng 외, 2024), OmniBind (Lyu 외, 2024), VIT-LENS-2 (Lei 외, 2024)가 있습니다. 여러 모달을 묶는 방법으로 ImageBind (Girdhar 외, 2023)를 참고합니다.
3. **여러 센서의 공동 학습.** UniTouch (Yang 외, 2024)는 여러 데이터셋의 촉각을 한 모델로 묶었고, AnyTouch가 가장 직접 비교하는 대상입니다. T3 (Zhao 외, 2024)는 약 3M 데이터로 여러 센서를 학습했으며, 사전학습에 다운스트림 과제 라벨을 씁니다. Rodriguez 외 (2024)는 두 센서의 쌍 데이터(32,256쌍)로 교차 센서 생성을 했습니다.

센서 불변 표현의 직접 선행 연구로 Gupta 외 (arXiv:2502.19638, 2025, Sensor-invariant tactile representation)가 v2부터 인용됩니다. AnyTouch 논문의 2절이 이 계보를 정리합니다.

시뮬레이션 촉각 데이터셋: ObjectFolder 1.0과 2.0 (Gao 외, 2022a, 2022b)과 이를 렌더링하는 TACTO (Wang 외, 2022a), Taxim (Si·Yuan, 2022). 실제 데이터셋: Touch and Go (Yang 외, 2022), VisGel (Li 외, 2019), Cloth (Yuan 외, 2018), ObjectFolder Real (Gao 외, 2023), YCB-Slide (Suresh 외, 2023), SSVTP (Kerr 외, 2022), Octopi (Yu 외, 2024), Feel (Calandra 외, 2017).

후속 연구: **AnyTouch 2** (Feng 외, ICLR 2026 Poster, OpenReview 기록 ndilONnABZ 확인). 같은 저자 계열로 보이며, 다른 분석 폴더 README에서는 "동적 촉각 비디오 표현"으로 소개됩니다. 이 분석에서는 AnyTouch 2 논문을 읽지 않았습니다.

## 2. 가까운 대안 (비교 방법 표)

| 방법 | 학습 데이터·구조 | AnyTouch와의 차이 | 논문 내 비교 |
|---|---|---|---|
| UniTouch (Yang 외, 2024) | 여러 촉각 데이터셋, 촉각–비전 쌍 | 텍스트 정렬과 센서 토큰이 없음. 쌍 데이터 조건이 더 까다로움 (AnyTouch는 텍스트 앵커 사용) | 표 2, 표 3 (같은 데이터 짝 AnyTouch와 비교) |
| T3 (Zhao 외, 2024) | 약 3M, 다운스트림 라벨 사용 | 라벨을 사전학습에 쓰므로 정적 인지 비교는 불공정하다고 논문이 스스로 말함 | 표 4 (실로봇), 표 7 (교차 센서 생성)만 |
| TLV-Link (Cheng 외, 2024) | Touch100k, 세 모달 쌍 | 단일 센서 (GelSight) 위주. 논문은 TLV-Link†를 같은 데이터로 재학습 | 표 2 |
| OmniBind (Lyu 외, 2024), VIT-LENS-2 (Lei 외, 2024) | GelSight 계열 단일 센서 | 여러 센서를 쓰지 않음 | 표 2 |
| Rodriguez 외 (2024) | 두 센서 쌍, 32,256쌍, 교차 센서 생성 | 텍스트와 여러 센서가 없음. 논문이 "쌍 데이터의 가치"를 강조하는 데 쓰는 기준 | 본문 서술만 |
| Sparsh (Higuera 외, CoRL 2024) | 자기지도 촉각 표현, 약 460k 장 | 텍스트 정렬이 없음 | 본문 서술 (v2부터 인용) |
| Gupta 외 (2025) | 센서 불변 표현 | 같은 목표. 방법은 다름 | v2부터 인용 |

## 4. GitHub, 프로젝트 페이지, 관련 저장소

| 저장소 | 관계 | 라이선스 | 대조 기준 | 상태 |
|---|---|---|---|---|
| [GeWu-Lab/AnyTouch](https://github.com/GeWu-Lab/AnyTouch) | **공식 구현** (ICLR 2025 README) | MIT (`LICENSE`, 2025 GeWu-Lab) | 커밋 `9c43a1a6eb38d904fd767712eb9dcb2d98b8d56b` (2026-01-13 push), 별 97, 포크 10 | 얕은 복제 `repo/` (정적 읽기만, gitignore). 대조는 `04_code_verification.md` |
| [GeWu-Lab/AnyTouch2](https://github.com/GeWu-Lab/AnyTouch2) | 후속 공식 구현 (ICLR 2026) | MIT (API 확인) | 생성 2026-01-28, 마지막 push 2026-04-17, 별 68 | 복제하지 않음. 이 분석 범위 밖 |
| [gewu-lab.github.io/AnyTouch](https://gewu-lab.github.io/AnyTouch/) | 프로젝트 페이지 | — | arXiv 링크와 TacQuad HF 링크 확인 | 본문 읽음 |
| [xxuan01/TacQuad](https://huggingface.co/datasets/xxuan01/TacQuad) | 데이터셋 (HF) | 카드 license: mit | 마지막 수정 2025-08-28, 다운로드 1,412회, 좋아요 4. 파일: `tacquad.zip.001`–`.007` 이상 분할 압축, `contact_*.csv`, `obj1.0.zip`, `octopi.zip`, `split_3.npy` | 받지 않음 (데이터 크기 확인은 목록만) |
| Google Drive 체크포인트 (README 링크) | 공식 체크포인트 (AnyTouch 전체 모델) | 확인하지 못함 | — | 받지 않음. 크기 미확인 |
| `arxiv.org/abs/2502.12191` | 논문 | CC BY 4.0 (`creativecommons.org/licenses/by/4.0/`) | v1–v3 | 본문 읽음 (`01_paper_analysis.md`) |

**구현 준비 상태 메모**: 공식 저장소는 학습과 선형 탐침 스크립트, 데이터 전처리 코드를 포함합니다. 기준선 (UniTouch, TLV-Link, OmniBind, VIT-LENS-2, T3)과 실로봇 pouring, 교차 센서 생성 (표 7) 코드는 없습니다.

## 5. 조회 메모

- ICLR 2025 채택: OpenReview API (`api2.openreview.net/notes/search`)에서 제목으로 찾아 venue "ICLR 2025 Poster", venueid `ICLR.cc/2025/Conference` 확인. ICLR 가상 포스터 페이지는 찾지 못했습니다.
- 라이선스: arXiv abs HTML에서 "view license" 링크 `creativecommons.org/licenses/by/4.0/` 확인.
- GitHub 메타데이터: GitHub REST API 공개 조회 (인증 없음).
- HF 메타데이터: `huggingface.co/api/datasets/xxuan01/TacQuad` 조회. 데이터 파일은 다운로드하지 않았습니다.
