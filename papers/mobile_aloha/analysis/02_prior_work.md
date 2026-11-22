# 02. 선행연구, 구성 요소별 출처, 코드와 프로젝트 페이지

## 1. 계보

- **ALOHA** (Zhao, Kumar, Levine, Finn, RSS 2023, 논문 [104]): 저가 양팔 텔레오퍼레이션 장비. 리더 팔의 관절 각도를 그대로 실제 팔에 보냅니다. Mobile ALOHA는 이 장비 (팔 두 쌍)를 그대로 가져와 이동 기지를 얹은 것입니다 (§3).
- **ACT** (같은 논문 [104]): 청크 예측 CVAE 트랜스포머. Mobile ALOHA의 기본 정책입니다. 이 저장소의 `act_aloha/` 분석 대상입니다.
- **Diffusion Policy** (Chi et al., RSS 2023, [18]): 행동 확산 정책. 표 2에서 비교 대상입니다.
- **VINN** (논문 [63]): 시각 표현 + 최근접 이웃 검색 기반 모방 학습. 논문은 BYOL [37] 표현에 청크와 고유감각 가중치를 더한 "VINN + Chunking"을 씁니다.
- **공동 학습 (co-training) 근거**: Open X-Embodiment / RT-X ([20])와 RoboCat ([11])가 서로 다른 로봇 데이터를 함께 학습하는 흐름을 보였습니다. 논문은 이 흐름을 **이동 조작**으로 옮겼습니다 (§2 "Imitation Learning for Robotics", §4). 논문 스스로 "정적 양팔 데이터로 이동 기지 정책을 개선한 첫 보고"라고 씁니다 (§2 마지막 문단, "to our knowledge").
- **이동 조작 선행**: 모델 기반 (DARPA Robotics Challenge [56], 모바일 매니퓰레이션 [9, 17, 33, 52, 93]), 강화 학습 기반 ([38, 48, 58, 94, 101]), 전신 제어 ([36, 42, 99]), 텔레오퍼레이션 장비 ([14, 15, 19, 22, 23, 26, 32, 45, 53, 65, 66, 72, 75, 87]). 가격 비교는 PR2 · TIAGo 20만 달러 이상, PR1 haptic, Hello Robot Stretch, Fetch (§1, §2).

## 2. 구성 요소별 출처와 가까운 대안

| 구성 요소 | 출처 (논문 참고문헌 번호) | 가까운 대안 (논문에 나온 것) | 논문의 차이점 |
|---|---|---|---|
| 저가 양팔 장비 | ALOHA [104] | PR1 [93] (햅틱 두 개, 발판), TIAGo + 모션캡처 [5], PR2, Stretch [2], Fetch [1] | 기지 + 줄로 허리 묶기 (전신 텔레오퍼레이션). 장비 약 3.2만 달러 |
| 정책 (ACT) | [104] | Diffusion Policy [18], VINN [63] | 기지 속도 지연을 청크 시프트로 보정 (§6), 정책 구조는 그대로 |
| 공동 학습 | RT-X [20], RoboCat [11] (단일 팔 조작), 항법 공동 학습 [79] | 사전학습 후 미세조정 (표 4) | 정적 양팔 데이터 825 에피소드를 50:50 혼합 |
| 행동 공간 | 14 관절 + 기지 2 (선속도·각속도) | 엔드이펙터 공간 | 기지 행동 2차원을 관절 행동에 이어 붙임 (§1 "requiring almost no change in implementation") |
| 시각 표현 | ResNet-18 [40], BYOL [37] (VINN) | — | 정적 데이터는 정면 카메라를 제외 (3 카메라 맞춤) |
| 확산 스케줄 | DDIM [85] | — | 학습 50 스텝, 추론 10 스텝 (표 7) |

## 4. 코드와 프로젝트 페이지 (2026-11-22 조회)

| 구분 | 저장소 / 페이지 | 라이선스 | 스타 | 마지막 push | 커밋 (복제 여부) | 역할 |
|---|---|---|---|---|---|---|
| 공식 하드웨어·수집 코드 | [MarkFzp/mobile-aloha](https://github.com/MarkFzp/mobile-aloha) | MIT | 4,481 | 2024-06-22 | `0e40324` (2024-01-03, 복제: `repo/mobile-aloha/`, 6.2 MB) | ROS 1 기반 4팔 + 기지 + 카메라 3대 수집·재생 (`aloha_scripts/`, `config/`) |
| 공식 학습 코드 | [MarkFzp/act-plus-plus](https://github.com/MarkFzp/act-plus-plus) | MIT | 3,677 | 2024-05-15 | `26bab07` (2024-01-10 (PST), 복제: `repo/act-plus-plus/`, 2.6 MB) | ACT·DP·VINN 학습, 공동 학습 설정 (`aloha_scripts/constants.py`는 `mobile-aloha`에 있음) |
| 프로젝트 페이지 | [mobile-aloha.github.io](https://mobile-aloha.github.io) | — | — | — | — | 논문 링크, 영상, 데이터셋 링크 (Google Drive 폴더 · 문서 링크) |
| 기반 저장소 | [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | MIT | (이 분석에서 조회하지 않음) | — | `act_aloha`의 742c753 | ACT 원본 (act-plus-plus의 상위) |

- 두 공식 저장소는 **얕은 복제 (`--depth 1`)**로 받아 **정적으로만** 읽었습니다. 설치, import, 실행은 하지 않았습니다. 복제본은 `repo/`에 있고 `.gitignore`로 제외합니다.
- 데이터셋: 프로젝트 페이지의 Google Drive 폴더 링크는 확인만 했고 다운로드하지 않았습니다. 크기는 확인하지 않았습니다.
- 두 저장소의 최신 커밋은 2024년 1월입니다. 논문 CoRL 판 이후 갱신이 있었는지는 이 조회로 알 수 없습니다.

## 5. 이 논문을 읽을 때 기억할 점

- 공동 학습의 "정적 데이터"는 논문에서 **825 에피소드**로 적혀 있지만, 코드의 설정은 `aloha_compressed_dataset`이라는 이름만 씁니다. 그 데이터의 크기와 구성은 저장소에서 확인되지 않습니다.
- 공식 코드의 기본 설정과 논문 표 5–9가 완전히 같지는 않습니다. 04에서 항목별로 적었습니다.
