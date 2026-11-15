# 02. 선행연구와 관련 자료

## 1. 계보 한눈에 보기

| 흐름 | 대표 연구 (논문 참고문헌 번호) | FlashVLA와의 관계 |
|---|---|---|
| 가벼운 백본, 토큰 가지치기, 압축 | TinyVLA [15], SmolVLA [9], Evo-1 [16], QVLA [10], EfficientVLA [11], VLA-Cache [19] | 순전파 한 번을 싸게 하는 방향. FlashVLA는 **패스 수**를 바꾸므로 직교한다고 주장(§2) |
| 디노이징 스텝 축소 | Consistency models [23], Shortcut models [24] | 재학습과 품질 대 스텝 수 절충이 필요. FlashVLA는 증류 없이 같은 스텝 수를 씀 |
| 비동기 추론, 미래 상태 조건화 | VLASH [6], Training-time action conditioning (RTC 후속) [7], StreamingVLA [8], RTC [25] | 비동기의 오래된 관측 문제를 **사후 보정**으로 풂. FlashVLA는 구조로 풂 |
| 실시간 가속 시스템 | Realtime-VLA [26], FASTER [22] | 시스템 최적화 기준선. FASTER는 반응 지연 비교 기준 |
| 스트리밍 청크 확산 (영상) | Diffusion Forcing [12], StreamDiT [13], MAGI-1 [14] | 방법의 직접 출처. 영상에서 쓰던 "노이즈 수준이 다른 버퍼 + 인과 마스크"를 행동 청크로 옮김 |
| 행동 헤드 구조 | FAST [21], FiLM [27] | FiLM은 타임스텝 조건화에 씀. FAST는 관련 토큰화 기법 |

- 저자가 말한 "최초의 VLA 스트리밍 정식화"(§2, 마지막 단락)는 본문 검색으로 확인할 수 없었습니다. 이 주장은 저자 서술로만 적습니다.

## 2. 가까운 대안 (논문 비교 대상과 같은 축)

- **VLASH [6]**: 미래 상태를 조건으로 넣는 비동기 방법. 논문 Tab. 1에서 d=1 1.15×, 성공률 +0.3점, d=4에서 −3.8점.
- **StreamingVLA [8]**: 행동 흐름 매칭과 적응형 조기 관측. Tab. 1에서 1.70×, −2.0점.
- **RTC / 실시간 청크 [25], 학습 시 조건화 [7]**: 실로봇 비교군(RTC). 논문 §5 평균 80.0%.
- **Realtime-VLA [26]**: 추론 지연 기준선(Tab. 2). 4090 2뷰 29.2 ms, FlashVLA 26.7 ms.
- **FASTER [22]**: 반응 시간 기준선(Fig. 4).

## 3. 구성 요소별 출처

| 요소 | 출처 | 비고 |
|---|---|---|
| 흐름 매칭 정식화 식 (1) | 흐름 매칭 일반 (논문은 인용 없이 씀) | 식 (1)의 $\tau$ 방향은 $\tau=1$이 노이즈 |
| 스태거드 노이즈 버퍼, 식 (2) | 영상 스트리밍 확산 [12–14] | Beta(1.5, 1.0) 샘플링은 논문 고유 설정 |
| 청크 단위 인과 마스크 | 영상 Diffusion Forcing [12]의 인과 마스크 | 토큰 단위가 아니라 청크 단위 |
| FiLM 타임스텝 조건화 | FiLM [27] | 부록 A.4 |
| 다중 버퍼 공동 학습 식 (3) | 논문 고유 | 공유 관측, 격리 마스크 |
| 기반 모델 π0.5 | π0.5 [5] | 상위 기준 모델 |
| 벤치마크 | LIBERO [28], RoboTwin 2.0 [29], LingBot-VLA 과제 분류 [4] | RoboTwin 길이 분류는 LingBot-VLA 방식 |
| 실로봇 원격조작 | Gello [30] | 시연 수집 도구 |

## 5. GitHub, 프로젝트 페이지, 관련 저장소

확인 날짜 2026-11-15. 별 수와 마지막 커밋은 GitHub 저장소 페이지(공개 HTML)에서 읽었습니다. GitHub API는 쓰지 않았습니다.

| 자료 | 링크 | 라이선스 | 별 | 마지막 커밋 | 비고 |
|---|---|---|---|---|---|
| FlashVLA 공식 저장소 | https://github.com/z-lab/flashvla | Apache-2.0 | 61 | 2026-10-24 (`8539f1c`, main) | 75 커밋, 포크 5. 얕은 클론을 `repo/`에 받아 정적으로만 읽음 |
| 프로젝트 페이지 | https://z-lab.ai/projects/flashvla/ | - | - | - | 저장소 README에 링크. 이번 분석에서 열지 않음 |
| 체크포인트 모음 | https://huggingface.co/collections/z-lab/flashvla | 미확인 | - | - | 다운로드하지 않음 |
| VLASH | https://github.com/mit-han-lab/vlash | 미확인 | 미확인 | 미확인 | 저장소 README의 감사 항목. 이번 분석에서 조회하지 않음 |
| LeRobot | https://github.com/huggingface/lerobot | 미확인 | 미확인 | 미확인 | 학습과 평가 기반. 조회하지 않음 |

- 공식 저장소의 `feat/lerobot-parity` 브랜치와 PR 1, 2가 있습니다(`git ls-remote`). 이 분석은 `main`(`8539f1c`)만 봤습니다.
- 로컬 클론 커밋: `8539f1cdfcf878e3a93c39f3d6c885c19ede6af8` ("Align LIBERO evaluation and latency benchmark defaults", 2026-10-24).

## 6. 이 분석이 주장하지 않는 것

- 선행연구 계보가 완전하다고 말하지 않습니다. 논문 참고문헌 30개 중 주요 비교 대상만 정리했습니다.
- 다른 기준선의 수치를 독립 재현하지 않았습니다. 논문이 보고한 값만 옮겼습니다.
