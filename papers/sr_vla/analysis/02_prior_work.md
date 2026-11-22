# 02. 선행연구와 관련 자료: SR-VLA

## 1. 공식 코드와 프로젝트 페이지

SR-VLA에는 공식 코드가 확인되지 않았습니다. 논문에 링크가 없고, GitHub 저장소 검색(키워드 "SR-VLA", "Stall-Reduced VLA", "predicted-input lookahead", "selective KV cache repair")은 관련 없는 결과만 냈습니다. 웹 검색도 논문 제목으로 프로젝트 페이지를 찾지 못했습니다. 확인 날짜 2026-11-22입니다. 따라서 `04_claim_verification.md`로 주장을 검증했고, 코드 정적 대조(`04_code_verification.md`)는 하지 않았습니다.

백본과 비교 대상의 저장소는 메타데이터만 확인했습니다(GitHub API, 복제 없음, 2026-11-22).

| 저장소 | 관계 | 라이선스 (GitHub 표기) | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 백본 구현으로 알려진 저장소 (논문 [15]의 공식 여부는 본문에 없어 확인 못 함) | Apache-2.0 | 14,727 | 2026-08-24 | 복제 안 함 |
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft) | OpenVLA-OFT (논문 [5]) 저장소 | MIT | 1,450 | 2025-09-09 | 복제 안 함 |
| [mit-han-lab/vlash](https://github.com/mit-han-lab/vlash) | VLASH (논문 [18], "Real-Time VLAs via Future-state-aware Asynchronous Inference", 가장 가까운 비교 대상) | Apache-2.0 | 525 | 2026-04-22 | 복제 안 함 |
| [siyuhsu/vla-cache](https://github.com/siyuhsu/vla-cache) | VLA-Cache (논문 [22]) 저장소로 보이나 공식 여부 미확인 | Apache-2.0 | 107 | 2026-02-27 | 복제 안 함 |
| SR-VLA 공식 저장소 | 찾지 못함 | — | — | — | — |
| SR-VLA 프로젝트 페이지 | 찾지 못함 | — | — | — | — |

- 저장소 두 개(openpi, openvla-oft)는 실제 체크포인트와 추론 코드를 담고 있어 SR-VLA 재구현의 기반이 될 수 있습니다. 다만 SR-VLA의 예측기와 수리 로직은 들어 있지 않습니다.
- VLASH 저장소는 같은 "미래 상태 인지 비동기 추론" 계열이라 SR-VLA와 비교 실험을 하려면 가장 먼저 필요한 기준선입니다. 이 맥에서 실행하는 것은 `04` §4에서 판단합니다.

## 2. 계보: 무엇을 이어받았고 무엇이 새로운가

SR-VLA는 세 흐름이 만납니다(논문 §I, §II).

1. **동기 추론의 정지 문제.** 청크 단위 실행에서는 다음 추론이 끝날 때까지 로봇이 멈춥니다(논문 [4], [5]). 이 문제를 줄이는 방법은 세 가지입니다. 추론 비용을 줄이는 쪽(양자화, 가지치기, 적응 계산, [6]–[11]), 호출 빈도를 줄이는 쪽(청크, [4], [5]), 병렬·추측 디코딩 쪽([12]–[14])입니다. 이들은 동기 의존을 없애지 못한다고 논문은 봅니다.
2. **비동기 추론.** 현재 청크가 실행되는 동안 다음 추론을 시작합니다(SmolVLA [16], Real-time action chunking [17], VLASH [18], Jetson-PI [19]). 결과가 실제 입력과 어긋날 수 있다는 문제가 남습니다. VLASH는 미래 상태를 예측해 이 문제를 줄입니다.
3. **상태 재사용.** VLA-Cache [22]는 인접한 실제 관측에서 나온 상태를 재사용합니다. 추측 디코딩 [23], [24]은 조건 입력이 이미 있을 때 계산을 미리 합니다.

**SR-VLA가 더하는 것 (논문 §II-B와 contribution).**

- 다음 경계의 *입력 자체*를 예측합니다(이미지와 고유감각, 식 4). 미래 상태 표현만 예측하는 [19]와 다릅니다.
- 예측 입력으로 만든 KV 캐시와 행동을 실제 입력 도착 후 **패치 단위로 검증하고 일부만 고칩니다**. 전부 버리는 것(동기 재계산)과 전부 쓰는 것(무조건 재사용) 사이의 중간입니다. VLA-Cache의 캐시 재사용과 다른 점은 "예측된 다음 경계" 대상이라는 것입니다.

**논문이 다루지 않은 것.** 예측기가 틀렸을 때 수리 빈도가 올라가 이득이 줄어드는 상황은 목표 이동 실험(§IV-D)에서만 봅니다. 여러 다음 경계를 한꺼번에 예측하는 경우는 다루지 않습니다(논문이 "targets only the next decision boundary"라고 명시).

