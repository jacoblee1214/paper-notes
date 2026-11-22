# 02. 선행연구와 관련 저장소

번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다. 분석일 2026-11-23.

## 1. 선행연구 계보 (구성 요소별 출처)

| UniIntervene의 요소 | 출처 | 관계 |
|---|---|---|
| 사람 개입 + 오프폴리시 RL로 실로봇 학습 | HIL-SERL [27](Luo 외, Science Robotics 2025), SERL [28], ConRFT [9], ReBoot [18] | 이 논문의 기반 학습기입니다. 제안 방법은 HiL-SERL에 끼워 넣는 모듈이고, 정책 갱신은 HiL-SERL이 합니다(알고리즘 1) |
| 개입 데이터를 지도 신호로 재사용 | HG-DAgger [20], TRANSIC [19], Mandlekar 원격조작 [29], Compliant Residual DAgger [43], Juicer [2] | 논문이 "개입 데이터를 더 잘 쓰는 줄기"로 분류하고 자기와 구분합니다(§1, §2). 같은 저자 팀의 E2HiL [11](엔트로피 기반 표본 선택, RA-L 2026)도 이 줄기입니다 |
| 언제 도움을 청할지 모델링 | ThriftyDAgger [16], 런타임 모니터링 [25], When to ask for help [41], ARMADA [44] | **가장 가까운 대안.** 트리거를 학습하지만 안전 위반이나 실패 사건에 묶여 있다고 논문이 구분합니다(§2). ARMADA(실패 자동 감지 + 사람 공유 제어)는 본문에서 비교하지 않았습니다 |
| 실패 예측 + 복구 정책 | Failure-aware RL, FA-RL [24](arXiv:2601.07821), Recovery RL [39], 안전 비평가 [4, 35] | 비교 기준선입니다(표 1). 부록 H가 FA-RL을 ResNet-18 실패 분류기 + EfficientNet-B0 복구 회귀기로 다시 구현했다고 적습니다. 공식 구현을 썼는지는 불명입니다 |
| 리셋 없는 학습 | Leave No Trace [13], Gupta 외 [14], 하위 목표 커리큘럼 [34, 49], 리셋 게임 [42] | 관련 연구에서 "되돌릴 수 있는가"를 보는 줄기. 이 논문은 "가치가 오르는가"를 본다고 차별화합니다(§2) |
| 미래 잠재 예측으로 가치 평가 | V-JEPA2 [3] (얼어 있는 목표 인코더) | 목표 인코더를 JEPA류로 쓰는 것은 월드모델 계열과 같습니다. 단 예측 대상이 한 단계 뒤 풀링 벡터 하나라서 롤아웃 모델은 아닙니다 |
| 진행 기반 가치 라벨 | NTTG(time-to-go 정규화), 쌍둥이 비평가, CQL [23] | 식 9–13. CQL은 "작은 CQL 항"이라고 부르지만 코드의 항은 다른 형태입니다(`04` D4) |
| 가치 변화로 정체 감지 | 이 논문의 시간 가치 위험(식 5) | 새 정의입니다. 변화율 임계 ε에 대한 누적 부족분을 거리(1−V)로 가중합니다 |
| 기억 검색 + 목표 조건부 복구 | 검색 기반 정책(일반) | 검색한 구간을 BC 목표로 쓰는 구조는 단순 검색 모방에 가깝습니다(`04` D9) |
| 행동 토크나이저 | FAST [31] | 복구 행동 덩어리의 토큰화(식 8) |
| 정책 기반 | π0.5 [6] (SFT 기준선), OpenVLA·π0 [7, 22] | SFT 기준선과 정책 비교 맥락 |
| 대규모 온라인 사후 학습 시스템 | SOP [30], Learning while deploying [40], IG-RFT [37], GigaBrain-0.5M* [38] | 같은 문제 맥락에서 인용(HiL-RL 확장). 개입 비용 자체를 줄이는 방향은 아닙니다 |

## 2. GitHub와 공개 자료

| 저장소 | 관계 | 라이선스 | 별 | 마지막 푸시 | 상태 |
|---|---|---|---|---|---|
| [Denghaoyuan123/UniIntervene](https://github.com/Denghaoyuan123/UniIntervene) | **공식 구현** (오프라인 학습 파이프라인만) | MIT (`LICENSE`, "Copyright (c) 2026 UniIntervene authors") | 19 | 2026-09-19 | 얕은 복제 `repo/`, 커밋 `40f461f97573383c9cc8fdba43df56f9eb5bbea1`, 정적 읽기만 |
| [Denghaoyuan123/UniIntervene-project](https://github.com/Denghaoyuan123/UniIntervene-project) | 프로젝트 페이지 소스 (GitHub Pages) | 표기 없음 (페이지 하단은 CC BY-SA 4.0 주장) | 1 | 2026-09-19 | 페이지 HTML만 확인 |
| [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl) | 기반 HiL-RL 프레임워크 | Apache-2.0 | 1,599 | 2025-10-27 | 읽지 않음. 논문의 로봇 쪽 통합 코드는 이 저장소 밖 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | π0.5 SFT 기준선, FAST 토크나이저 출처 | Apache-2.0 | 14,732 | 2026-08-24 | 읽지 않음 |
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | 미래 잠재 목표 인코더 | MIT | 4,912 | 2026-10-29 | 읽지 않음 |
| [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) | 백본(Qwen3-VL-2B-Instruct) | Apache-2.0 | 20,244 | 2026-01-30 | 읽지 않음 |
| FA-RL (arXiv:2601.07821) 저장소 | 비교 기준선 | 찾지 못함 | – | – | `gh search`로 두 번 찾았으나 없음 |
| E2HiL(같은 저자 팀, RA-L 2026) 저장소 | 이전 연구 | 찾지 못함 | – | – | `gh` 검색 결과 없음(저자 계정 목록에도 없음) |

- 공식 저장소에는 로봇 배포, HIL-SERL 통합, 평가 코드, 데이터, 체크포인트, 기억 은행이 **없습니다**(README: "outside this package"). 가중치와 데이터는 받지 않았습니다.
- 공식 저장소 규모: 파일 약 40개, 948 KB, Python 약 1.5만 줄 (`04` §1).
- 프로젝트 페이지에는 영상(YouTube)과 요약 수치(+8.6%, −57%, 88%, 14.6%)가 있고 새로운 정량 결과는 없습니다.

## 4. 가까운 대안과 읽을 순서

1. HIL-SERL [27]: 이 논문의 기준선이자 기반. IR 정의와 개입 스텝 집계 방식을 먼저 확인해야 합니다.
2. ARMADA [44], ThriftyDAgger [16]: "자동 실패 감지로 사람 호출을 줄인다"는 같은 목표. UniIntervene과의 직접 비교는 없습니다.
3. FA-RL [24]: 표 1의 유일한 자동 개입 기준선. 재구현 여부와 코드가 열쇠입니다.
4. V-JEPA2 [3], FAST [31]: 구성 요소 이해.
