# 02. 선행연구와 관련 저장소

이 문서는 ACTUS의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 프로젝트 페이지가 있는지를 정리합니다. 번호는 논문 참고문헌 번호(`paper/paper.txt`의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| ACTUS의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 세계모델 계획 (MPC, 잠재 예측) | Hafner 외 [16, 17], Ha & Schmidhuber [15], V-JEPA 2 [1], Sobal 외 [36], DINO-WM [48] | 계획 루프의 배경. ACTUS는 이 계획기를 바꾸지 않고 읽기만 합니다 |
| 세계모델 학습 (JEPA, 픽셀에서) | LeWM [25] (Maes 외, arXiv:2603.19312), stable-worldmodel [24] | 실험 세계모델과 환경 도구 |
| 반사실 시뮬레이션 | Bear 외 [7], Buesing 외 [10], Pearl [32] | "동결 모델을 행동의 반사실 시뮬레이터로 쓰기"의 착안점 |
| 리군 행동 변환 | Solà 외 [37] (micro Lie theory) | 작동기 변환 T(θ) = exp(Σθ_k G_k)의 틀 |
| 통계적 국소 접근 (고장 검출) | Benveniste 외 [9], Basseville [4], Basseville & Nikiforov [5] | 점수 검정, 효율 점수 (δ의 국소 최적 검정) |
| Rao 점수 검정, Le Cam 점근 | Rao 1948 [34], Le Cam [21], Lehmann & Romano [22] | 검출 법칙 (명제 2)과 국소 최적성 |
| 혁신 기반 진단 | Mehra & Peschon [26], Willsky [46], Kalman [18] | 잔차 에너지 기준선과 혁신 χ² 기준선 |
| 변화점 GLR, 지연 하한 | Lorden [23], Lai [19] | 스트리밍 검출과 지연 식 (10) |
| 가우스-뉴턴 추정, 피셔 점수 | Wedderburn [45], Nocedal & Wright [30], Bell & Cathey [8] | 추정기 (식 11) |
| 혼합 효과 분산 성분 | Laird & Ware [20] | 에피소드와 세션 효과 모형 (식 9) |
| 테스트 시 적응 기준선 | AdaJEPA [43], OpTTA [27], JEPA-TTT [47], Sandwich-Residuals [39], ALRA [38] | 비교 대상. 가중치나 잔차 모듈을 갱신 |
| 테스트 시 학습 원류 | Sun 외 [40], Tent [42], Nagabandi 외 [28] | 배경 |
| 세계모델 이상 탐지 | Domberg & Schildbach [14] | 가까운 대안: 예측 오차 기반 이상 탐지 |
| CEM 계획 | Rubinstein [35] | 계획기 |
| 강건 계획 (기준선) | Chua 외 [12] (PETS) | 불확실성 있는 앙상블 계획 |
| 환경 | PushT [11] (Diffusion Policy), TwoRoom [36], OGBench-Cube [31] | 평가 과제 |

**가까운 대안과의 차이.** 잔차 에너지(예측 오차 모니터)는 같은 잔차를 쓰지만 모형 오차와 이동을 구별하지 못합니다(명제 1). 테스트 시 적응 방법은 가중치를 갱신하므로 작동기 오차를 예측기에 흡수할 위험이 있습니다(논문 §5.3의 해석). ACTUS는 예측기를 고정하고, 이동의 K차원 점수만 읽습니다. 이 점이 차별점입니다.

## 2. 코드와 저장소 (GitHub 및 웹 표)

검색한 범위: arXiv abs·HTML 페이지(링크 목록), GitHub 저장소 검색(제목, 저자 소속 조직 `OSUPCVLab`의 저장소 목록, 키워드 ACTUS, actuation shift, Self-Correcting Visual Planning, Uncertain Is My Action), 저자 프로젝트 페이지 후보(`osupcvlab.github.io`, 조직 페이지). 2026-11-22 기준입니다.

| 저장소·페이지 | 관계 | 라이선스 | 최근 push | 스타 | 상태 |
|---|---|---|---|---|---|
| **공식 코드** | 없음 (검색 범위에서 찾지 못함) | — | — | — | `04_claim_verification.md`로 대조 |
| **프로젝트 페이지** | 없음 (arXiv 링크에 없음, `osupcvlab.github.io`에 항목 없음) | — | — | — | — |
| [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | LeWM 공식 코드 (ACTUS가 쓰는 세계모델, [25]) | MIT | 2026-05-26 | 4654 | 복제하지 않음. ACTUS 구현에 필요할 때 읽기 |
| [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) (LeWM README가 가리킴) | 환경과 계획 도구 ([24]). 메타데이터는 `lucas-maes/stable-worldmodel`(스타 1, 2025-10-17 push, MIT)로 확인. galilai 저장소 자체는 API 확인 안 함 | MIT (확인한 사본) | 2025-10-17 (사본) | 1 (사본) | 설치하지 않음 |
| arXiv:2611.23620 저자 페이지, 소스(TeX) | arXiv 소스 링크(`/src/2611.23620`)는 있으나 내용을 받지 않음 | — | — | — | — |

- **코드를 복제하지 않았습니다.** 공식 ACTUS 코드가 없어서 "정적 읽기"할 대상이 없습니다. LeWM과 DINO-WM은 세계모델 자체의 코드이므로 이 분석의 범위 밖입니다. 구현 가능성 판단에 필요한 LeWM README의 사양만 확인했습니다(`04` §4).
- **논문의 그림 라이선스:** `README.md`의 라이선스 항 참조.

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- 명제 2의 검출 법칙은 Rao 검정과 Le Cam 점근 이론에 기대므로, 그 법칙 자체는 교과서 결과입니다. 새로운 것은 프레임(동결 세계모델의 점수)과 실험입니다.
- 검출 AUROC가 식 (8)을 따른다는 주장은 정확 비중심 χ²와 대략 맞지만 정확하지 않습니다(`03` V9). 논문 본문은 "≈"로 쓰므로 이것이 주장 자체를 무너뜨리지는 않습니다.
- 정보 기반 지연(식 10)은 Lai [19]의 점근 하한을 쓰므로 실제 지연 예측은 점근 근사입니다. 실험에서 순위 상관 0.95 이상이라고 주장하지만 그림 값을 확인하지 못했습니다.
