# 02. 선행연구와 구성 요소 출처

## 1. 계보

**행동 조건 영상 세계모델 (정책 평가·데이터 엔진 쪽).**
IRASim [5] (프레임 단위 행동 조건, 궤적 → 영상) → EnerVerse-AC (EVAC) [2] → Ctrl-World [1] (SVD 백본, 외부 2대와 손목 1대 다시점 공동 예측, 이력 프레임 + 자세 조건 메모리, DROID 학습) → Interactive World Simulator [6] (헤드 시점, 일관성 모델, 실시간 폐루프) → Cosmos Predict 2.5 [7] (단일 시점 기준선). 이 계보의 용도는 정책 평가와 정책 개선용 합성 데이터이고, Mem-World는 같은 용도를 유지하면서 이력 선택(메모리 읽기)만 바꿉니다.

**장기 영상 생성의 기억.**
- 암묵적 문맥 조건: WorldMem [16] (자세·FOV 겹침으로 검색), Context-as-Memory [15] (FOV 겹침), WorldPlay [32].
- 명시적 3D: ViewCrafter [29], Uni3C [30], GEN3C [31] (3D 기반 모델로 재구성한 장면을 새 시점으로 렌더링해 조건으로 사용. 재구성 품질과 계산량에 의존한다고 논문이 서술).
- VMem [17]: 서펠 인덱스 뷰 메모리. 서펠마다 그것을 관측한 과거 프레임 인덱스를 저장하고, 미래 카메라 자세에서 서펠을 렌더링해 보이는 서펠이 많이 겹치는 과거 프레임을 검색합니다. **Mem-World의 직접 출발점**입니다(§1 "inspired by VMem").

## 2. Mem-World의 위치와 구성 요소별 출처

| 구성 요소 | 출처 | Mem-World의 변경 (논문 서술) |
|---|---|---|
| 서펠 인덱스 메모리, 렌더링 기반 검색 | VMem [17] | 정적 → 시간 인식. 서펠에 시간 집합 t_k와 조작 표지 m_k를 추가 |
| 점 지도 추정 | Depth Anything 3 [18] | 첫 프레임의 세 카메라 시점과 이후 손목 프레임에 적용 (변형·해상도 미기재) |
| 미래 카메라 자세 | 순기구학 + 고정 손목 변환 | 항법 모델의 카메라 자세 입력 대신 행동에서 계산 (§3.2) |
| 조작 물체 표지 | Qwen3 [33] + RoboEngine [34] 분할 | 새로 추가한 과제 관련 가중 |
| 타임스텝별 렌더링과 시간 감쇠 | 새 제안 (식 2) | VMem은 한꺼번에 렌더링 |
| NMS | VMem에도 있음(카메라 자세 거리 기준) | 기준·임계값 미기재 |
| 월드모델 본체 | Ctrl-World [1] | 시간 어텐션 층만 11K 궤적으로 재학습 (§4) |
| 평가 | PSNR [35], SSIM [36], LPIPS [37], SAM3 [38] + DINOv2 [39] | Obj. Con.은 새 지표 |
| 정책 | π0, π0.5 | 실로봇 5과제에서 후학습 |

**VMem과의 공개 코드 대조 (참고).** VMem의 공개 저장소(MIT)를 정적으로 읽어 Mem-World가 "새로 한 것"의 범위를 확인했습니다. 자세한 내용은 `04` §3.

## 4. GitHub와 자료 표 (확인 2026-11-24)

| 자료 | 링크 | 라이선스 | 별 / 마지막 push | 관계 |
|---|---|---|---|---|
| **Mem-World 공식 코드** | 찾지 못함 | - | - | arXiv 초록 페이지, Hugging Face 논문 페이지(연결된 모델·데이터셋 없음), GitHub 저장소 검색("Mem-World", "W-VMem"), 웹 검색 모두 없음 |
| 프로젝트 페이지 | 찾지 못함 | - | - | 위와 같음 |
| Ctrl-World | https://github.com/Robert-gyj/Ctrl-World | MIT | 별 586 / 2026-04-08 | 월드모델 본체와 기준선 |
| VMem | https://github.com/runjiali-rl/vmem | MIT | 별 452 / 2025-07-25 (커밋 `39291e4`) | 서펠 메모리 원본. 정적으로만 읽음 |
| Depth Anything 3 | https://github.com/ByteDance-Seed/Depth-Anything-3 | Apache-2.0 | 별 6731 / 2026-07-27 | 점 지도 추정기 |
| SAM 3 | https://github.com/facebookresearch/sam3 | 별도 SAM 라이선스 (API는 NOASSERTION) | 별 12397 / 2026-11-17 | 물체 일관성 지표용 분할 |
| Cosmos Predict 2.5 | https://github.com/nvidia-cosmos/cosmos-predict2.5 | Apache-2.0 | 별 1406 / 2026-06-08 | 단일 시점 기준선 |
| openpi (π0, π0.5) | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 14745 / 2026-08-24 | 평가 대상 정책 |
| DROID 정책 학습 | https://github.com/droid-dataset/droid_policy_learning | MIT | 별 307 / 2025-04-22 | 학습 데이터 계열 |

## 5. 가까운 대안과 비교

- **자세·FOV 겹침 검색 (WorldMem, Context-as-Memory).** 카메라 자세만으로 겹침을 추정. 가림과 동적 물체를 반영하지 못하는 점이 Mem-World의 비판 대상입니다. 그러나 표 2의 비교 대상은 "stride"와 "short-term"뿐이고 자세 근접 검색은 없습니다 (`04` C3).
- **Ctrl-World의 이력 + 자세 조건.** 이력 프레임을 일정 간격으로 넣고 자세를 프레임별 교차 주의로 주입(Ctrl-World §4.1). 표 2의 "Stride"는 이 가운데 간격 표본 부분입니다. 논문 §1이 말하는 "관절 자세 유사도로 가중"과 표 2의 비교 방법이 같은지는 본문에서 확인되지 않습니다.
- **명시적 3D 렌더링 (GEN3C 등).** 정확한 기하를 쓰나 재구성에 의존하고 계산량이 큽니다. Mem-World는 렌더링으로 프레임을 직접 합성하지 않고 렌더링을 검색 점수 계산에만 씁니다.
- **누락된 비교.** 같은 점 지도 + 서펠 메모리 틀에서 VMem 원형(정적, 한꺼번에 렌더링)을 그대로 쓴 기준선이 없습니다. 새 요소 각각의 기여를 가를 수 있는 가장 직접적인 비교입니다.
