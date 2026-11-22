# 02. 선행연구, 가까운 대안, 공식 저장소

## 1. 계보

| 흐름 | 대표 연구 (논문 참고문헌 번호) | 이 논문과의 관계 |
|---|---|---|
| 비디오 확산 기반 WAM | Genie Envisioner [37], Cosmos Policy [29], DreamZero [60] (14B), Motus [5] (8B), LingBot-VA [33] (5.3B), DiT4DiT [44] | 같은 계열. 대부분 2B–14B 규모 |
| 테스트 시 상상 제거 | Fast-WAM [63] | **직접 기반.** 비디오 전문가를 질의당 한 번 돌리고 KV 캐시를 행동 전문가가 읽는 구조를 제안. LoopedWAM은 이 구조를 그대로 가져옴 |
| Fast-WAM 가속 계열 | Light-WAM [35] (1.9B, 동결 비디오 백본), LiLa-WAM [56] (0.5B, 잠재 동역학), LaWAM [11] (2.3B), Efficient-WAM [32] (1B), Faster-WAM [64], AHA-WAM [9] (서로 다른 속도 비동기), Flash-WAM [1] (교사 증류 한 스텝), GigaWorld-Policy-0.5 [51], SelfWAM [46] | 모델 일부를 줄이거나 샘플링을 줄임. LoopedWAM은 **인터페이스는 그대로 두고 깊이를 공유**한다는 점에서 다름 |
| 가중치 공유 깊이 (looped transformer) | Universal Transformer [15], ALBERT [30], looped transformers [22, 48, 57], recurrent depth [20], Relaxed Recursive [2], Mixture-of-Recursions [3] | 깊이 공유 기법의 원조. LoopedWAM은 비디오와 행동 두 전문가를 동시에 루프 |
| 루프 + 엘라스틱 깊이 | ELT [23] (시각 생성, 전체 루프에서 중간 루프 증류), LoopFormer [27] (언어, 스텝 크기에 따른 루프 조건화) | 가장 가까운 선행. ELT는 중간 루프를 증류하지만 행동 전문가와 샘플링 스텝을 같이 다루지 않음 |
| 루프 + 세계모델·정책 | Looped World Models (arXiv:2606.18208, 웹 검색으로 제목만 확인, 본문 미확인), RD-VLA [52] (가중 공유 행동 헤드 반복) | 세계모델 루프와 VLA 헤드 루프. 두 전문가를 함께 루프하는 것은 LoopedWAM의 차별점 |
| 샘플링 스텝 증류 | Shortcut models [19], MeanFlow [21] 등 [1, 19, 21, 47, 50] | 엘라스틱 증류가 스텝 크기 d를 조건으로 쓰는 방식의 원조 |
| 조기 종료 | PonderNet [4], Depth-adaptive [17], LayerSkip [18], AdaComp [24] | 같은 목적(계산량 선택)을 다른 방식으로 푸는 기법 |

## 2. 논문이 직접 비교하는 대안 (표 1–4)

| 방법 | 파라미터 | 로봇 사전학습 | LIBERO (보고값 / 재평가) | RoboTwin 2.0 | 출처 |
|---|---|---|---|---|---|
| LoopedWAM | 0.58B | 없음 | 96.1 | 91.9 | 표 1, 2 |
| Fast-WAM | ∼6B | 없음 | 97.6 보고 / 96.6 재평가 (표 A1: 96.63) | 91.8 | 표 1, 2, A1 |
| Faster-WAM | ∼6.3B | 없음 | 99.0 보고 / 98.7 재평가 | 92.6 | 표 1, 2, A1 |
| Light-WAM | 1.9B | 없음 | 97.2 보고 / 95.8 재평가 | 76.4 | 표 1, 2, A1 |
| LiLa-WAM | 0.5B | 없음 | 97.1 | 89.8 | 표 1, 2 |
| LaWAM | 2.3B | 있음 | 98.6 | 91.2 | 표 1, 2 |
| Motus | 8B | 있음 | 97.7 | 87.8 | 표 1, 2 |
| LingBot-VA | 5.3B | 있음 | 98.5 | 92.2 | 표 1, 2 |
| AHA-WAM | 7.2B | 없음 | — | 92.8 | 표 2 |
| DiT4DiT | — | — | — | — | 표 3: RoboCasa 50.8 |
| StarVLA-GR00T | 4B | — | — | — | 표 3: RoboCasa 47.8 |

비고: 표 1, 2의 "보고값"은 각 논문이 제시한 평균이고, LoopedWAM 평가 프로토콜에 맞춘 재평가는 * 표시. 본문은 Fast-WAM의 재평가값(96.6)과 비교.

## 4. 공식 저장소와 자료 (2026-11-22 확인)

| 항목 | 링크 | 상태 |
|---|---|---|
| 공식 코드 | https://github.com/anhdao69/LoopedWAM | 별 4, 포크 0, 커밋 3 (웹 페이지). 최신 커밋 51aac25 (2026-11-19, "Link the arXiv paper in the README and citation"). 얕은 복제(`repo/`, 커밋 1개)로 정적 읽기 완료. **라이선스: LICENSE 파일은 MIT** (제3자 코드는 원래 라이선스 유지 조항 포함). GitHub API는 NOASSERTION으로 표시 (API 호출은 한도 초과로 실패, 웹 페이지 기준) |
| 프로젝트 페이지 | https://anhdao69.github.io/LoopedWAM | 요약, 그림, 로봇 영상, BibTeX. 학회 표기 없음 |
| 체크포인트 | https://huggingface.co/anhdao69/LoopedWAM | 최근 수정 2026-11-20. 메타데이터 라이선스 **apache-2.0** (코드 MIT와 다름). 파일 6개의 `policy.pt` (LIBERO, RoboTwin 50, RoboCasa-GR1 각각 LoopedWAM·LoopedWAM-E). LIBERO LoopedWAM `policy.pt` = 2,338,586,905 바이트 (≈2.34 GB, 0.58B × FP32 4바이트와 일치). **다운로드하지 않음** |
| 사전학습 donor | Wan-AI/Wan2.1-T2V-1.3B | 코드가 요구. 라이선스는 이 분석에서 확인하지 않음 |
| 벤치마크 | LIBERO, RoboTwin 2.0, RoboCasa-GR1 | 설치·데이터 다운로드 안 함. 데이터 크기: LIBERO 잠재 캐시 약 42 GB, RoboTwin 약 1.1 TB, RoboCasa 약 590 GB (저장소 README 기재) |

### 확인하지 않은 항목
- 학습 데이터셋 `yuanty/LIBERO-fastwam`, `yuanty/robotwin2.0-fastwam` (FastWAM 것, 크기 미확인).
- 학회 채택 여부: 2절 참조.

## 5. 학회 표기 확인

| 확인 경로 | 결과 |
|---|---|
| arXiv abs 메타데이터 (journal-ref, comments) | 없음 |
| 프로젝트 페이지, GitHub README | 학회 표기 없음 |
| `conferences/corl2026/papers.json` (CoRL 2026, 755편, 제목·arXiv 번호 색인) | 제목 "LoopedWAM" 없음, arXiv 2611.25595 없음. "Elastic" 포함 제목은 ELASTIC(2606.31132) 하나뿐으로 다른 논문 |
| `conferences/iros2026/papers.json` (518편) | 일치 없음 |
| 웹 검색 ("LoopedWAM" + 제목) | 허깅페이스 모델 카드 등만 나옴. 학회 채택 언급 없음 |

**판정: 학회 표기 없음 (arXiv).** 제출은 2026-11-19로 최근이고, 채택 결정이 나왔다는 근거가 없음.

## 6. 분석 방식 메모
- 이 절에서 선행연구의 결과 수치는 LoopedWAM 논문 표를 그대로 옮긴 것이고, 각 원 논문을 직접 확인하지 않았음.
