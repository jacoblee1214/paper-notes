# 선행연구와 위치

블로그 참고문헌 번호는 [n]으로 씁니다(`paper/paper.pdf` p.21–22). 블로그가 인용하지 않았지만 가까운 연구는 arXiv 번호를 함께 적습니다.

## 1. 계보: 이 글이 이어 붙이는 세 흐름

| 흐름 | 대표 연구 | 이 글과의 관계 |
|---|---|---|
| LLM으로 텍스트 산출물 최적화 | OPRO [1], DSPy [2], MIPRO [3], GEPA [4], Meta-Harness [5], autoresearch [6] | 저자 중 Khattab이 DSPy·GEPA 계열의 저자입니다. "프롬프트·프로그램을 LLM이 고치듯 가중치도 LLM이 고친다"는 확장이 글의 동기입니다(p.3) |
| 가중치를 사람이 정한 프로그램에서 만들기 | RASP(Weiss et al., 2021), Tracr(Lindner et al., 2023, arXiv:2301.05062, 블로그 p.3 링크), AdderBoard [28] | 실험 5·6은 "원하는 프로그램을 먼저 설계하고 층별로 옮기는" 방식이라 이 흐름에 가장 가깝습니다. 차이는 컴파일러 대신 LLM이 설계와 변환을 둘 다 한다는 점입니다 |
| 다른 네트워크가 가중치를 쓰기 | fast weight programmer(Schlag et al., 2021, arXiv:2102.11174, 블로그 p.3 링크), hypernetwork(Ha et al., 2016, arXiv:1609.09106), p-diff [22], HyperTuning [23], NeuroGen [24] | 이들은 가중치를 쓰는 모델을 **그 일에 맞게 학습**합니다. 이 글은 학습하지 않은 범용 LLM을 그대로 씁니다 |

## 2. 가까운 대안: 같은 결과를 다른 방법으로 낼 수 있는가

이 글의 각 실험에는 LLM 없이도 같은 일을 하는 기존 기법이 있습니다. 블로그는 이들과 직접 비교하지 않았습니다. 이 비교가 "LLM이 무엇을 더하는가"를 가르는 기준입니다.

| 실험 | 기존 기법 | 어떻게 대응되는가 |
|---|---|---|
| 1·2 (얕은 층 읽기) | 첫 층·마지막 층 가중치 시각화(feature visualization 이전의 고전적 방법) | numpy로 접으면 같은 구조가 보입니다(`03` §1). LLM은 텍스트 숫자만 보고 이 구조를 찾아 이름과 연결합니다 |
| 3 (복사 헤드) | Elhage et al. [10]의 QK·OV composition 점수, Olsson et al.(2022, arXiv:2209.11895)의 induction head | 블로그도 Astra가 이 방법을 스스로 다시 찾아냈다고 씁니다(p.8). 같은 계산을 스크립트로 돌리면 사람이 해석할 필요 없이 같은 헤드가 나옵니다 |
| 7 (아첨 탐지) | 두 모델의 activation·attention 비교, 자동 해석 에이전트 MAIA(Shaham et al., 2024, arXiv:2404.14394) | 가중치만으로 실패하고 attention 도구로 성공했으므로, 결과는 activation 기반 해석에 속합니다 |
| 4 (정책 수정) | 손 설계 제어기와 학습 정책의 결합(잔차·전환 제어), 보상 shaping 후 재학습 | 편집은 별도 경로를 배선해 특정 구간만 덮어쓰는 방식입니다(`01` §2.3) |
| 8 (반복 제거) | 방향 하나를 지우는 Arditi et al. [14], activation steering(ITI, Li et al., 2023, arXiv:2306.03341), Representation Engineering(Zou et al., 2023, arXiv:2310.01405), 반복 원인 분석 [13] | Astra가 쓴 방법(반복·정상 문맥 비교로 방향 → `down_proj` 일부 열에서 제거)이 이 계열 그대로입니다. 디코딩 쪽 처방(반복 패널티, 토큰 예산)과의 비교도 없습니다 |
| 5·6 (처음부터 쓰기) | 사람이 쓴 휴리스틱을 Tracr식으로 컴파일, 행동 복제(휴리스틱 → 데이터 → 지도학습) | 실험 6은 데이터 적합을 금지했으므로, 휴리스틱을 손으로 층에 옮기는 것만 허용됩니다 |

경사 없는 최적화 쪽 기준선도 있습니다: SPSA [25], ES [26], MeZO [27]. 실험 6·8의 pass 예산(최대 50번 측정)은 이들이 보통 쓰는 평가 횟수보다 훨씬 적습니다. 그래서 "같은 측정 예산에서 무작위 섭동 탐색보다 LLM 제안이 낫다"는 비교가 있으면 설득력이 커집니다. 실험 8에서 Astra는 자기 검증에 노름을 맞춘 무작위 방향 대조군을 넣었다고 적었습니다(`runs/p50c/final_answer.md`). 이 대조군은 연습 문제 기준이고 블로그 본문에는 나오지 않습니다.

## 3. 해석 연구 쪽의 위치

| 연구 | 무엇을 하는가 | 이 글과의 차이 |
|---|---|---|
| Circuits [15], Elhage et al. [10] | 사람이 도구로 회로를 추적 | 이 글은 같은 분석을 LLM이 자연어 지시로 수행 |
| ROME [16], MEMIT(Meng et al., 2023, arXiv:2210.07229) | 사실 연관을 특정 MLP에서 찾아 rank-1로 편집 | 실험 8의 10 pass 편집(30개 행렬에서 rank-1 제거)과 형태가 비슷 |
| Bills et al. [19] | LLM이 뉴런 하나의 activation 예시를 보고 설명 | activation을 본다는 점이 다름. 이 글의 실험 1–3은 가중치만 봄 |
| Natural Language Autoencoders [17], Weight Oracles [18] | activation이나 가중치를 읽도록 **학습된** 모델 | 이 글은 학습 없이 범용 모델을 그대로 씀. Weight Oracles(arXiv:2610.07334)가 가장 가까운 비교 대상인데, 같은 과제로 비교하지 않았음 |
| WorkspaceBench [21] | 해석 방법을 비교하는 벤치마크 | 저자들이 만들겠다는 "Weight-Space Programming 벤치마크"(p.19)와 목적이 겹침 |

## 4. 이 저장소의 다른 논문과의 관계

- **`gpt6_astra_tactile`:** 같은 GPT-6 Astra를 고정된 촉각 정책으로 쓴 논문입니다. 거기서는 LLM이 **행동을 직접 출력**하고, 여기서는 LLM이 **행동을 내는 신경망의 가중치를 씁니다**. 그 분석에서 지적한 결정당 지연(약 30초) 문제를, 가중치를 한 번 써 두고 작은 네트워크로 실시간 실행하는 방식이 피해 갈 수 있습니다. 실험 4·5가 바로 그 형태입니다.
- **`dust`:** 역전파 없이 forward만으로 Transformer를 사전학습하는 영차 방법입니다. 실험 8의 "gradient 없음, 측정은 있음" 구조와 같은 축에 있습니다. Dust는 무작위 섭동을 수많이 평가하고, 여기서는 LLM이 소수의 구조적 후보를 제안해 50번 안에 고릅니다. 두 글을 겹쳐 보면, 측정 예산 대비 효율을 비교하는 실험을 설계할 수 있습니다.
- **로봇 정책 논문들(`dexscrew`, `flashdexretarget` 등):** 실험 4처럼 PPO 정책이 국소 최적(경사로 앞에서 멈춤)에 빠지는 문제는 이 저장소의 RL 기반 손 조작 논문에서도 반복됩니다. 재학습 대신 정책 가중치를 진단·편집하는 접근은 이들에 바로 옮겨 볼 수 있습니다.

## 5. GitHub 요약 (2026-11-03 확인)

| 저장소 | 무엇인가 | 라이선스 | 확인 |
|---|---|---|---|
| RikiyaT/lm_understand-logs | 실험 1–8의 전체 로그, 평가, 신경망(실험 7 미세조정 모델 17 GB는 제외) | 로그·데이터·신경망 CC BY 4.0, 스크립트 MIT, 실험 5 프롬프트 속 MinAtar 소스 GPL-3.0 | 커밋 `4d09ba6`, `logs_official/`에 clone(367 MB) |
| kenjyoung/MinAtar | 실험 5의 게임 | GPL-3.0 | 메타데이터만 |
| FLAIROx/Kinetix | 실험 4의 환경과 공식 PPO | MIT | 메타데이터만 |
| google-deepmind/tracr | 프로그램 → Transformer 가중치 컴파일러(비교용) | Apache-2.0 | 메타데이터만 |
| andyrdt/refusal_direction | Arditi et al.의 방향 제거 구현(실험 8 기준선 후보) | Apache-2.0 | 메타데이터만 |

공식 하네스 코드(프롬프트 서버, forward pass 서버)는 공개되지 않았습니다. 로그 README의 "Not included"에 적혀 있습니다. 공개된 스크립트는 평가·채점·편집 적용(`exp8_math/code/`), 실험 7 학습·평가(`exp7_sycophancy/code/`), sandbox(`harness/sandbox.py`)입니다.
