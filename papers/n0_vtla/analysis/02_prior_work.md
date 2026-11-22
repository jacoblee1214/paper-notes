# 02. 선행연구, 공식 저장소, 관련 분석

## 1. 계보

**기반 VLA.** PaliGemma [6]의 시각-언어 백본과 π0 [9], π0.5 [58]의 flow-matching action expert를 쓴다. 정규 32차원 행동 컨테이너는 π0.5에서 가져왔다(§3). 비교 기준선 ACT [93], StarVLA-α [82], InternVLA-A1 [14], Xiaomi-Robotics-0 [15], GigaWorld-Policy [81]는 논문이 내부에서 재현했거나 같은 조건으로 평가했다고 적는다. 재현 근거는 없다.

**촉각 입력 방식.** 논문은 기존 방식을 두 갈래로 나눈다. (a) 촉각 토큰을 VL prefix에 이어 붙여 카메라처럼 다룬다 [22 OmniVTLA, 36 Tactile-VLA]. (b) 현재 촉각 판독을 action 경로에 주입한다 [77 Reactive Diffusion Policy, 91 TouchGuide]. 논문의 대안은 셋째 길, 즉 예측된 촉각 변화 z를 action expert에 주는 것이다. 공식 저장소에도 두 갈래 방식이 `tactile_mode`의 `vlm_concat`, `expert_concat`으로 남아 있다(04 D6).

**촉각 VLA와 힘 입력.** TLA [31], VLA-Touch [7], ForceVLA [84], TA-VLA [90], HapticVLA [30], FTP-1 [85](촉각 정책 일반화, 수천 시간 규모), VITaL [29](기준선 표에 등장), UniVTAC [18](시뮬 플랫폼, 시뮬 평가 기반).

**예측형 촉각.** Imagine2Touch [3], TTP [88], Visuo-Tactile World Models [34], OmniVTA [94], Tactile-WAM [71], Dream-Tac [45], VT-WAM [65], TacForeSight [87], ContactWorld [92], UniPi [26], GR-1 [69], GR-2 [17], Unified World Models [96], World Guidance [62].

**JEPA 계열.** I-JEPA [1], V-JEPA [5], V-JEPA 2 [2], LeJEPA [4], LeWorldModel [49]. 논문은 잠재 공간에서 예측한다는 점에서 이들과 관련을 주장한다. 촉각 목표와 학습 배치가 다르다.

**사후 학습과 진행도.** RECAP [57](advantage 조건 정책 추출, 이 논문 ALTER의 원형), χ0 [83](단계 인식 상대 진행도, 논문의 직접 비교 대상), SARM [20], ARM [50], GR-RL [42], ProgVLA [38], ProgressVLA [78], Robo-ValueRL [73], SOP [56], LWD [68], VLA-RL [46], Robo-Dopamine [64], Vision-language value estimators [48]. ALTER는 χ0와 달리 단계 가중치를 시연 평균 시간으로 정하고, 촉각 기반 낙하 사건과 HIL 보정으로 국소 비교 감독을 더한다(§7 마지막 단락).

**사전 학습 데이터.** NeoData는 논문 밖 데이터 보고서 [51]에 있다. 공개 여부는 확인하지 못했다.

## 2. 공식 저장소와 프로젝트 페이지

| 항목 | 내용 | 확인 방법 | 상태 |
|---|---|---|---|
| [neoteai/N0-VTLA](https://github.com/neoteai/N0-VTLA) | 공식 코드(사후 학습 도구, 서빙, 검증 스크립트). 별 205, 포크 12, 생성 2026-07-25, 마지막 push 2026-08-17, 갱신 2026-11-20 | GitHub API(메타데이터) 와 `raw` README, LICENSE. API의 license 필드는 `NOASSERTION`(other) | 얕은 복제 `repo/`(커밋 `03a0ce4`, 2026-08-18, "docs: add contact information"). 정적 읽기만 |
| 저장소 LICENSE | CC BY-SA 4.0. NOTICE에 Apache-2.0 구성요소 표시. 가중치는 Gemma 이용 약관 | `repo/LICENSE`, `repo/README.md` | 코드 라이선스는 "CC BY-SA"라는 점이 독특하다. GitHub 자동 감지는 실패했다 |
| [NeoteAI/n0-vtla-base](https://huggingface.co/NeoteAI/n0-vtla-base) | 사전 학습 기반 체크포인트 `model.safetensors` **8.25 GB**, 수정 2026-10-20 | HF API 메타데이터(파일 목록과 크기). 받지 않음 | 미다운로드 |
| HF 체크포인트 두 개 | `n0_VTLA_insert_hole`(UniVTAC 단일 팔), `n0_VTLA_dual_bowl_place_stack`(NeoSim 이중 팔) | 저장소 README 표 | 이름만 확인, 크기 미확인 |
| [research.neoteai.com/n0-vtla/](https://research.neoteai.com/n0-vtla/) | 프로젝트 페이지. "Technical Report" 표기, 2026-07-25 날짜, BibTeX는 `eprint`만 있고 학회 없음 | 페이지 HTML | 학회 표기 없음 |

- **공식 코드의 범위.** README는 "대규모 사전 학습 파이프라인은 포함하지 않는다"고 적는다. 파일 머리말(`n0vtla_policy.py` 1–4행)은 "논문의 감독형 예측기 기반 단계(z*를 회귀)는 이 사후 학습 저장소에 없다"고 적는다. 따라서 공식 코드로 할 수 있는 것은 사후 학습, 추론, 검증 탐침이다. 논문의 1단계는 재현할 수 없다(04 D1).

## 4. 판단

- N0-VTLA는 "예측형 촉각 토큰 + 공동 학습 + 오프라인 advantage"의 세 요소를 묶은 기술 보고서다. 세 요소 각각의 기여를 분리한 결과는 보고되지 않았다.
- 공식 코드에는 촉각 입력 방식 세 가지와 탐침 스크립트가 있다. 그러나 결과 수치는 없다.
