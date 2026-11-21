# HINT-WAM 선행연구와 공식 저장소 (arXiv:2611.24789)

## 1. 선행연구 계보

| 계보 | 대표 연구 | 본 논문과의 관계 |
|---|---|---|
| 세계 행동 모델 (WAM) | Video Prediction Policy [17], Unified World Models [49], Unified Video Action [25], LingBot-VA [21], DreamZero, Fast-WAM [45] | 기반 모델은 Fast-WAM입니다(§3.1). 본 논문은 행동 전문가만 학습합니다 |
| 흐름 정합 VLA | π0 [5], OpenVLA [18] (벤치마크 관례) | 흐름 정합 행동 전문가의 구조적 배경 |
| 온폴리시 RL for VLA/WAM | SimpleVLA-RL [20], VLA-RL [32], WAM-RL [36] | 희소 또는 재구성 보상 기반. 본 논문은 보상과 시뮬레이터가 필요 없다고 주장합니다 |
| DAgger와 노출 편향 | DAgger [38], 입력 교란 [33] | 학생이 자기 상태에서 학습한다는 원칙의 출처 |
| 온폴리시 증류 (LLM) | GKD [1], MiniLLM [15], Self-Distilled Reasoner [47] | 학생 생성 + 교사 점수의 개념. Self-Distilled Reasoner는 같은 모델을 특권 정보로 교사로 쓰는 방식의 출처 |
| 확산/흐름 온폴리시 증류 | Flow-OPD [13], DiffusionOPD [24], dOPSD [11] (확산 언어 모델, 저자 본인의 선행 연구) | 흐름 모델에서 속도 회귀를 쓰는 방식. 본 논문은 이를 속도 회귀 대조군으로 둡니다 |
| 특권 교사 자기 증류 (다른 영역) | PRISM [12] (다중 에이전트 설계, 특권 형제 모델) | 같은 발상을 다른 영역에 적용한 연구. 저자 본인의 선행 연구입니다 |
| 온폴리시 증류 for WAM | WAM-OPD (joint video–action, arXiv:2608.22364) [44], WAM-OPD (replay, arXiv:2609.34250) [29] | 외부 교사를 쓰는 방식. 본 논문은 교사를 학생 자신에서 만든다는 점이 다릅니다 |
| 특권 정보 학습 | LUPI [43], Learning by Cheating [8], HER [4] | 특권 정보로 교사를 만드는 개념 |
| 에너지 점수 | Gneiting–Raftery [14], Székely–Rizzo [41] | 적합 규칙. 생성 모델 학습 목적으로 쓰는 경우는 드묾 |
| 게이트 적응기 | Flamingo [3], ControlNet [46], Set Transformer [19] | 게이트 교차 어텐션과 영-초기화. 본 논문은 α = 1과 W_up = 0 조합을 씁니다 |
| 망각과 재생 | RL's Razor [40], tiny episodic memory [7] | 온라인 RL이 덜 잊는다는 주장 [40]과 재생 기준 [7] |

- 본 논문의 새로움 주장은 "WAM을 위한 첫 온폴리시 자기 증류" (§2.3)입니다. 이 분석은 문헌을 전수 검색하지 않았으므로 이 주장을 확인하지 않았습니다.
- 가장 가까운 대안은 WAM-OPD 두 편입니다. 외부 교사가 필요하다는 점에서 계산 비용이 본 논문과 다릅니다(§2.3).

## 2. 공식 코드와 저장소

| 저장소 | 관계 | 라이선스 | 커밋과 날짜 | 별 | 이 분석에서 한 것 |
|---|---|---|---|---|---|
| [tuandattt/hintwam_code](https://github.com/tuandattt/hintwam_code) | 공식 학습 코드 (HINT-WAM) | `LICENSE` 파일은 MIT (저작권 "The FastWAM Authors"). GitHub API는 `NOASSERTION` | `6a49a33` (2026-11-19), 저장소 생성 2026-11-16 | 4 | 얕은 복제 `repo/` (정적 읽기만). 실행과 설치 안 함. `04_code_verification.md` |
| [tuandattttt/HINT-WAM](https://huggingface.co/tuandattttt/HINT-WAM) (Hugging Face) | 체크포인트 | 태그 `license:mit` | 마지막 수정 2026-11-19 | 다운로드 36 | 메타데이터만 확인. 가중치는 받지 않음 |
| [tuandattt.github.io/hintwam](https://tuandattt.github.io/hintwam/) | 프로젝트 페이지 | 명시 없음 | 확인 안 함 | – | 읽음. 학회 표기 없음. BibTeX는 `journal={arXiv preprint}` |
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | 기반 모델과 평가기 (Fast-WAM, arXiv:2603.16666) | `LICENSE` 파일은 MIT (GitHub API는 `NOASSERTION`) | `7faa711` (2026-08-20), 생성 2026-03-17 | 1,669 | 이 분석에서 복제하지 않음. HINT-WAM 저장소 README가 평가기를 이 저장소에서 쓰라고 함 |

- 두 저장소 모두 GitHub API 라이선스 필드가 `NOASSERTION`입니다. 저장소의 `LICENSE` 파일이 MIT이므로 실사용 판단은 파일 기준으로 합니다.
- 체크포인트 가중치는 받지 않았습니다. 학습 설정 파일과 README는 `04_code_verification.md`에서 대조했습니다.

## 4. 본 논문의 위치

- **새로운 점.** 같은 모델을 교사와 학생으로 쓰고, 교사는 미래 영상을 받고, 학생의 샘플러 상태를 감독하며, 흐름 정합 행동 전문가에 에너지 점수를 씁니다. 외부 교사와 보상, 시뮬레이터 상호작용이 필요 없습니다.
- **기존 것의 결합.** 특권 정보 학습(LUPI), 자기 증류(SDR), 온폴리시 상태(DAgger), 에너지 점수(Gneiting–Raftery)를 WAM 행동 전문가에 조합한 것입니다. 각 요소의 효과는 절제 실험으로 나눠 보았습니다(01 §4.4).
- **논문이 놓친 비교(추정).** 외부 교사 방식 WAM-OPD와의 직접 비교는 없습니다. 같은 계산 예산에서 비교했는지 확인되지 않았습니다.
