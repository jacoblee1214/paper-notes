# 02. 선행연구, 구성 요소의 출처, 자료

## 1. 계보

이 논문은 세 줄기가 만난 곳에 있습니다.

1. **코드 생성 LLM.** Codex(Chen 외, 2021, 논문 [1])가 docstring에서 Python 함수를 쓸 수 있음을 보였고, HumanEval 벤치마크를 만들었습니다. PaLM(논문 [11])과 code-davinci-001 계열이 뒤를 이었습니다. 논문은 이 모델들이 "주석 → 코드"의 짝을 배운다는 점을 로봇 정책에 쓰려고 합니다.
2. **LLM 기반 로봇 계획.** Huang 외(2022, 논문 [14])는 LLM으로 행동 계획을 만들었습니다. SayCan(Ahn 외 2022, 논문 [17])은 LLM 점수와 기술의 가치 함수를 곱해 실행 가능한 기술을 고릅니다. Inner Monologue(논문 [18])는 성공 검출기 피드백으로 재계획합니다. Socratic Models(논문 [16])는 시각-언어 모델의 출력을 프롬프트에 넣습니다. 이 방법들은 모두 **미리 정해진 기술 집합**을 전제합니다. 논문은 이 전제를 "지각과 제어 사이의 피드백 루프를 LLM이 직접 다루지 못하게 한다"고 비판합니다(§I).
3. **언어 조건부 조작 정책.** CLIPort(Shridhar 외, 논문 [36])는 언어 조건부 픽앤플레이스 정책으로, 시뮬에서 30k 시연으로 학습합니다. CALVIN(논문 [10])과 BC-Z(논문 [9])는 대규모 시연과 언어 조건부 학습의 대표입니다. 논문의 기준선은 CLIPort입니다.

그 밖의 근거:

- **프롬프트 기반 추론.** Chain-of-Thought(Wei 외, 논문 [47]), Least-to-Most(논문 [46]), Zero-shot reasoner(Kojima 외, 논문 [15])는 "단계적으로 쓰기"를 제시합니다. 논문은 계층 함수 생성이 "함수형 프로그래밍 방식의 CoT"라고 설명합니다(§III).
- **일반화 분류.** Hupkes 외(2020, 논문 [23])는 구성성 일반화를 다섯 유형으로 나눕니다. RoboCodeGen 분석은 이 분류를 씁니다(부록 E.2).
- **프로그램 합성.** 논문은 프로그램 합성으로 그림을 그리거나 2D 정책을 만든 선행 연구([50], [51])를 인용하고, 이들이 2D 과제에 머물렀다고 씁니다. 이 연구들이 CaP의 직접 선행 연구입니다.
- **인지 모듈.** MDETR(Kamath 외, ICCV 2021, 논문 [2])과 ViLD(Gu 외, 논문 [3])는 열린 어휘 검출기입니다. 논문은 이를 "프로그램에 붙는 API"로 씁니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문 내 역할 | 공개 여부 |
|---|---|---|---|
| 코드 LLM (code-davinci-002) | OpenAI Codex, Chen 외 2021 [1] | 기본 생성 모델 | API(외부). 키 필요 |
| GPT-3 6.7B, InstructGPT 175B | Brown 외 2020 [12], Ouyang 외 2022 [22] | 표 I의 텍스트 모델 | API(외부) |
| 계층 함수 생성 | 본 논문 | 핵심 기여 | 노트북(`lmp_fgen`) |
| 공간 연산 | NumPy, Shapely | 프롬프트가 쓰도록 힌트 | 공개 라이브러리 |
| 객체 검출 | MDETR [2], ViLD [3] | 실로봇 인지 | 실로봇 코드 없음 |
| 시뮬 환경 | Socratic Models [16], Inner Monologue [18]의 환경 | 표 III, VI | PyBullet 환경(노트북) |
| 기준선 CLIPort | Shridhar 외 2021 [36] | 표 III, VI | 학습 코드 없음 |
| 기준선 NL 계획기 | 본 논문(언어 프롬프트) | 표 III, VI | 프롬프트 링크만 |
| HumanEval 평가 | `human-eval` (OpenAI) | 표 II, V | 노트북이 호출 |
| 표 IV의 CoT 기준선 | Wei 외 2022 [47] | 공간 추론 | 프롬프트 노트북 |

## 4. GitHub와 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 최근 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (code_as_policies 폴더)** | https://github.com/google-research/google-research/tree/master/code_as_policies | Apache-2.0 (모노레포) | 모노레포 별 39,253(폴더 단위 아님), 모노레포 최근 push 2026-11-21, 폴더 마지막 커밋 2024-02-16(`8c116f61`) | 희소 얕은 clone을 `repo/`에 보관, 정적 읽기만. 노트북 8개, `.py` 없음 |
| 프로젝트 페이지 | https://code-as-policies.github.io/ | 미확인 | - | 프롬프트 `.txt` 링크(시뮬, 모바일, 그리기 등), 영상 |
| Colab 노트북 | 저장소 노트북과 같은 내용 | Apache-2.0 | - | 실행하지 않음 |
| Hugging Face Space | https://huggingface.co/spaces/jackyliang42/code-as-policies | 미확인 | 미확인 | 링크만 확인, 내용 미검토 |
| Google AI 블로그 (2022-11) | https://ai.googleblog.com/2022/11/robots-that-write-their-own-code.html | - | - | 프로젝트 페이지에서 링크, 열람하지 않음 |
| 시뮬 환경 | 공식 저장소의 `Interactive_Demo.ipynb` | Apache-2.0 | - | PyBullet UR5e 환경과 LMP 래퍼 포함 |
| 기준선 CLIPort, 학습 코드 | 저장소 밖 | 미확인 | 미확인 | 이 저장소에 없음 |
| OpenAI 모델(code-davinci-002, text-davinci-002) | OpenAI API | 상용 API | - | 키가 필요하고 호출하지 않음. 종료 여부 미확인 |

- 모노레포의 별 수는 `google-research` 전체의 값이라 이 폴더의 인기와는 관계가 없습니다.
- 공식 노트북의 내용과 논문의 대응은 `04_code_verification.md` §2에 있습니다.
