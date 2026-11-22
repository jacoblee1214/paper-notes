# 02. 선행연구, 가까운 대안, 자료

## 1. 계보

RoboCasa는 **시뮬레이터 + 과제 세트 + 생성 데이터**를 한 묶음으로 낸 벤치마크 논문입니다. 세 축의 선행연구가 따로 있습니다.

| 축 | 선행연구 (논문 번호) | RoboCasa에서 한 일 |
|---|---|---|
| 시뮬레이터 기반 | RoboSuite [51] (MuJoCo 기반 모듈형 조작 환경) | 물리·컨트롤러를 그대로 씀. 이동 조작과 방 단위 장면을 추가 |
| 데이터 생성 | MimicGen [34] (CoRL 2023), Optimus [6] (arXiv 2305.16309) | 사람 시연을 객체 중심 하위 과제로 나눠 시뮬에서 변환·이어 붙임. MimicGen을 원자 과제 24개에 확장 |
| 정책 학습 | BC-Transformer (Optimus 계열, RoboMimic 구현) [6], Diffusion Policy [4] | BC-Transformer를 주 정책으로 씀. Diffusion Policy는 부록에서 낮은 성능을 보고 |
| 과제 설계 | LLM 기반 과제 생성 (논문 [38, 35, 42]의 생성 AI 도구) | GPT-4와 Gemini 1.5로 활동 20개와 과제 블루프린트 75개 |
| 실로봇 | DROID [20] | 공동학습 실험에 DROID 하드웨어 사용 |

## 2. 같은 문제를 다룬 가까운 대안 (논문 Table I 기준)

논문 Table I은 RoboCasa를 AI2-THOR [21], Habitat 2.0 [40], ManiSkill2 [10], BEHAVIOR-1K [27], Optimus [6], MimicGen [34] 등과 비교합니다. 표 셀의 체크 표시는 PDF 텍스트 추출로 열 정렬이 깨져 있어 **이 분석에서는 셀별로 옮기지 않았습니다.** 논문이 강조하는 차이는 본문 II에 있습니다.

- **이동 조작과 방 단위 장면:** RoboCasa와 몇몇 프레임워크만 방 전체를 다룹니다.
- **실제 물리 상호작용:** AI2-THOR와 Habitat 2.0은 물체 파지·배치를 실제 물리로 하지 않는다고 논문이 지적합니다.
- **생성 데이터 포함:** 공개 데이터와 MimicGen 생성을 함께 제공하는 것은 RoboCasa 중심이라는 주장입니다(논문은 BEHAVIOR-1K가 대규모 데이터셋을 제공하지 않는다고 비교합니다).

**후속 연구.** RoboCasa365(arXiv 2603.04356, ICLR 2026 포스터 목록 확인, CC BY 4.0)는 365개 과제, 2,500개 장면, 사람 시연 600시간 이상, 합성 시연 1,600시간 이상을 제공합니다. 이 저장소의 공식 코드(v1.0.1)는 이 후속 버전 기준입니다. **이 분석의 대상은 2024년 원판(arXiv:2406.02523)이며, RoboCasa365는 분석하지 않았습니다.**

## 4. 공식 코드와 자료

| 저장소·페이지 | 관계 | 라이선스 | 스타·포크 | 마지막 푸시 | 대조 기준 |
|---|---|---|---|---|---|
| [robocasa/robocasa](https://github.com/robocasa/robocasa) | 공식 코드 (2024 원판 + 후속 RoboCasa365) | LICENSE 파일은 MIT (2026 저작권). GitHub API 표시는 `Other` | 1,881 / 274 | 2026-11-18 | HEAD `7bfc60c` (2026-11-18). 2024 원판 대조는 `be5d99c`(2024-05-29), `a6c2dc6`(2024-10-31) |
| [robocasa.ai](https://robocasa.ai) | 프로젝트 페이지, 문서, 리더보드 | - | - | - | 2026-11-22 확인. BibTeX는 RSS 2024 |
| [robocasa.ai/docs](https://robocasa.ai/docs/build/html/introduction/overview.html) | 문서 | - | - | - | v0.2 문서의 데이터 설명 확인 |
| [RoboCasa365 (arXiv 2603.04356)](https://arxiv.org/abs/2603.04356) | 후속 논문 | CC BY 4.0 (HTML 링크 확인) | - | - | 분석하지 않음 |
| 공식 데이터셋 (사람 시연, MimicGen 생성) | 레지스트리의 Box 링크 | 저장소 라이선스 외, 확인 안 함 | - | - | 받지 않음. 용량 미측정 |
| 공식 에셋 (약 10 GB) | `download_kitchen_assets` | 확인 안 함 | - | - | 받지 않음 |

- 공식 코드 클론: `repo/robocasa/` (HEAD, 85 MB), `repo/robocasa-v0.2/` (태그 v0.2, 14 MB), `repo/robocasa-history/` (커밋 목록용 treeless, 6.4 MB), `repo/robocasa-2024-05/` (S0), `repo/robocasa-2024-10/` (S1). 모두 정적 읽기만 했고 gitignore 되어 있습니다.
- 공개 저장소에 쓴 GitHub 수치는 2026-11-22 API 조회 기준입니다.

## 5. 버전과 논문 기록

| 항목 | 값 |
|---|---|
| arXiv | 2406.02523, v1만 있음 (2024-06-04 제출, `[v1]`, 확인 2026-11-22) |
| 학회 | RSS 2024 (Robotics: Science and Systems XX), 공식 proceedings p050, DOI 10.15607/RSS.2024.XX.050 |
| proceedings 제목 | "RoboCasa: Large-Scale Simulation of **Household** Tasks for Generalist Robots" (arXiv와 다름, "Everyday") |
| 저자 | Nasiriany, Maddukuri*, Zhang*, Parikh, Lo, Joshi, Mandlekar (NVIDIA), Zhu |
| 소속 | UT Austin, NVIDIA Research |
| 라이선스 | arXiv abs "view license" → `creativecommons.org/licenses/by/4.0/` (CC BY 4.0) |
