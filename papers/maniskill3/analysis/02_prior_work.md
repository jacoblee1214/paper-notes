# 02. 선행연구, 관련 자료, 이 저장소의 관계

## 0. 기본 정보

| 항목 | 값 | 출처 |
|---|---|---|
| 논문 | ManiSkill3 (arXiv:2410.00425 v2, 2025-05-30) | arXiv abs 페이지 (2026-11-22 확인) |
| 학회 | **RSS 2025** (Robotics: Science and Systems XXI, Los Angeles). 제목은 "Demonstrating GPU Parallelized Robot Simulation and Rendering for Generalizable Embodied AI with ManiSkill3", DOI 10.15607/RSS.2025.XXI.021 | roboticsproceedings.org/rss21/p021.html (공식 proceedings, 확인) |
| 발표 형식 | 포스터 (RSS 프로그램 목록 기준, 검색 결과. proceedings 페이지에는 형식 미표기) | roboticsconference.org/2025/program/papers/21/ (검색 결과) |
| 워크숍 판 | ICLR 2025 Robot Learning 워크숍 포스터(같은 제목 계열) | iclr.cc/virtual/2025/32495 (검색 결과, 원문 미확인) |
| 라이선스 | arXiv 페이지 "view license" 링크: `creativecommons.org/licenses/by/4.0/` (CC BY 4.0). 아이콘 by-4.0.png | arXiv abs HTML (2026-11-22 확인) |
| 저작권 표기 | PDF 본문에 ©, copyright, All rights reserved 없음 (pdftotext 전체 검사) | `paper/paper.txt` |
| 코드 라이선스 | Apache-2.0 (논문 초록에도 명시) | 초록, 공식 저장소 `LICENSE` |

## 1. GitHub·프로젝트 자료 표

| 자료 | 주소 | 라이선스 | 최근 push / 날짜 | 스타 | 클론 여부 |
|---|---|---|---|---|---|
| **공식 코드 (ManiSkill)** | github.com/mani-skill/ManiSkill (haosulab/ManiSkill 경로도 이 저장소로 연결) | Apache-2.0 | 2026-08-04 (pushedAt), 클론 HEAD `62ff3a5` (커밋 2026-08-01) | 3,533 (gh 조회 2026-11-22) | **예** (`maniskill3/repo/`, `--depth 1`, 정적 읽기만). 615 MB |
| 프로젝트 사이트 | maniskill.ai | — | — | — | 아님 (논문 초록에 명시) |
| 문서·튜토리얼 | maniskill.readthedocs.io | — | — | — | 아님 |
| 데모 갤러리 | maniskill.readthedocs.io/en/latest/user_guide/demos/gallery.html | — | — | — | 아님 |
| 선행 버전 ManiSkill2 | 논문 [38, 19]에서 인용 | 미확인 | 미확인 | 미확인 | 아님 |
| SIMPLER (real2sim 기준) | 논문 [31]. 저장소 주소는 본문에 없음 | 미확인 | 미확인 | 미확인 | 아님 (논문 Fig. 25의 원본 비교만 인용) |
| Isaac Lab (비교 대상) | 논문 [36]. 주소는 본문에 없음 | 미확인 | 미확인 | 미확인 | 아님 |
| LeRobot (Koch 설정) | 논문 [4], 주소 미기재 | 미확인 | 미확인 | 미확인 | 아님 |

판단: 논문 저자 이름이 붙은 공식 저장소가 하나이고 Apache-2.0입니다. 공식 저장소가 있으므로 `04_code_verification.md`(D1–Dn)에서 대조했습니다.

- 클론 이후 `pyproject.toml`과 `setup.py`의 버전은 3.0.1입니다(`setup.py` 10행).
- 설치, 실행, import는 하지 않았습니다. 클론 내용은 신뢰하지 않는 데이터로 취급했습니다.

## 2. 선행연구 계보

| 계열 | 대표 작업 (논문 인용) | ManiSkill3와의 관계 |
|---|---|---|
| 전신 | ManiSkill, ManiSkill2 [38, 19] | 과제 정의와 로봇 모델 상당 부분을 이어받음. ManiSkill2는 CPU 시뮬레이션 기준 |
| GPU 병렬 시뮬레이션 | Isaac Gym·Isaac Lab [36], MJX·Brax [17, 49] | 같은 계열 (PhysX 기반 Isaac Lab과 공통). 렌더링 병렬화는 ManiSkill3가 먼저라고 주장 (§II) |
| CPU 기반 장면 시뮬레이터 | RoboCasa [39], Habitat [46], AI2-THOR [29], OmniGibson [30], RLBench [25] | 대비 대상. 장면 다양성은 높지만 느림 |
| 데이터 수집 | MimicGen [35], RFCL [47], RLPD [2] | 데모 생성에 RFCL, RLPD 사용 (§III-G) |
| 시각 조작 기준선 | Diffusion Policy [13], ACT [54], PerAct [45], Octo [40], RT-X [14], RDT-1B [32] | ManiSkill3가 기준선을 함께 제공 (§IV-B). VLA는 평가만 |
| real2sim 평가 | SIMPLER [31] | ManiSkill3가 GPU 포팅 (§III-E, §VII-K) |
| 로봇 teleop | 웹 기반 teleop [11, 15] | ManiSkill3는 VR 정렬 방식 주장 (§III-D) |

## 3. 가까운 대안과 차이 (Table I 기준, 저자 주장)

Table I의 열은 ManiSkill3, Isaac Lab, RoboCasa, RLBench, OmniGibson, Habitat, AI2-THOR 순서입니다(열 머리 글자 대조). "Parallelized Heterogeneous Simulation"은 ManiSkill3만 ✓입니다. "Task Categories"는 ManiSkill3 12, Isaac Lab 5, OmniGibson 2, RoboCasa 2, RLBench·Habitat·AI2-THOR 1입니다. 이 표는 저자가 만든 표이므로 독립 검증이 없습니다.

## 5. 버전·학회 정리

| 판 | 제목 (PDF) | 날짜 | 학회 |
|---|---|---|---|
| arXiv v1 | ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI | 2024-10-01 | — |
| arXiv v2 (최신) | Demonstrating GPU Parallelized Robot Simulation and Rendering for Generalizable Embodied AI with ManiSkill3 | 2025-05-30 | RSS 2025 (데모·포스터 판 추정) |

v1과 v2의 차이는 `01_paper_analysis.md` §0에 있습니다.
