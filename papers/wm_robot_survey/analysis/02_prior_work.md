# 선행연구와 자료 위치 (arXiv:2605.00080)

## 1. 서베이가 정리한 계보

서베이의 계보는 두 갈래입니다(Figure 2).

- **정책 쪽:** 분리형 "예측 후 행동"(UniPi, 2023 → VPP·Video2Act·Vidar, 2025) → 단일 백본(UVA, Cosmos Policy, DreamZero) → MoE/MoT(Motus, LingBot-VA, BagelVLA, Fast-WAM) → 통합 VLA(F1, InternVLA-A1, HALO, WorldVLA) → 잠재 공간(VLA-JEPA, JEPA-VLA, WoG).
- **시뮬레이터 쪽:** 롤아웃 검증(IRASim, GPC, Ctrl-World) → 강화학습용 학습 환경(World-Env, VLA-RFT, WMPO, WoVR) → 공동 진화(PlayWorld 등).

서베이가 자기 위치를 잡는 방식은 이렇습니다. 기존 서베이 Zhang 외(2025d)보다 세분화했고, 정책 중심이며, VLA와의 관계를 정의했다고 주장합니다(§1). 이 주장은 대체로 맞지만, 비교 대상 서베이의 내용은 이 분석에서 읽지 않았습니다.

## 2. 가까운 대안: 다른 서베이

- Zhang 외(2025d): 논문이 직접 언급하는 유일한 서베이 비교 대상입니다. 이 분석에서는 본문을 열어 보지 않았습니다.
- Ai 외(2025, Science Robotics): 학습 기반 동역학 모델 리뷰입니다. 서베이의 §2.2 배경에서 인용됩니다.

## 3. 구성 요소별 출처

| 구성 요소 | 서베이의 출처(인용 키) | 확인 |
|---|---|---|
| 정책 정의와 청크 예측 | Chi 외 2023, Zhao 외 2023 | 인용 키 해석 |
| VLA 계열 | Zitkovich 외 2023(RT-2), Kim 외 2025(OpenVLA), Black 외 2024(π0) | 인용 키 해석 |
| 흐름 정합·확산 | Lipman 외 2023, Liu 2022(Rectified flow, arXiv:2209.14577) | 참고문헌 확인 |
| 비디오 생성 기반 | Wan 2025(Team Wan, arXiv:2503.20314), Ali 외 2025(Cosmos, arXiv:2511.00062) | 참고문헌 확인 |
| JEPA 계열 | Assran 외 2023(I-JEPA), Assran 외 2025(V-JEPA 2, arXiv:2506.09985) | 참고문헌 확인 |
| 벤치마크 | Liu 외 2023a(LIBERO), Chen 외 2025d(RoboTwin 2.0), Mees 외 2022(CALVIN) | 참고문헌 확인 |

## 4. 서베이 인용과 이 저장소 폴더의 대응

서베이 본문에서 이름이 나오거나 표에 보이는 논문 중 이 저장소에 분석 폴더가 있는 것입니다. 본문 위치는 서베이 절 번호입니다.

| 폴더 | 논문 (arXiv, 학회 표기는 루트 README) | 서베이에서의 위치 |
|---|---|---|

이름만 나오고 폴더가 없거나 서베이가 인용하지 않은 것은 다음과 같습니다.


## 5. 서베이 저장소 (큐레이션 목록)

| 항목 | 값 | 확인 방법 |
|---|---|---|
| 저장소 | `NTUMARS/Awesome-World-Model-for-Robotics-Policy` | project 페이지와 arXiv 초록의 링크 |
| 성격 | 큐레이션 목록(awesome list). 서베이의 방법을 구현한 코드가 아닙니다 | 클론 후 README 구조 확인 |
| 별 · 포크 | 882 · 24 | GitHub API (2026-11-20) |
| 마지막 push | 2026-05-16 (commit `5f69b4e`) | GitHub API |
| 라이선스 | 파일 없음. API의 license 값 `null` | 클론에 LICENSE 없음 |
| 클론 | `repo/` (얕은 클론, 정적 읽기만). `.gitignore`에 포함 | 실행하거나 설치하지 않음 |
| 프로젝트 페이지 | https://ntumars.github.io/wm-robot-survey/ | 학회 표기 없음, BibTeX는 `@article` |


## 6. 관련 외부 자료

- 프로젝트 페이지 BibTeX는 `@article`입니다. 학회 표기는 없습니다.
- 학회 목록 확인: `conferences/corl2026/papers.json`에는 이 서베이의 제목이 없습니다. 웹 검색(2026-11-20)에서도 학회 채택 표기는 찾지 못했습니다. 마이크로소프트 리서치 게시 페이지는 venue를 "arXiv"로 표기합니다.
