# 02. 선행 연구, 코드·라이선스, 학회 확인

## 1. 학회 표기 확인

- **arXiv 댓글:** `Code: this https URL`만 있고 학회명 없음. 저널 참조 없음. 버전은 v1 하나(abs 페이지 "[v1] Thu, 26 Jun 2025"로 확인).
- **NeurIPS 2026 공식 virtual 페이지** (`neurips.cc/virtual/2026/poster/155786`, 2026-11-20 확인): 제목이 **"WorldVLA: A Unified Vision-Language-Action and World Model"**, 저자 목록 14명(Siteng Huang, Yuqian Yuan, Kehan Li, Bohan Hou 포함, Yibing Song 없음). 요약 첫 문장은 arXiv v1과 같지만, 두 번째 문장 이후에 "**97.4% success rate on the LIBERO benchmark without pretraining**"이 있습니다. arXiv v1에는 97.4%가 없고, 이 수치는 후속 저장소 **RynnVLA-002**(arXiv:2511.17502) README의 수치와 일치합니다. 페이지에는 arXiv 링크가 없고 OpenReview(`openreview.net/forum?id=2uT2MbEwuy`)와 GitHub `alibaba-damo-academy/WorldVLA`가 연결되어 있습니다.
- **판단:** NeurIPS 2026 포스터 페이지로 채택은 확인됩니다. 채택본은 제1저자(Jun Cen)와 초록 첫 문장("an autoregressive action world model that unifies action and image understanding and generation")이 arXiv v1과 같으므로 **같은 연구의 개정판**으로 판단합니다(메인 세션 2026-11-20 확인). 다만 제목, 저자 구성, LIBERO 97.4% 등 수치가 바뀌었으므로, 이 분석의 수치와 판정은 모두 arXiv v1 기준입니다. OpenReview 본문은 열어 보지 않았습니다.
- 따라서 front matter는 `venue: "NeurIPS 2026"`, `venue_note`에 이 불일치를 함께 적었습니다.

## 2. 코드와 라이선스

| 항목 | 값 | 확인 방법 |
|---|---|---|
| 논문 arXiv 댓글·본문의 코드 링크 | `github.com/alibaba-damo-academy/WorldVLA` | arXiv abs, 원문 1쪽 |
| 이 링크의 현재 상태 | **RynnVLA-002로 이름이 바뀌어 리다이렉트**. GitHub API가 `alibaba-damo-academy/RynnVLA-002`를 반환 | `gh api` |
| 현재 기본 브랜치 `main` | RynnVLA-002 코드(2025-11 이후). 최종 push 2025-12-02, 별 1,143개, archived 아님 | `gh api`, 로컬 clone `repo/` |
| **WorldVLA v1 코드** | 브랜치 **`worldvla`** (`origin/worldvla`), 커밋 15개, 2025-06-23 최초 커밋 ~ 2025-10-10 최종 `ebdc36d` | `git log origin/worldvla` |
| 라이선스 (`worldvla` 브랜치) | Apache License 2.0, `LICENSE` 파일 있음 | `git show origin/worldvla:LICENSE` |
| 라이선스 (`main`) | `LICENSE` 파일 없음(GitHub API license=null). README는 Apache 2.0 링크를 걸지만 링크가 깨짐 | `gh api contents/LICENSE` 404 |
| 모델 가중치 | HF `Alibaba-DAMO-Academy/WorldVLA`(v1), `.../RynnVLA-002`(후속). 다운로드하지 않음 | README |

- **논문 라이선스:** arXiv abs 페이지 확인 결과 `licenses/by-nc-sa/4.0/` ("Rights to this article"), 배지 `by-nc-sa-4.0.png`, 저작권 링크 `creativecommons.org/licenses/by-nc-sa/4.0/`. **CC BY-NC-SA 4.0**입니다. CC BY가 아니므로 게시본에는 논문 그림을 넣지 않습니다. 논문 그림은 개인용 리포트에서만 씁니다.
- **코드 라이선스(Apache 2.0)와 논문 라이선스(CC BY-NC-SA 4.0)는 별개**입니다. 코드를 가져다 쓰는 것은 Apache 조건을 따르고, 논문 텍스트·그림은 BY-NC-SA 조건을 따릅니다.
- 공식 코드는 `repo/`에 얕은 clone이 아니라 전체 clone(이력 필요)으로 받았습니다. 정적 읽기만 했고 실행, import, 설치는 하지 않았습니다. `.gitignore`에 `repo/`가 있습니다.

