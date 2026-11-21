# 02. 선행연구, 공식 저장소, 관련 분석

## 1. 선행연구 계보

PhysVAM이 직접 겨냥하는 질문은 "동적 물체 조작에서 행동이 미래 작업 진행도를 어떻게 바꾸는가"입니다. 논문은 다섯 갈래에 자신을 놓습니다(§2 Related Work, 참고문헌 [1]–[32]).

| 갈래 | 대표 연구 (논문 참고문헌) | PhysVAM과의 관계 |
|---|---|---|
| VLA와 월드 행동 모델 (WAM) | RT-2 [1], OpenVLA [11], Fast-WAM [3], DynamicWAM [4], ImageWAM [16], OpenWAM [17], InternVLA-A1.5 [13] | ActionDiT는 Fast-WAM의 백본 설정을 따릅니다. Fast-WAM과 DynamicWAM은 표 1의 기준선입니다 |
| 동적 조작 VLA | DynamicVLA [2] (DOM 벤치마크), PUMA [5] (광흐름과 미래 물체 상태) | DOM의 폐루프 설정을 빌려 시뮬을 만들었습니다. DynamicVLA와 PUMA는 주 기준선입니다 |
| 가치 모델 (정책 개선용) | π0.6* [9] (가치 기반 이점), ViVa [8], WCM [10], RISE [20], WVM [18] | 가치 목표를 학습하는 점은 같습니다. PhysVAM은 현재 값이 아니라 여러 지평의 미래 값을 씁니다 |
| 미래 시각 특징과 물리 | VLA-JEPA [6], PhysMani [24], Physics-aware SE(3) 예측 [26], ChronoFlow-Policy [27] | 시각 미래 예측 대신 작업 진행도 값을 씁니다 |
| 잠재 조건과 게이트 주입 | Faster-WAM [29] (미래 조건의 추론 시 주입), RoboFlow4D [32] | 가치 특징을 게이트된 교차 주의로 넣는 방식의 근거입니다 |

보조 인용: FoundationPose [30] (물체 자세 추정, 실로봇), OnePoseviaGen [31] (메시 생성), Wan [25] (영상 확산 백본).

**계보상의 위치.** 논문은 "현재 값 평가"(가치 모델)와 "미래 영상"(WAM) 사이의 빈칸을 겨냥합니다. 빈칸의 실질적 차이는 감독 신호입니다. 미래 영상은 픽셀을, 현재 가치는 한 시점의 진행도를 맞춥니다. PhysVAM은 같은 시점에서 여러 미래 지평의 진행도를 맞춥니다(§1의 Fig. 1A).

## 2. 공식 저장소와 프로젝트 페이지 (GitHub 표)

2026-11-21 기준으로 확인했습니다. GitHub 검색은 `PhysVAM`, `physvam`, `"Future Value Action Model"`로 했습니다. 코드 검색 API는 인증이 필요해 쓰지 못했습니다.

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 클론 | 상태 |
|---|---|---|---|---|---|---|
| 공식 코드 | 찾지 못함 | - | - | - | 없음 | 프로젝트 페이지의 Code 버튼이 빈 값(`config.js`의 `codeUrl: ''`) |
| [MetaRobo/physvam.github.io](https://github.com/MetaRobo/physvam.github.io) | 프로젝트 페이지 소스 (GitHub Pages 사이트) | 없음 (API license: null) | 0 | 2026-11-19 10:58 UTC | 안 함 (웹사이트 소스이며 코드가 아님) | 생성 2026-11-19 08:34 UTC. 검색으로 찾은 PhysVAM 관련 저장소는 이것 하나뿐 |
| 프로젝트 페이지 | https://metarobo.github.io/physvam.github.io/ | - | - | - | - | arXiv 링크는 있음. Code, Dataset, Model 버튼은 빈 값 |

- 프로젝트 페이지의 저자, 소속, BibTeX 항목은 비어 있거나 arXiv와 같습니다. 저작권 표기는 없습니다.
- 공개 데이터와 모델 가중치는 찾지 못했습니다. `config.js`의 `datasetUrl`, `modelUrl`이 비어 있습니다.

## 4. 학회, 버전, 라이선스

### 4.1 학회 표기 (venue)

- arXiv 초록 페이지: 학회, 저널 참조, DOI 발행 정보 없음. Comments는 "23 pages, 10 figures"입니다.
- 논문 PDF: 학회 표기 없음. 날짜는 "November 19, 2026"입니다.
- 프로젝트 페이지: 학회 표기 없음. 영어와 한국어 어디에도 CoRL, ICRA, RSS 같은 말이 없습니다.
- 결론: **학회 표기 없음 (arXiv)**. 공식 목록에서 확인하지 못했고 저자 측 학회 표기도 없습니다. 학회 연도 표기를 쓰지 않습니다.

### 4.2 라이선스 (arXiv abs 페이지, 2026-11-21 확인)

- "view license" 링크: `http://arxiv.org/licenses/nonexclusive-distrib/1.0/` (arXiv 비배타적 배포 라이선스 1.0)입니다.
- CC BY 4.0이 아닙니다. 따라서 논문 그림은 공개 산출물에 넣지 않습니다. 저자가 Creative Commons 조건을 주지 않았으므로 재배포 권리는 arXiv 배포에 한정됩니다.
- PDF 본문 검사: `pdftotext` 결과에서 © 표기, "All rights reserved", 회사 저작권 문구를 찾지 못했습니다. 첫 쪽의 소속(Tencent, Cambridge)은 고지가 아닙니다. 그래도 라이선스가 CC BY가 아니므로 그림은 쓰지 않습니다.
- 공개본은 저자 자체 그래프(`report/figures/own_*.png`)와 텍스트 인용만 씁니다.

### 4.3 버전

- v1만 있습니다(2026-11-19). 분석 대상은 v1입니다.
