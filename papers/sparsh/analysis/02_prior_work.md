# 02. 선행연구와 공개 자원

## 0. 학회와 발표 상태

- **CoRL 2024 채택, 공식 목록 확인.** PMLR 공식 목록(proceedings.mlr.press/v270, "Proceedings of The 8th Conference on Robot Learning", PMLR 270:885–915, 저자 목록과 제목 일치)에 있습니다. 프로젝트 페이지의 BibTeX도 "8th Annual Conference on Robot Learning", 2024를 씁니다.
- OpenReview 포럼 `xYJn2e1uu8`(BibTeX에 적힌 주소)는 자동 접근이 막혀(403 챌린지) 직접 확인하지 못했습니다. 발표 형식(oral, spotlight, poster)은 확인하지 못했습니다.
- arXiv 제출일(2024-10-31)은 학회 개최 전입니다. 채택 여부는 arXiv 메타데이터가 아니라 PMLR 목록으로 확인했습니다.

## 1. 계보

- **촉각 표현 학습 이전.** 촉각 이미지의 특성 추출은 센서·과제마다 따로 설계했습니다. 마커 추적과 FEM 기반 힘 추정(GelSight, GelSlim 계열 [1][2][28][29]), 과제별 종단 간 학습(삽입 [5][6], 파지 [8], 위치 추정 [9][10][11]).
- **촉각 사전학습 직전 시도.** MAE 기반 촉각 표현(Cao 외, IROS 2023 [33]), 시각 파운데이션 모델을 촉각에 옮긴 연구(Zandonati 외 [15]), 다중 감각 대조 학습(ObjectFolder [21][36], Touch and Go [20]).
- **이 논문의 위치.** 같은 시기 T3 [45], UniT [46]과 가장 가깝습니다. T3는 센서별 인코더에 MAE와 과제 라벨을 함께 씁니다. UniT는 GelSight Mini(마커)만 다루는 VQGAN입니다. Sparsh는 DIGIT·GelSight 2017·GelSight Mini 세 센서를 한 사전학습으로 묶고, 벤치마크 TacBench를 함께 제공한다는 점이 다릅니다(§1 Related work).
- **SSL 방법의 출처.** MAE [23], DINO [48], DINOv2 [49], I-JEPA [50], V-JEPA [51]. 저장소 README와 config는 이 방법들의 공식 구현 저장소를 기반으로 합니다(`repo/` 의 `tactile_ssl/algorithm/`, `config/model/`).

## 2. 가까운 대안과 기준선

| 방식 | 대표 | 이 논문과 비교 가능한 점 |
|---|---|---|
| 과제·센서별 E2E 학습 | 논문 기준선 E2E | 같은 ViT 용량, 무작위 초기화 |
| 센서별 MAE + 라벨 보조 | T3 [45] | 같은 MAE 목적, 라벨 사용 방식이 다름 |
| 비전 파운데이션 특징 | 논문 §1, [15] | 촉각 데이터에서 무엇을 얻는지 비교가 본문에 없음 |
| 시각-촉각 정렬 | [40][42][43][44] | 물리 접촉 속성 대신 질감·색 위주 |

- 논문에 **시각 파운데이션 모델(예: DINOv2 원본 가중치)을 같은 TacBench에서 평가한 결과가 없습니다.** 개선의 원인이 촉각 데이터 사전학습인지 SSL 일반 효과인지 구분하는 기준선이 빠져 있습니다.


## 4. GitHub와 공개 자원 (확인 2026-11-21)

| 자원 | 주소 | 라이선스 | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| 공식 코드 (논문) | [facebookresearch/sparsh](https://github.com/facebookresearch/sparsh) | `LICENSE.md` = **CC BY-NC 4.0** (GitHub API는 NOASSERTION) | 225 (포크 31) | 2025-02-27 (커밋 `fee6a05`) | **archived = true**. 얕은 복제 `repo/`, 정적 읽기만 |
| 후속 코드 (Sparsh-X, Sparsh-Skin) | [facebookresearch/sparsh-multisensory-touch](https://github.com/facebookresearch/sparsh-multisensory-touch) | `LICENSE.md` = CC BY-NC 4.0 (텍스트 확인) | 55 | 2025-08-20 (커밋 `219ee2f`) | 활성 아님. 이 논문과 별개 (복제 안 함) |
| 프로젝트 페이지 | [sparsh-ssl.github.io](https://sparsh-ssl.github.io/) | 해당 없음 | — | — | 200 응답. BibTeX·영상 링크 |
| 사전학습 가중치 | [huggingface.co/facebook/sparsh](https://huggingface.co/facebook/sparsh) | 모델 카드 license 없음(API `None`) | — | 2024-10-21 | DINO ViT-B `.ckpt` 1.78 GB, `.safetensors` 345 MB (다운로드 안 함) |
| 데이터 | README 링크 (Hugging Face `facebook/touch-slide` 등, Touch-and-Go, ObjectFolder-Real, Feeling of Success, Clothing) | 데이터셋마다 다름 | — | — | 다운로드 안 함 |

- 공식 저장소의 `environment.yml`과 `setup.py`는 CUDA 전제 학습 환경입니다. 설치와 실행은 하지 않았습니다.
- 공식 저장소는 `dino`, `ijepa`, `jepa`, `mae` 원본 저장소에 기반합니다(부록 C 인용과 일치).
- 저장소 README의 인용 항목은 논문과 같이 "8th Annual Conference on Robot Learning, 2024"로 적혀 있습니다.

## 5. 라이선스 함의

- 공식 코드와 가중치 모두 **비상업(NC)** 조건입니다. 저장소의 `LICENSE.md`는 CC BY-NC 4.0 전문입니다. 연구·비영리 사용은 가능하지만 상업 제품에 넣으면 안 됩니다.
- 논문 PDF에는 저작권 문구가 없지만(`pdftotext` 검색 결과), arXiv 라이선스는 **nonexclusive-distrib 1.0**으로 CC BY가 아닙니다. 공개본에 논문 그림을 넣을 수 없습니다(권리 판정 절차는 `04` §1).
