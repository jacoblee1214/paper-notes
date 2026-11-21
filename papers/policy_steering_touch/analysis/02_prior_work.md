# 02. 선행연구, 공식 저장소, 관련 분석

## 1. 선행연구 계보

ViTaL은 세 줄기를 잇습니다. 확산·흐름 정책의 **추론 시 steering**, **촉각을 쓰는 정책과 촉각 표현**, **시각·촉각 세계 모델**입니다(Sec. 2).

**(a) 추론 시 steering.** 논문은 세 계열로 나눕니다(Sec. 2 첫 문단).

- 샘플 뒤 검증(sample-then-verify): RoboMonkey [4], 사람 개입 steering [5], 예측 세계 모델 [6], 불확실성 기반 steering [8]. 같은 저자(Bajcsy 연구실)의 [1] "From foresight to forethought" (RSS 2025)은 VLM을 루프에 넣은 잠재 정렬 steering입니다.
- 잡음 공간 최적화: 잠재 공간 강화학습 steering [7].
- 분류기 가이던스: DynaGuide [2], Latent Policy Barrier [3]. 논문은 이들이 "전역 문맥은 주지만 세밀한 접촉을 놓친다"고 씁니다.

ViTaL의 차이는 샘플링을 두 시간 척도로 나누고, 촉각 보상을 확산 편집에 넣는 점입니다. 시각 가이던스에는 분류기 방식을 쓰지 않습니다. VLM, 디코더, 순환 세계 모델을 미분하기 어렵기 때문입니다(Sec. 4.4).

**(b) 촉각 steering과 촉각 정책.** 가장 가까운 대안은 **TouchGuide** [12] (arXiv:2601.20239)입니다. 저자들은 TouchGuide가 대조 학습으로 행동 공간 검증기를 배우고, 양성 데모가 필요하며, 동역학을 명시하지 않는다고 씁니다(Sec. 2). Reactive Diffusion Policy [13] (RSS 2025)와 Manipulation by Feel [16] (ICRA 2019)은 촉각을 정책 입력이나 저수준 제어에 씁니다. Vi-TacMan [15] (ICRA 2026)은 시각과 촉각을 같이 쓰는 조작입니다.

**(c) 촉각 표현과 세계 모델.** GelSight 계열 데이터로 학습된 표현은 [17–24]에 있습니다. 이 논문이 쓰는 **AnyTouch 2** [21]는 촉각 비디오 표현이고, **Sparsh** [22]는 자기지도 촉각 표현입니다. 시각 세계 모델로는 DINO-WM [49]을 따릅니다. 촉각 세계 모델은 Visuo-Tactile World Models [14] (arXiv:2602.06001), OmniVTA [46], VTAM [47]과 관련됩니다.

**(d) 보상 모델과 확산 편집.** 시각 보상은 ROBOMETER [51] (VLM 기반 일반 보상 모델)을 쓰고, 텍스트 임베딩은 CLIP [53]입니다. 확산 편집은 SDEdit [11]을 따르고, 학습 손실 설계는 RAE [56]를 따릅니다. 보상 정확도 평가는 Tian 등 [54]의 방식을 따릅니다.

**(e) 기본 정책.** Diffusion Policy [9] (IJRR 2024)의 U-Net 구조와 학습 설정입니다. π0 [10]은 비교 대상으로만 언급합니다.

**새로움 주장에 대한 메모.** 저자들은 "언어 조건부 촉각 보상의 첫 일반 목적 사례"라고 씁니다(Sec. 4.3). 이 문헌 검색 범위는 이 분석에서 다루지 않았으므로 확인하지 못했습니다.

## 2. 공식 저장소와 관련 자료 (2026-11-22 확인)

| 저장소·페이지 | 관계 | 라이선스 | 별 | 최근 push | 상태 |
|---|---|---|---|---|---|
| [yilin-wu98/vital](https://github.com/yilin-wu98/vital) | **공식 코드** (논문 README의 Code 링크, 저자 계정) | 구성요소별 (`LICENSE.md`): latent-tactile MIT, anytouch-reward MIT, liquid-tactile-control Apache-2.0, 정책 백본 MIT. GitHub API는 NOASSERTION | 1 | 2026-10-27 (HEAD `147bdb9`, 커밋 날짜 2026-10-27) | 얕은 복제(`repo/`), **정적 읽기만**. 실행, 설치, import 안 함 |
| [corl2026vital/corl2026vital.github.io](https://github.com/corl2026vital/corl2026vital.github.io) | 프로젝트 페이지에 링크된 저장소 (사이트 소스로 보임. 이 저장소의 코드는 아님) | 없음 (API null) | 0 | 2026-10-02 | 메타데이터만 확인. 복제 안 함 |
| [프로젝트 페이지](https://yilin-wu98.github.io/vital_website/) | 공식 웹사이트 (영상, 그리퍼 마운트 설계 링크) | — | — | — | 열람 (HTML). 학회 표기 없음 |
| [Hugging Face: yilin-wu/ViTaL-datasets](https://huggingface.co/datasets/yilin-wu/ViTaL-datasets) | 공식 데이터셋 (에피소드 980개, 716.5 GB, 압축 HDF5) | CC BY 4.0 (`DATASET_RELEASE.json`, README) | — | — | **받지 않음**. 용량과 정책상 다운로드 금지 (README와 DATASETS.md만 읽음) |
| 외부 가중치 (ROBOMETER, DINOv3, AnyTouch2, CLIP) | 보상·인코더의 사전학습 가중치 | 각 원 프로젝트 조건 | — | — | 받지 않음 |
| 공식 체크포인트 (세계 모델 67k, 디코더) | README가 "별도 제공"이라고 씀 | 공개 여부 불명 (README에 위치 없음) | — | — | 받지 않음. 릴리스에 없음 |

- 공식 코드의 `pyproject.toml`과 `LICENSE.md`에 쓰인 라이선스는 구성요소마다 다릅니다. 저장소 전체를 한 라이선스로 취급하면 안 됩니다.
- 데이터셋 README는 "release was selected from the research working trees, including local changes"라고 씁니다. 공개 릴리스와 논문 실험 코드가 완전히 같은지는 확인할 수 없습니다.
- 학회 확인은 `conferences/corl2026/papers.json`의 공식 채택 목록으로 했습니다(포스터, 2026-11-10 화요일 오후 5시 세션, 포스터 번호 160).

## 4. 참고 문헌 중 핵심 항목

- [1] Y. Wu, R. Tian, G. Swamy, A. Bajcsy, *From Foresight to Forethought*, RSS 2025.
- [2] M. Du, S. Song, *DynaGuide*, NeurIPS 2025.
- [8] J. Yuan, Y. Wu, A. Bajcsy, *When to Act, Ask, or Learn*, RSS 2026.
- [9] C. Chi et al., *Diffusion Policy*, IJRR 2024.
- [11] C. Meng et al., *SDEdit*, arXiv:2108.01073.
- [12] Z. Zhang et al., *TouchGuide*, arXiv:2601.20239.
- [13] H. Xue et al., *Reactive Diffusion Policy*, RSS 2025.
- [14] C. Higuera et al., *Visuo-Tactile World Models*, arXiv:2602.06001.
- [21] R. Feng et al., *AnyTouch 2*, ICLR 2026.
- [48] O. Siméoni et al., *DINOv3*, arXiv:2508.10104.
- [49] G. Zhou et al., *DINO-WM*, ICML 2025.
- [51] A. Liang et al., *Robometer*, 2026.
- [53] A. Radford et al., *CLIP*, ICML 2021.
- [54] R. Tian, Y. Wu, A. Bajcsy, *Position: Good Embodied Reward Models Need Bad Behavior Data*, 2026.
- [56] B. Zheng et al., *Diffusion Transformers with Representation Autoencoders*, arXiv:2510.11690.
