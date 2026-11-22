# 02. 선행연구, 공식 저장소, 이 저장소의 관련 분석

## 1. 계보와 가까운 대안

DexJoCo가 다루는 질문은 "손가락이 있는 로봇의 과제 중심 평가와 시연 수집"입니다. 가까운 흐름은 세 갈래입니다.

**(a) 손 벤치마크(강화학습 중심).** Adroit[13], DexArt[15], Bi-DexHands[19]는 주로 손 안에서 돌리기나 강화학습 평가를 합니다. 논문은 이들이 기능적 상호작용이 적다고 봅니다(§II). 이 계열의 원 논문은 이 저장소에 분석이 없습니다.

**(b) 시연으로 다중 과제 학습과 평가하는 벤치마크(매니퓰레이터-그리퍼 중심).** LIBERO[16], RoboTwin 2.0[17], CALVIN[14]. DexJoCo는 손을 다루므로 같은 평가 틀을 손 설정에 옮긴 것입니다.

**(c) 손 시연 생성.** DexMimicGen[18](소수 시연 + MimicGen), DexCap(논문 참고문헌)과 Rokoko·VR 계열 수집 장치(§II). DexJoCo는 Rokoko 장갑과 Vive 트래커를 씁니다. 직접 비교는 하지 않았습니다.

**가까운 대안 정책:** π0.5[6](VLA, LoRA 미세조정), GR00T N1[7], ACT, Diffusion Policy[DP-C·DP-T]. 논문 기준선이 모두 이 계열입니다. 이 계열의 다른 분석은 이 저장소의 별도 폴더에 있습니다.

**기술적 연결:** GeoRT 재타깃팅은 Meta의 `facebookresearch/GeoRT`에서 가져왔습니다(§III-B, 참고문헌 [41]). 비상업 라이선스이므로 재사용 조건을 04에서 다룹니다.

## 2. GitHub 및 공식 자료

| 항목 | 주소 | 라이선스 | 별 | 마지막 push (API) | 클론 | 메모 |
|---|---|---|---|---|---|---|
| **공식 코드** | https://github.com/brave-eai/dexjoco | MIT (`LICENSE`, API spdx MIT) | 289 | 2026-11-11 | `repo/` 얕은 복제, HEAD `8d23b0f` (2026-05-29, "Add LeRobot and Zarr data conversion toolkit (#17)") | 저장소 생성 2026-05-07. 브랜치 `main`, `feat/lerobot`. 포크 35 |
| 프로젝트 페이지 | https://dexjoco.github.io/ | — | — | — | 정적 조회 | 논문 제목·저자·초록, 데모 영상, 데이터 링크. BibTeX "Coming soon". 학회 표기 없음 |
| 데이터 (LeRobot) | https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-LeRobot | 확인 안 함 | — | — | 다운로드 안 함 | 논문 1.1K 시연의 변환본(README 링크) |
| 데이터 (Raw) | https://huggingface.co/datasets/DexJoCo/DexJoCo-Datasets-Raw | 확인 안 함 | — | — | 다운로드 안 함 | 재생용 원본(README) |
| 가중치 | https://huggingface.co/DexJoCo/DexJoCo-Pi05 | 확인 안 함 | — | — | 다운로드 안 함 | π0.5 체크포인트(README) |
| DexJoCo2 (별도 데이터셋) | https://huggingface.co/datasets/DexJoCo/DexJoCo2 · https://dexjoco.github.io (DexJoCo2 탭) | 확인 안 함 | — | — | 안 함 | 촉각 데이터셋, Isaac Sim, 2,200 궤적, 10과제. **이 논문의 MuJoCo 벤치마크와 다른 자산** |
| arXiv | https://arxiv.org/abs/2605.16257 | 비독점 배포 (nonexclusive-distrib 1.0) | — | — | PDF만 | v1 2026-05-15, v2 2026-09-22 |

**기타 코드 의존(공식 저장소가 포함 또는 참조):**

| 항목 | 주소 | 라이선스 | 별 | 마지막 push | 저장소 내 위치 |
|---|---|---|---|---|---|
| openpi | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 14,729 | 2026-08-24 | `repo/openpi/` (π0.5 학습·평가) |
| MuJoCo Menagerie | https://github.com/google-deepmind/mujoco_menagerie | NOASSERTION (자산별 라이선스) | 4,400 | 2026-11-10 | 자산 출처 (Franka, Allegro 등) |
| GeoRT | https://github.com/facebookresearch/GeoRT | 비상업 (DexJoCo README 표기) | 확인 안 함 | 확인 안 함 | `repo/teleoperation/GeoRT`(서브모듈, 체크포인트 포함) |
| diffusion_policy | https://github.com/real-stanford/diffusion_policy | MIT (업스트림, 확인 안 함) | — | — | `dexjoco-data-converter/src/3rd/` (서브모듈) |

별·push 날짜는 2026-11-22 GitHub API 조회값입니다. 공식 저장소의 `pushed_at`(11-11)은 기본 브랜치 HEAD 날짜(05-29)보다 늦습니다. 기본 브랜치가 아닌 `feat/lerobot` 등에서 올라온 것으로 보이지만 확인하지 않았습니다.

## 4. 이 논문의 위치 (요약)

- 새로운 것: 손 + 매니퓰레이터 설정의 기능 중심 11개 과제, Rokoko·Vive·GeoRT 기반 저비용 수집, 시각·동역학 무작위화를 재생으로 만드는 방식.
- 새롭지 않은 것: 기본 장면은 RoboSuite 구성을 따르고 자산은 RoboCasa, PartNet-Mobility를 씁니다(§III-C). 정책 평가 틀은 openpi와 LeRobot 비동기 추론을 씁니다(공식 코드 `dexjoco_openpi_client/`, `docs/reference_code/`). 재생으로 시각 변형을 만드는 방식은 논문이 주장하는 기여이지만 방법 자체는 간단합니다.
- 아직 없는 것: 실로봇 결과, 여러 로봇에서의 교차 평가, 3시드 이상 통계.
