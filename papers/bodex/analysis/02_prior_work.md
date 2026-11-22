# 02. 선행연구, 구성 요소, 공식 저장소

## 1. 계보

| 흐름 | 대표 연구 | BODex와의 관계 |
|---|---|---|
| 해석적 파지 품질 (Q1, 힘-폐쇄) | Ferrari & Canny (1992, [15]), GraspIt! [16] | 기준선이 아니라 평가 지표의 뿌리. Q1은 미분이 어려워 BODex 비교에서 합성에 쓰지 않음 (§V-C-2) |
| 미분 가능한 힘-폐쇄 에너지 | DFC [3], DexGraspNet [4] | 접촉 힘 균등과 마찰 없음 가정. BODex는 이 가정을 QP로 대체 |
| 미분 가능한 시뮬 | Grasp'D [17], Fast-Grasp'D [18] | 시뮬레이터로 기울기를 얻음. BODex는 기하 기반 |
| 새 에너지 함수 | FRoGGeR [5], SpringGrasp [6], TDG [19] | 기준선 (FRoGGeR, SpringGrasp 비교). TDG는 에너지 비교에서 기준선 |
| QP 기반 에너지 (사후 처리) | Wu, Guo, Liu (2022, [20]) | 가장 가까운 선행. 생성 결과의 후처리에 씀. BODex는 대규모 생성에 씀. 논문은 QP_baseline으로 비교 (β = 0 특수형) |
| 학습 기반 | ISAGrasp [22], GraspTTA [21], DP3 [33], UniDexGrasp [23] | 데이터셋 비교의 학습기. UniDexGrasp 생성 모듈 (GraspGlow)만 씀 |
| 강화학습 | UniDexGrasp++ [27] | 본문에서 Sim2Real이 어렵다고만 언급 |
| 대규모 데이터셋 | DexGraspNet [4] (ICRA 2023), GenDexGrasp [12], RealDex [13] | 생성 시뮬 데이터셋 기준. BODex는 DexGraspNet 자산을 재사용 |

BODex 본문 Related Work의 주장: 해석적 방법은 완전한 형상이 필요해 실시간으로는 어렵다, 경사 기반 방법은 느리거나 가정이 강하다 (§II).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 역할 |
|---|---|---|
| 손끝 접촉 구, 충돌 질의 | cuRobo [7] | 병렬 충돌·FK |
| 배치 QP | ReLU-QP [8] | GPU 배치 ADMM 풀이기. 저장소에 PyTorch 구현 있음 |
| 비교 QP 풀이기 | ProxQP [9], OSQP [10] | 10배 비교 대상 |
| 최근접점, GJK | Gilbert et al. (1988, [29]), Coal [30] | fine 단계 메쉬 질의 |
| 시뮬 자산 | MuJoCo Menagerie [31] | 손 모델 |
| 마찰 설정 | Xydas & Kao (1999, [32]) | 접선·비틀림 마찰 (0.6, 0.02) |
| 물체 자산 | DexGraspNet [4] | 2,397 물체 |
| 학습 백본 | 3D SparseConv [34] | 단일 시점 점군 |
| 부분 점군 마스크 | SAM [36] (본문 표기, 실제 인용은 SAM 2) | 실로봇 분할 |

## 4. 공식 저장소와 GitHub 표 (확인 2026-11-22)

| 저장소 | 역할 | 라이선스 | 별 | 마지막 push | 이 작업의 조치 |
|---|---|---|---|---|---|
| [JYChen18/BODex](https://github.com/JYChen18/BODex) | **공식 코드.** cuRobo 포크 기반 파지 합성 (`example_grasp/`, `src/curobo/rollout/cost/grasp_energy/qp.py`) | NVIDIA License (cuRobo 부분, `LICENSE`) + **CC BY-NC 4.0** (README "All other parts"). GitHub API는 NOASSERTION | 156 | 2026-04-21 (`06b9a3c`) | `repo/`에 얕은 클론 (`06b9a3c`, 2026-04-21). 정적 읽기만 |
| [JYChen18/DexGraspBench](https://github.com/JYChen18/DexGraspBench) | MuJoCo 벤치마크 (본문 §V 평가의 후속 코드). README: "main 브랜치는 논문과 설정이 조금 다름" | 없음 (API null) | 67 | 2025-09-21 | README만 읽음 (클론 안 함) |
| [JYChen18/DexLearn](https://github.com/JYChen18/DexLearn) | 학습 기반 파지 합성. BODex README의 "Learning code" 링크 (표 IV 학습 코드인지는 확인하지 않음) | 없음 (API null) | 51 | 2026-10-19 | 메타데이터만 |
| [JYChen18/Dexonomy](https://github.com/JYChen18/Dexonomy) | 후속 연구 (RSS 2025) | 없음 (API null) | 170 | 2026-03-26 | 메타데이터만 |
| [pku-epic.github.io/BODex](https://pku-epic.github.io/BODex/) | 프로젝트 페이지. ICRA 2025 표기. Paper·Code·Benchmark·Dataset·Learning 링크 | 페이지에 라이선스 문구 없음 | — | — | 텍스트 읽음 |

**저장소 라이선스 해석.** `repo/README.md` 97–104행은 "cuRobo 코드는 NVIDIA 라이선스, 그 밖의 모든 부분은 CC BY-NC 4.0"이라고 씁니다. `LICENSE` 파일은 NVIDIA License 전문입니다 (2023, "non-commercially" 사용 허용). 따라서 **코드와 데이터는 비상업 조건**입니다. 논문 자체는 arXiv abs 페이지에서 CC BY 4.0입니다 (`01` §0, `README.md` 라이선스 절).

**데이터셋.** Hugging Face `JiayiChenPKU/BODex`, 카드 license `cc-by-nc-4.0`, 마지막 수정 2025-05-15. 파일 크기는 HEAD 요청의 Content-Length로만 확인했습니다 (받지 않음):

| 파일 | 크기 |
|---|---|
| `object_assets/DGN_2k_processed.tar.gz` | 0.55 GB |
| `object_assets/DGN_2k_vision.tar.gz` | 4.37 GB |
| `object_assets/DGN_obj_raw.zip` | 1.31 GB |
| `synthesized_grasps/allegro.tar.gz` | 0.32 GB |
| `synthesized_grasps/leap.tar.gz` | 0.32 GB |
| `synthesized_grasps/shadow.tar.gz` | 0.37 GB |
| `synthesized_grasps/ur10e_shadow.zip` | 10.77 GB |

합계 약 18 GB. 합성 파지 세 손과 테이블 버전 (UR10e+Shadow) 중 테이블 버전이 대부분을 차지합니다. 본문 표 I의 파지 수 (4.09 M, 3.08 M)와 맞는지는 데이터를 받아야 확인합니다.

## 5. 이 저장소의 `repo/` 클론에 대한 메모

- 얕은 클론 (`--depth 1`), 7.1 MB, 파일 734개. 에셋 대부분은 Git LFS 포인터일 수 있어 실제 객체 파일은 없습니다.
- 공식 README의 설치 명령 (conda, pytorch 2.2.2, coal 3.0.1, numpy 1.26.4, `src/curobo/geom/cpp` 빌드)은 CUDA 12.1을 전제합니다. 이 맥에서는 설치하지 않았습니다.
- 실행하지 않았습니다. `example_grasp/`와 `src/`는 읽기만 했습니다 (`04` 참조).
