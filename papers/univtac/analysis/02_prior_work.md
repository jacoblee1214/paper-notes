# 02. 선행 연구와 공식 자료

이 문서는 UniVTAC의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 공식 코드와 페이지가 무엇인지 정리합니다. 번호 [n]은 `paper/paper.txt`의 참고문헌 번호입니다.

## 1. 구성 요소별 계보

| 구성 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 시뮬레이션 기반 | TacEx [35] (libuipc [13, 14] + Isaac Sim [36]) | 플랫폼의 바탕. 센서 여러 개와 조작 API를 추가했다고 논문이 밝힘 (II-A) |
| 모션 계획 | cuRobo [40] | Move, Place에 사용 (III-A-2) |
| IPC 접촉 시뮬레이션 | SapienIPC [5, 20], Taccel [24] | 마커 변형 모델링의 영감 (II-A) |
| 재구성 기반 촉각 표현 | Marker-Embedded GAN [18], UniT [46] | 픽셀 충실도 위주라 접촉 물리를 놓친다고 비판 (II-B) |
| 기하 기반 촉각 표현 | RDP [47], [5] | 마커 좌표를 쓰는 대안. 원문 입력에서 텍스처를 버린다고 비판 (II-B) |
| 대조 학습 촉각 표현 | VITaL [11] | 실험 기준선. CLIP식 전역 정렬 (II-B, IV-A) |
| 14개 기하 원형 | [18]의 프로토콜을 따름 | 데이터 생성 설계 근거 (III-A-4) |
| 정책 아키텍처 | ACT [52] | 모든 비교의 기준 정책 (IV-A) |
| 벤치마크 설계 방식 | RoboTwin 2.0 [3], "RoboTwin 패러다임" | 과제 구성과 평가 파이프라인 (III-C) |
| 전문가 궤적 생성 | MimicGen [30], DexMimicGen [16] | 비교 대상. 접촉 다양성 부족을 비판 (II-C) |

## 2. 가까운 대안

- **시뮬레이션 촉각 플랫폼:** 이 논문 이전과 같은 시기에 나온 대안은 Tacchi [6](MPM), DiffTactile [39](MPM, 미분 가능), TacFlex [51](FEM)입니다. 논문은 이들이 센서 설계나 정책 평가 프로토콜에 덜 맞춘다고 봅니다. 이 주장은 근거 자료로 확인하지 않았습니다.

## 4. 공식 자료 표

조회일 2026-11-22. GitHub 값은 `gh repo view`로 받았습니다. 라이선스는 저장소의 `LICENSE` 파일로 확인했습니다.

| 자료 | 위치 | 라이선스 | 최근 push | 별 | 커밋 | 비고 |
|---|---|---|---|---|---|---|
| 공식 코드 (메인 브랜치, Isaac Sim 4.5) | [github.com/univtac/UniVTAC](https://github.com/univtac/UniVTAC) | `LICENSE` 파일은 Apache-2.0. README는 "MIT"라고 적음 (불일치, 04 D18) | 2026-09-07 | 196 | `0dafa10` (2026-09-07, "fix readme"), `repo/`에 얕은 복제 | 정적 읽기만 함. 설치·실행 안 함 |
| 공식 코드 (isaac51 브랜치, Isaac Sim 5.1) | 같은 저장소의 `isaac51` 브랜치 | 같음 | 미확인 | — | 미확인 | 이 분석은 메인만 받음. README는 5.1 데이터가 5배 빠르다고 적음 |
| 프로젝트 페이지 | [univtac.github.io](https://univtac.github.io/) | — | — | — | — | 학회 표기 없음. README에는 "CoRL 2026 채택" |
| 데이터 (Isaac Sim 4.5, 5.1, 접촉 사전학습, 체크포인트) | Hugging Face `byml/UniVTAC`, ModelScope `byml2024/UniVTAC` | 미확인 | — | — | — | 받지 않음. 체크포인트는 4.5 데이터에만 맞음 (README) |
| arXiv 페이지 | [arxiv.org/abs/2602.10093](https://arxiv.org/abs/2602.10093) | arXiv 비독점 배포 1.0 ("view license") | — | — | — | 제출 2026-02-10, v1 하나 |
| 동봉 TacEx | `repo/third_party/TacEx` | 미확인 (LICENSE 파일 없음) | — | — | — | 복사본 |
| 동봉 ACT | `repo/policy/ACT` | MIT (폴더 LICENSE) | — | — | — | DETR 하위 디렉터리는 Apache-2.0 |
| 동봉 ViTAL | `repo/policy/ViTAL` | MIT (폴더 LICENSE) | — | — | — | README는 케이블 꽂기 논문(arXiv:2403.11898) 기준 |
| 동봉 smolvla, smolvla_vision (lerobot) | `repo/policy/smolvla*` | 미확인 | — | — | — | 이 분석은 열지 않음 |

**구현 가능성과 관련된 자료:**
- 저장소는 `scripts/install.sh`로 Isaac Sim 4.5.0, Isaac Lab 2.1.1, TacEx, cuRobo를 설치하게 합니다. 이 맥에는 CUDA가 없어서 설치와 실행이 불가능합니다 (04 §3).
- 데이터는 HF에서 받는 구조입니다 (`data/download.sh`). 용량은 이 분석에서 확인하지 않았습니다.

## 5. 저장소 밖 용어 정리

- **VITaL** (논문 [11]): 대조 학습으로 촉각과 시각 인코더를 사전학습하는 방법. 공개 코드의 `policy/ViTAL`은 원 논문의 케이블 꽂기 저장소에서 가져온 것으로 보이고, 기준선 설정이 UniVTAC의 ACT와 다릅니다 (04 D14).
- **TacEx** [35]: 이 논문이 기반으로 삼은 시뮬레이터 확장. 공식 저장소에 복사본이 있습니다.
- **Scratch**: 촉각 인코더를 무작위 초기화로 시작하는 기준선(표 II). 공개 코드에서는 `from_scrach.yml`로 구현되어 있습니다(`tactile_ckpt: null`).
