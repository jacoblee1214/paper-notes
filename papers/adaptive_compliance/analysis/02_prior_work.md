# 02. 선행 연구와 공개 자료

## 1. 공식 코드와 프로젝트 페이지

| 항목 | 위치 | 라이선스 | 별 | 마지막 push | 확인 |
|---|---|---|---|---|---|
| 공식 구현 (ACP 학습, 데이터 처리, 제어 파이프라인) | `github.com/yifan-hou/adaptive_compliance_policy` | MIT | 145 | 2026-04-18 (커밋 `74d68ec` "Updated bibtex") | 얕은 복제 `repo/` (정적 읽기만, `04_code_verification.md`) |
| 프로젝트 페이지 | `adaptive-compliance.github.io` | 웹 페이지 | — | — | 확인. 논문 PDF, 영상, BibTeX |
| 데이터와 체크포인트 | `real.stanford.edu/adaptive-compliance/` (README의 wget 주소) | 명시 없음 | — | — | 받지 않음 (데이터 크기와 라이선스 미확인) |
| 하드웨어 인터페이스 | `github.com/yifan-hou/hardware_interfaces` (저장소 안내) | 확인 안 함 | — | — | 복제하지 않음 |
| 기반 저장소 | `real-stanford/universal_manipulation_interface` (UMI) | 확인 안 함 | — | — | 확인만 함 (README에서 기반이라고 명시) |

공식 코드는 Ubuntu 22.04, mamba, 실로봇 C++ 제어기를 전제로 합니다. 재현용 설치는 하지 않았습니다.

## 2. 버전과 학회 표기

- arXiv 버전: **v1 2024-10-12, v2 2025-03-07** (최신). 차이는 `01_paper_analysis.md` §5.
- 학회 표기 근거: 프로젝트 페이지 BibTeX가 `2025 IEEE International Conference on Robotics and Automation (ICRA)`라고 적고, 웹 검색 결과 한 곳도 ICRA 2025 페이지 4829–4836을 제시했습니다. 페이지 상단 배너에는 "Best paper award at ICRA 2025 Workshop on Contact-rich Manipulation"이 있습니다. **워크숍 수상은 본학회 발표와 다릅니다.**
- Crossref에 ICRA 2025 학회 논문(DOI 10.1109/icra55743.2025.11128452, 2025-05-19)으로 등록되어 있어 **ICRA 2025**로 적습니다(메인 세션 확인, 2026-11-21).

## 3. 선행 연구 계보

논문 §II(관련 연구)를 기준으로 정리했습니다. 인용 번호는 v2 참고문헌 번호입니다.

| 축 | 논문 인용 (v2 번호) | 논문에서 한 역할 | 이 논문과의 차이 |
|---|---|---|---|
| 접촉이 많은 조작의 능동 컴플라이언스 | Lozano-Pérez [31], Uchiyama [41], Sawasaki [40], Hou 2018 [15] | 이론 배경 | 접촉 모델이 필요함 |
| 기하와 마찰을 알 때 최적 강성 | Hou & Mason 2019 [18], 2021 [17], Hou et al. shared grasping [16] | 최적 강성 계산 근거 | 모델 기반 |
| 강화학습으로 컴플라이언스 | Kalakrishnan [19], Beltran-Hernandez [2, 3], Chang [5], Martín-Martín [33], Noseworthy [36], Portela [38] | 대안 | 시나리오마다 재학습 |
| 사람 강성 추정 (공분산) | Calinon [4], Duan [11, 12], Kronander & Billard 2012 [25] | 대안 | 같은 동작 반복 필요 |
| 단일 데모 강성 추정 | Grioli & Bicchi [13] | 가장 가까운 대안 | 사람 질량과 감쇠를 알아야 함 |
| 균일 강성을 사용자가 지정 | Kronander & Billard 2013 [24], Kamijo et al. [20] | 대안 | 공간 변화 없음 |
| 휴리스틱 단계 분할 | ForceMimic [29] (논문에서 "동시 연구"로 표기) | 대안 | 단계 수작업 |
| 힘 입력 시각 정책 (기준선 포함) | Compliant policy [26] (Fig. 2b), Stiff 기준선은 확산 정책 [6, 7] | 기준선 | 강성 출력 없음 |
| 힘 시계열 인코더 | 인과 합성곱 [37], TCN [2], LSTM [9], VAE [1], 자기주의 [22, 23] | 인코더 선택지 | 논문 §II가 인용한 대로 옮긴 것 |
| 확산 정책 | Chi et al. [6] RSS 2023, [7] IJRR 2024 | 정책 구조의 기반 | 위치만 출력 |
| 시각 특징 | CLIP ViT [10], CoordConv [28] | 구성 요소 | 구성 요소로만 사용 |

참고: ForceMimic(arXiv:2410.07554)은 v2에서 명시적으로 인용된 것으로 보입니다. Martín-Martín et al. (IROS 2019)은 v2 참고문헌 [33]입니다. 각 번호와 저자 대응은 v2 PDF 참고문헌 목록을 기준으로 옮겼습니다.

## 5. 자료 취급

- 저장소와 프로젝트 페이지는 외부 데이터로 취급했습니다. 실행과 설치는 하지 않았습니다.
- 공식 저장소의 README 안내 주소(데이터, 체크포인트)는 다운로드하지 않았습니다. 용량과 라이선스를 확인하지 못했기 때문입니다.
