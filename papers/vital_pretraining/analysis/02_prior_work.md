# 02. 선행연구와 관련 저장소

## 1. 계보

- **촉각 정책 학습.** 케이블과 와이어를 촉각으로 다루는 고전 제어와 휴리스틱 [6, 12, 13], 촉각 시각 강화학습 [14–16], 최근접 이웃 모방 학습 VINN [17]과 그 촉각 확장 [8, 18]이 있습니다. VINN은 적은 데이터로 잘 되지만 일반화가 약해 ACT 같은 직접 행동 예측기보다 낮다고 논문은 봅니다(§II-A).
- **촉각 대비 사전학습.** 시각과 촉각을 같은 잠재 공간에 맞추는 자기지도 학습이 옷감 결함 탐지 [25], 소수 샷 분류 [26, 27], 병 뚜껑 덮기 강화학습 [28]에 쓰였습니다. 논문은 "시각 전용 정책을 촉각 데이터로 사전학습한 사례는 알려진 바 없다"고 씁니다(§II-B).
- **모방 학습의 뼈대.** ACT [10]: CVAE 기반 행동 청크와 시간 앙상블. Diffusion Policy [11]: DDPM 행동 예측, 학습과 추론 스케줄 분리. VITaL은 두 뼈대를 그대로 쓰고 인코더와 위치 표현만 바꿉니다.
- **CLIP 계열 대비 학습.** Radford 외 [24]의 CLIP 손실을 시간 간격 샘플링과 위치 조건으로 바꿨습니다.

## 2. 가까운 대안과 구분점

| 연구 (논문 참고문헌 제목 기준) | 사전학습 방식 | 이 논문과의 차이 (본문이 설명한 범위에서) |
|---|---|---|
| Guzey 외 [8], *Dexterity from Touch* | 로봇 놀이 데이터 기반 촉각 자기지도 | 본 논문은 시각 전용 배포 정책을 목표로 함 |
| Liu 외 [28], *Masked visual-tactile pre-training* | 마스크 기반 시각-촉각 사전학습 (시뮬레이션) | 본 논문은 실로봇 과제와 시각 전용 배포를 다룸 |
| Sferrazza 외 [14], *The power of the senses* | 마스크 다중모달 학습 | 본 논문은 대비 손실과 시간 간격 샘플링을 씀 |
| **VITaL (본 논문)** | 과제별 시간 간격 대비 사전학습 | 학습 때만 촉각을 쓰고, 추론은 시각 전용 가능 |


## 4. 공식 코드와 프로젝트 페이지

| 항목 | 위치 | 라이선스 | 최근 활동 | 기록 |
|---|---|---|---|---|
| 프로젝트 페이지 | <https://sites.google.com/andrew.cmu.edu/visuo-tactile-pretraining> | (해당 없음) | 확인 불가 | 데모 영상과 코드 링크 |
| 공식 코드 | <https://github.com/Abraham190137/TactileACT> | MIT (`LICENSE`, 저작권 2024 Abraham George) | 마지막 push 2024-08-23, 별 48, 포크 3 (gh API, 2026-11-22 조회) | 얕은 복제 `repo/` (커밋 `5a533f76f15c`, 2024-08-23, 크기 약 1.0 MB, 파일 55개). 정적 읽기만 |
| arXiv 원문 | `paper/paper.pdf` (v2, 5.6 MB) | CC BY-NC-SA 4.0 (arXiv abs의 license 링크, 아래 `README.md` 라이선스 절) | v2 (2024-09-26) | `paper/paper.txt` |

- 공식 코드의 마지막 커밋(2024-08-23)은 논문 v2(2024-09-26)보다 앞섭니다. 따라서 코드는 v1 시기 또는 중간 개발 상태일 가능성이 있습니다. 대조는 v2 본문 기준으로 하되, 불일치는 "코드가 논문 최종본과 다를 수 있음"으로 기록합니다(`04_code_verification.md`).
- 데이터 셋, 학습된 체크포인트, 로봇 제어 스택(Franka 컨트롤러, GelSight 드라이버)은 저장소에 없습니다. README는 `data/clip_models/*.pth`와 `data/data_dir` 경로만 가정합니다.
- 실로봇 이외에 공개된 정량 재현 자료는 없습니다.

## 5. 관련 연구 목록 (논문 참고문헌 중 본문과 직접 관련된 것)

- [8] I. Guzey 외, *Dexterity from Touch*, arXiv:2303.12076 (2023).
- [10] T. Z. Zhao 외, *Learning fine-grained bimanual manipulation with low-cost hardware* (ACT), arXiv:2304.13705 (2023).
- [11] C. Chi 외, *Diffusion Policy*, arXiv:2303.04137 (2023).
- [14] C. Sferrazza 외, *The power of the senses*, 2023.
- [17] J. Pari 외, *The surprising effectiveness of representation learning for visual imitation* (VINN), arXiv:2112.01511 (2021).
- [24] A. Radford 외, *Learning transferable visual models from natural language supervision* (CLIP), ICML 2021.
- [28] Q. Liu 외, *Masked visual-tactile pre-training for robot manipulation*, ICRA 2024, pp. 13859–13875.
- [38] A. George, A. Bartsch, A. B. Farimani, *OpenVR: Teleoperation for manipulation*, arXiv:2305.09765 (2023).
- [33] N. M. Shafiullah 외, *Behavior transformers*, NeurIPS 2022.

참고문헌 번호는 v2 본문 기준입니다.
