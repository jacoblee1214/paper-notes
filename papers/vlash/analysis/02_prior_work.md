# 02. 선행연구와 관련 자료: VLASH

## 1. 계보

```
행동 청킹 (ACT, Zhao 외 2023)  ── 청크 단위 추론, 청크 실행 후 다음 추론(동기)
   │
   ├─ 비동기 추론 (SmolVLA, Shukor 외 2025): 새 청크가 준비되면 바로 교체 (naive async)
   │      └─ 예측-실행 구간 어긋남 → 불안정 (VLASH §III c)
   ├─ Real-Time Chunking (RTC, Black 외 2025, arXiv:2506.07339): 실행이 확정된 행동을 고정(freeze)하고 나머지를 inpainting
   │      └─ 추론 시간 증가(VLASH 표 II: L 48.4 ms 대 23.2 ms)
   ├─ Training-time RTC (Black 외, arXiv:2512.05964): 보정을 학습 시간으로 이동. VLASH가 "동시 연구"라 명시
   ├─ A2C2 (Sendai 외, arXiv:2509.23224): 보정 헤드 추가. 동시 연구
   ├─ Running VLAs at real-time speed (Ma 외, arXiv:2510.26742)
   └─ VLASH: 입력 상태를 실행 시점 상태로 교체 + 오프셋 증강 (구조 변경 없음)
행동 양자화: Figure AI Helix (2025), DemoSpeedup (Guo 외, arXiv:2506.05064)
```

**VLASH가 새로 한 것(저자 서술, §II b).** 모델 입력만 바꾸고 아키텍처와 추론 절차는 그대로 둡니다. 가까운 대안과의 차이는 다음과 같습니다.

| 방법 | 보정 방식 | 추가 추론 비용 | 구조 변경 |
|---|---|---|---|
| naive async | 없음 | 없음 | 없음 |
| RTC | 확정 행동 고정 + inpainting | 있음 (표 II: L 약 2배) | 없음 |
| training-time RTC | 학습 시 행동 조건화 | 없음(저자 서술) | 조건화 입력 추가 |
| A2C2 | 보정 헤드 | 있음 | 헤드 추가 |
| VLASH | 실행 시점 상태 입력 + 오프셋 증강 | 없음 | 없음(π0.5는 상태 투영 권장, `04` D4) |
| RDT-2, DynamicVLA | 작은 모델로 고주파 추론 | - | 모델 교체 |

**한계로 보이는 점(우리 판단).** 미래의 **환경**은 예측하지 않습니다(영상은 과거). 저자도 이를 인정합니다(§IV-A "While the future environment is still unknown"). 공식 이슈 #25도 같은 질문(핑퐁 공의 미래를 예측하지 않는데 왜 되는가)을 올렸고 답은 없습니다(2026-11-22 확인). 후속 연구 EagleVLA는 VLASH가 로봇 상태만 예측하므로 긴 지연에서 환경 변화를 놓친다고 주장합니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 모델 | π0.5 (Physical Intelligence, arXiv:2504.16054), π0, SmolVLA (arXiv:2506.01844), GR00T N1.6 | 코드 main은 π0와 π0.5만(`04` §1) |
| 오프셋 증강, 공유 관측 | 이 논문 | 공식 코드 주석에 "nano-lerobot's apply_async_offset" 언급 (`04` D5) |
| 어텐션 마스크 | 블록 희소(오프셋 분기 간 차단) | 코드는 dense 마스크 텐서를 직접 만듦(`vlash/policies/pi05/utils.py`) |
| Kinetix 설정 | Kinetix (ICLR 2025), RTC의 평가 설정 | `real-time-chunking-kinetix` 저장소(MIT) |
| LIBERO | Liu 외, arXiv:2306.03310 | MIT |
| 행동 양자화 | Helix, DemoSpeedup | 코드는 서브샘플링(`04` D3) |
| 학습 하이퍼파라미터 | openpi 기본값과 같음(AdamW β (0.9, 0.95), 가중치 감쇠 1e-10, 코사인 5e-5 → 2.5e-6, 워밍업 1,000) | 표 V |
| LoRA | PEFT 0.18.0 | 코드 전용 기능, 논문에 없음 |
| 로봇 실행 | LeRobot 0.4.1 | 코드 의존성 |

## 4. GitHub와 코드 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 마지막 푸시 | 비고 |
|---|---|---|---|---|
| **공식 코드 main** | https://github.com/mit-han-lab/vlash | Apache-2.0 | 별 525, 포크 34 / 2026-04-22 | `repo/` 얕은 clone, 커밋 `22cbabf`. π0·π0.5 학습, SO-101 추론, 지연 벤치마크. 열린 이슈 19건 |
| 공식 가지 `sim/libero` | 같은 저장소 | Apache-2.0 | 최신 `d2da223` (2026-04-22) | LIBERO 학습·평가(`vlash/eval_libero.py`, `libero_gym.py`, `libero-eval-scripts/run.sh`). 정적으로만 읽음 |
| 공식 가지 `sim/kinetix` | 같은 저장소 | Apache-2.0 | `fa1e65c` (2026-01-11) | Kinetix 코드는 서브모듈 `Sakits/vlash-kinetix`(MIT, 별 3, 2026-01-11, 받지 않음) |
| 공식 가지 `dev/gr00t` | 같은 저장소 | Apache-2.0 | `7a66b22` (2026-02-28) | GR00T N1.6 지원. 읽지 않음 |
| 프로젝트 페이지 | https://hanlab.mit.edu/projects/vlash | - | - | 데모, 논문, 코드, 슬라이드, 포스터 링크. **학회 표기 없음** (초록은 v1) |
| 체크포인트 | HF `mit-han-lab/vlash-pi05-libero-async5` (이슈 #22 답변에서 안내) | 미확인 | - | 받지 않음. 이슈 본문의 이름이 `vlash-libero-async5`로 달라 이름 불일치 |
| LeRobot | https://github.com/huggingface/lerobot | Apache-2.0 | 별 약 29,000 / 2026-11-22 | VLASH가 0.4.1에 고정 의존 |
| RTC Kinetix | https://github.com/Physical-Intelligence/real-time-chunking-kinetix | MIT | 별 636 / 2025-12-08 | 논문 Kinetix 설정의 출처 |
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT | 별 2,565 / 2025-03-15 | 평가 환경 |
| openpi | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 14,729 / 2026-08-24 | π0.5 원본 구현 |
| 제3자 포크 | https://github.com/Chung-I/vlash | Apache-2.0 | 별 0 / 2026-08-06 | 확인만 함 |

공식 이슈 중 재현에 중요한 것: #22(LIBERO 표 I 재현 어려움, MuJoCo 버전), #10(main에서 LIBERO 학습 불가, 상태 차원 오류), #18(Kinetix·LIBERO 평가 초안 PR), #21(π0.5 공유 관측에서 상태 처리 방식이 다름), #25(핑퐁 설정 질문, 카메라 수와 데이터 양 미공개), #23(SmolVLA 가중치 요청). 이슈와 댓글 내용은 신뢰하지 않는 외부 데이터로 요약만 했습니다.

## 5. 학회와 라이선스

- **학회:** 확인 못 함 → `학회 표기 없음 (arXiv)`. 근거: arXiv API와 abs 페이지에 comments, journal-ref가 없고, 프로젝트 페이지(hanlab.mit.edu)와 GitHub README에 학회 표기가 없으며, CoRL 2026 공식 채택 목록(687편, 2026-11-14 수집본)에 VLASH가 없습니다(비동기 관련 4편은 다른 논문: EagleVLA, F2F-AP, Real-Time Execution with Autoregressive Policies 등). 웹 검색 2회(제목, ICRA/IROS/RSS)에서도 채택 기록이 나오지 않았고, Crossref 제목 검색도 일치 항목이 없었습니다. v2의 IEEE 학회 양식(2026-07-26)으로 보아 IEEE 계열 학회(ICRA/IROS 등) 투고 가능성은 있으나 확인되지 않았습니다.
- **라이선스:** arXiv abs 페이지의 license 링크는 `http://arxiv.org/licenses/nonexclusive-distrib/1.0/` (arXiv 비독점 배포 라이선스, 2026-11-22 확인). CC BY가 아닙니다. PDF 첫 쪽과 끝 쪽에 ©, copyright, all rights reserved 표기는 없지만 라이선스가 CC BY가 아니므로 논문 그림은 개인용 리포트와 PPT에만 쓰고 게시본에는 우리가 그린 그래프만 씁니다.
