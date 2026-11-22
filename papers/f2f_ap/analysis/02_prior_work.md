# 02. 선행연구, 구성요소 출처, 저장소, 학회

## 1. 선행연구 계보

```
행동 청크 (ACT 2023, Diffusion Policy 2023/IJRR, UMI 2024)
   └─ 비동기 추론: 추론과 실행을 겹침
        ├─ SmolVLA (arXiv:2506.01844) 비동기 스택, A2C2 "Leave No Observation Behind" (arXiv:2509.23224): 청크 실시간 보정
        ├─ Real-Time Chunking (RTC, Black 외, arXiv:2506.07339): 확정 행동 고정 + 인페인팅
        │    ├─ Training-time action conditioning (arXiv:2512.05964): 보정을 학습으로 이동
        │    └─ Masked action chunking (arXiv:2601.20130)
        ├─ VLASH (arXiv:2512.01031): 정책에 지연 H 뒤 로봇 상태를 조건으로 줌 (영상은 낡은 채)
        ├─ Delay-Aware Diffusion Policy (arXiv:2512.07697): 관측–실행 간극, 동적 과제 (논문은 인용만 하고 비교하지 않음)
        └─ F2F-AP (이 논문): 영상도 H 뒤로 맞춤. 미래 영상을 예측 흐름을 그려 합성
점 궤적·흐름 표현 (General Flow 2024, ATM 2024, Track2Act 2024, Flow as Cross-Domain Interface 2024, 3DFlowAction 2025, Dream2Flow 2025, PointWorld 2026)
   └─ 대부분 "정적 환경 속 물체를 옮기는" 용도 → 이 논문은 물체 자체의 자발적 운동 예측에 사용 (§2.1)
   └─ 가장 가까운 것: Motion Before Action (arXiv:2411.09658, 물체 운동을 조건으로 확산)
동적 조작: DynamicVLA (arXiv:2601.22153, 서론에서만 인용)
```

핵심 차이: 기존 비동기 방법들은 **행동 쪽 경계**(청크 사이 연속성, 낡은 상태)를 고치고, F2F-AP는 **관측 쪽 낡음**을 영상 수준에서 고칩니다. 가장 큰 가정은 "짧은 지평선(0.4 s)에서 물체 점 운동이 영상만으로 예측 가능하다"입니다.

## 2. 구성요소 출처와 GitHub 표

### 2.1 구성요소별 출처

| 구성요소 | 출처 | 비고 |
|---|---|---|
| 행동 생성(U-Net, DDIM 16단계, 상대 자세) | Diffusion Policy, UMI | 공식 코드는 UMI 저장소 코드를 벤더링 |
| 영상 인코더 | CLIP ViT-B/16 (timm `vit_base_patch16_clip_224.openai`), 어텐션 풀링 | 코드 확인 (`04` §2) |
| 지연 보정(미래 상태 입력) | VLASH | 코드에서 `policy_type: vlash`와 `f2f`가 같은 지연 처리 사용 |
| 비동기 기준선 | UMI 동기·비동기, RTC(UMI 정책에 이식), VLASH | 이식 코드는 대부분 비공개 (`04` D16) |
| 정답 흐름 | SAM(2) 추적 + K-Means(K = 4) | 부록 B.2. 데이터 처리 스크립트는 공개 |
| 흐름 예측기 | BERT형 인코더 + 국소 상관(RAFT/MemFlow류 아이디어) + 히트맵 | 헤드 구조는 저자 설계 |
| 대조 손실 | CLIP InfoNCE, CLAP (arXiv:2601.04061) 참고 | 시간 마스크는 저자 설계 |
| 그리퍼 가림 합성 | SAMEO / Segment Anything Even Occluded (arXiv:2503.06261) | 부록 B.3 |
| 데이터 수집 | UMI 장치(GoPro 어안, 그리퍼) | UMI-on-Legs의 WBC를 사족에 사용 |

### 2.2 GitHub 표 (2026-11-22, GitHub API)

| 저장소 | 역할 | 라이선스 | 별 | 마지막 푸시 | 메모 |
|---|---|---|---|---|---|
| github.com/Why-peace/F2F-AP | **공식 코드** (논문 초록 링크의 "Website" 소유자 계정) | **없음** (LICENSE 파일 없음, API license null) | 17 | 2026-09-05 | 커밋 `2c382e9` "v1". 548개 파일. 정적 분석함 (`04`) |
| github.com/f2f-ap/f2f-ap | 프로젝트 페이지의 "Code (GitHub)" 링크 대상 | 없음 | 8 | 2026-06-12 | 파일 548개 모두 위 저장소와 **같은 내용**(트리 해시 비교). 커밋 이력만 다름(`corl` 2회) |
| why-peace.github.io/F2F-AP/ | 프로젝트 페이지 (영상, 요약) | - | - | - | 중국어 해설(Zhihu) 링크 포함 |
| github.com/real-stanford/universal_manipulation_interface | UMI (벤더링 원본) | MIT | 1,647 | 2026-05-26 | F2F 저장소의 `third_party/umi`, 상속 |
| github.com/real-stanford/umi-on-legs | UMI-on-Legs (사족 기준선) | MIT | 570 | 2026-07-20 | 사족 이동 조작 WBC. F2F 저장소에는 연결 코드 없음 |
| github.com/real-stanford/diffusion_policy | Diffusion Policy (벤더링 원본) | MIT | 4,738 | 2024-12-24 | `third_party/diffusion_policy` |
| github.com/mit-han-lab/vlash | VLASH (기준선) | Apache-2.0 | 526 | 2026-04-22 | VLA용. F2F는 UMI 정책에 아이디어만 이식 |
| github.com/Physical-Intelligence/real-time-chunking-kinetix | RTC 공식 (Kinetix 평가) | MIT | 637 | 2025-12-08 | F2F 코드의 `predict_action_rtc`는 RTC식 가이던스를 UMI diffusion 정책에 구현한 함수로 보임(원 저장소와의 대조는 하지 않음) |
| github.com/facebookresearch/sam2 | SAM 2 (정답 흐름 파이프라인) | Apache-2.0 | 20,210 | 2026-05-30 | 데이터 처리 스크립트가 선택 의존 |
| github.com/Why-peace/realtime-vla-v2-reproduce | 같은 저자 계정의 보조 저장소 | MIT | 0 | 2026-03-31 | 실시간 VLA 재현용으로 보임, 내용은 읽지 않음 |

공식 코드의 의존 패키지(`conda_environment.yaml`): Python 3.9, PyTorch 2.1.0 + CUDA 12.1, diffusers 0.18.2, timm 0.9.7, lightning 2.1.3, deepspeed 0.12.6, lmdb, SAM2(git). 라이선스 파일이 없어 이 저장소 코드의 재사용 권리는 불명이고, 벤더링한 UMI·Diffusion Policy 폴더에는 원 라이선스(MIT) 파일이 따라오지 않았습니다.

## 4. 가까운 대안과 차이

| 대안 | 같은 점 | 다른 점 | 이 논문의 비교 |
|---|---|---|---|
| RTC | 청크 경계 연속성 | 낡은 관측은 그대로 | 비교함 (v3부터 전 과제) |
| VLASH | 미래 상태 조건화 | 영상은 낡음 | 비교함 |
| Delay-Aware Diffusion Policy (2512.07697) | 관측–실행 간극, 동적 과제 | 지연을 훈련으로 흡수 (원문 미확인, 제목과 인용 문맥만) | 인용만, 비교 없음 |
| 영상 생성 모델로 미래 예측 | 미래 관측 합성 | 수 초 걸림 (논문 주장, 측정 없음) | 비교 안 함 |
| 속도 입력(광학 흐름 등)을 정책 입력에 직접 | 운동 정보를 입력 | 그리기 대신 채널 입력 | 비교 안 함 (MF-UMI는 프레임 6개 적층일 뿐) |
| 상태 외삽(칼만 필터 등) | 지연 보상 | 학습 없는 외삽 | 비교 안 함 |

흐름을 영상 위에 "그려서" 정책에 넣는 설계는 사전학습된 RGB 인코더를 그대로 쓰기 위한 선택으로 보입니다. 그리기 두께·색 변경이 성능에 거의 영향이 없다는 절제(표 2)는 이 표현이 인코더에 쉽게 읽힌다는 근거이지만, 왜 채널 입력이나 속도 벡터를 직접 넣는 것보다 나은지는 시험되지 않았습니다.

## 5. 학회 확인과 라이선스

- **학회:** CoRL 2026 **포스터**. CoRL 2026 공식 채택 목록(corl.org 목록, 2026-11-14 수집본)에 제목이 정확히 일치하는 항목이 있습니다(OpenReview forum `iXJWieKME1`, 일정 Poster post-p2, 2026-11-10 5:00–6:30 PM, 포스터 번호 10, 발표 형식 poster). 논문 v3/v4 첫 쪽 하단 각주도 "10th Conference on Robot Learning (CoRL 2026), Austin, Texas, USA"이고, 공식 저장소 설명에 "CoRL2026 Accept"가 있습니다. arXiv 댓글 필드는 "Tsinghua University, 18pages,11 fugures"뿐이라 학회 표기가 없고 journal-ref도 없습니다.
- **라이선스:** arXiv abs 페이지(2026-11-22 확인)의 license 링크는 `http://arxiv.org/licenses/nonexclusive-distrib/1.0/` ("view license")입니다. CC BY가 아니므로 논문 그림은 개인용 리포트·PPT에만 쓰고 게시본에는 쓰지 않습니다. 원문 PDF 첫 쪽·끝 쪽의 ©, copyright, all rights reserved, 회사 표기는 없었습니다(pdftotext 전체 검색).
