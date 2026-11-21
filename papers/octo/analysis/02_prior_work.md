# 02. 선행연구, 구성 요소의 출처, 관련 자료 (Octo)

## 1. 계보

Octo는 세 줄기를 합친 연구입니다(논문 §II).

1. **로봇 데이터 규모화:** 자동 수집과 사람 원격 조작으로 모은 대규모 로봇 데이터(논문 §II의 [19, 41, 42, 12, 71, 30] 등).
2. **transformer 정책과 행동 청크:** ACT(Zhao 외, 2023)식 행동 청크와 transformer 정책(논문 [98, 17, 28]).
3. **범용 로봇 정책(GRP):** GNM(내비게이션), RoboCat(다중 임베디먼트, 목표 조건), RT-1·RT-2-X(OXE 기반 언어 조건)(논문 §I, [80, 9, 10, 67, 103]).

논문의 주장은 "개별 요소는 선행연구에 있지만 이 조합, 즉 여러 임베디먼트에서 학습하고 새 관측·행동 공간으로 미세조정할 수 있는 열린 모델은 새롭다"입니다(§I 끝). 이 주장은 맞는 편입니다. 특히 **재사용 가능한 블록 단위 마스크**(§III-A)와 **읽기 토큰**이 입력과 출력을 바꿀 때 가중치를 덜 버리게 합니다. 다만 이 설계는 Octo 고유라기보다는 transformer 정책의 관례를 따른 것입니다.

## 2. 구성 요소별 출처

| 구성 요소 | Octo에서의 쓰임 | 출처 (논문 참고문헌 번호) |
|---|---|---|
| 확산 행동 헤드, DDPM, 코사인 스케줄 | 행동 분포 학습, 20단계 잡음 제거 | Chi 외 (Diffusion Policy) [17]; Ho 외 DDPM [34]; Nichol & Dhariwal 코사인 스케줄 [66] |
| 행동 청크 | 한 번에 여러 행동 예측 (사전학습 4) | ACT [98]; Diffusion Policy [17] |
| 텍스트 인코더 | t5-base(111M), 동결 | Raffel 외 T5 [74] |
| 시각 표현 기준선 | VC-1 (ViT-B) | Majumdar 외 [57] |
| ViT 크기 체계와 학습 레시피 | Small·Base 비율, AdamW와 역제곱근 감쇠 | Zhai 외 Scaling ViT [97] |
| 목표 이미지 상대 샘플링 | hindsight relabeling | Andrychowicz 외 [2] |
| 기준선 RT-1-X, RT-2-X | zero-shot 비교 | Open X-Embodiment Collaboration [67]; RT-2 [103] |
| ResNet+FiLM 기준선 | 처음부터 학습 기준선 | RT-1 [10]; FiLM [70] |
| 데이터 혼합 | OXE 25개 데이터셋 | Open X-Embodiment [67] |

## 4. GitHub, 프로젝트 페이지, 관련 자료 (확인 2026-11-21)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (octo)** | https://github.com/octo-models/octo | MIT | 별 1,851, 마지막 push 2024-07-31 (GitHub API) | `repo/` 얕은 클론, HEAD `241fb35`(커밋 2024-07-30). 정적 읽기만 (`04`) |
| 프로젝트 페이지 | https://octo-models.github.io | - | - | arXiv 초록 페이지의 "Project website" 링크와 같음. 영상과 설명 |
| 체크포인트 (HF) | https://huggingface.co/rail-berkeley/octo-small-1.5, `octo-base-1.5` | 저장소 README 기준 | - | 받지 않음. 논문의 `octo-base`(v1 공개본)와 이름이 다름 (`04` D2) |
| 논문 PDF | arXiv:2405.12213 (v2) | arXiv 라이선스 확인은 §5 | - | `paper/paper.pdf` |

공식 코드는 README와 저장소 상태 기준으로 2024-07 이후 갱신이 없습니다. 2026-11-21 기준 최근 활동은 확인되지 않았습니다. 이 점이 재현 계획의 위험 요인입니다.

## 5. 라이선스 (arXiv 초록 페이지 직접 확인, 2026-11-21)

- arXiv 초록 페이지의 라이선스 링크는 **`http://creativecommons.org/licenses/by/4.0/`** (CC BY 4.0)입니다. 링크의 아이콘과 문구는 "view license", 알림 문구는 "Rights to this article"입니다.
- PDF 전체를 `pdftotext`로 훑어 ©, copyright, "All rights reserved", 회사 권리 고지가 없음을 확인했습니다.
- 그래도 이 리포트의 **공개 산출물은 우리가 직접 그린 그래프 두 장만 씁니다.** 논문 그림(로봇 사진, 데이터셋 프레임 포함)을 재배포할 필요가 없고, 표 I과 표 VI의 수치는 재계산 결과로 적었습니다(공개 범위 원칙에 따른 선택).
- 개인용 리포트와 PPT에도 논문 그림은 넣지 않았습니다.

## 6. 이 논문이 비교하는 대안

| 대안 | 논문의 비교 | 여기서 본 것 |
|---|---|---|
| RT-1-X (35M) | zero-shot 기준 (그림 5) | Octo가 평균 29% 높다는 수치는 그림에만 있음 |
| RT-2-X (55B) | WidowX, RT-1 Robot | 수치는 같다고 함 (그림 5) |
| VC-1 사전학습 인코더 + MLP 헤드 | 표 I 기준선 | 6개 설정 모두에서 Octo보다 낮음 (표 I, 평균 15 대 72) |
| 처음부터 학습한 ResNet+Transformer | 표 I, 표 VI 기준선 | 표 VI 기준 Octo와 유의한 차이 없음 (p=0.29) |
