# 02. 선행연구, 구성 요소 출처, 관련 저장소

참고 번호는 PLaW-VLA 논문의 참고문헌 번호입니다.

## 1. 계보

PLaW-VLA는 "VLA에 미래 예측을 넣는" 흐름 중 **예측 대상을 사전학습된 예측 지향 표현 공간(V-JEPA 2)으로 잡고, 예측값을 행동 전문가에 직접 조건화**하는 갈래입니다. 논문 §1–2가 그리는 갈래는 아래와 같습니다.

| 갈래 | 대표 | 예측 대상 | 행동과의 연결 | PLaW-VLA와의 차이 |
|---|---|---|---|---|
| 미래 이미지·영상을 만들고 역동역학으로 행동 복원 | [26–29] (본문 인용) | 픽셀·영상 | 역동역학 | 픽셀을 만들지 않음 |
| 영상–행동 공동 생성 (자기회귀) | WorldVLA [18], RynnVLA-002 [17], LingBot-VA [30] | 이미지·영상 토큰 | 같은 시퀀스 안 | 생성 비용이 큼 |
| 영상–행동 공동 생성 (확산·MoT) | Motus [16], mimic-video [12], 그 밖의 [10, 11, 13–15] | 영상 잠재 | 부분 노이즈 제거된 잠재를 조건으로 | 영상 생성기를 거침, 지연 큼 |
| 미래 표현을 보조 감독으로 | FLARE [7] | 미래 관측 임베딩 | 확산 정책 특징 정렬 | 예측값을 행동 조건으로 쓰지 않음 |
| 잠재 행동 + 미래 V-JEPA 예측 | VLA-JEPA [8] | V-JEPA 2 특징 | 잠재 행동 표현 | 행동 전문가가 예측 토큰을 읽지 않음 |
| V-JEPA 2 특징을 시각 입력에 융합 | JEPA-VLA [9] | (예측 없음, 표현만) | 특징 융합 | 미래를 예측하지 않음 |

논문 스스로의 정리(§2): 앞선 방법들은 예측 표현을 **보조 감독**, **잠재 행동**, **시각 특징** 중 하나로 쓰고, PLaW-VLA는 "관측 이력에서 예측한 미래 표현을 행동 생성의 조건으로 직접 쓴다"는 점이 다르다는 것입니다. 이 차별점은 구조상 사실입니다(코드에서도 행동 전문가가 예측 토큰의 KV를 읽음, `04` §2). 다만 이 차이가 **성능 차이로 이어지는지**를 같은 백본·같은 데이터로 가른 실험은 없습니다. VLA-JEPA나 FLARE는 같은 π0.5 백본에서 재현되지 않았고, 표 4(c)의 비교 대상은 같은 구조에서 예측 공간만 Cosmos VAE로 바꾼 변형 하나입니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| Mixture-of-Transformers + flow matching 행동 전문가 | π0 [2], π0.5 [1] (openpi) | 백본·초기화·`Beta(1.5, 1)` 시간 분포 모두 π0 계열 그대로 |
| VLM | PaliGemma [32] (SigLIP 400M + Gemma 2.6B) | π0.5 가중치로 시작 |
| 월드 모델 전문가 | Gemma 300M (행동 전문가와 같은 크기) | 처음부터 학습 |
| 예측 표현 공간 | V-JEPA 2 [31] (고정) | 크기(ViT-L/H/g)는 논문에 없음. 공식 코드 기본값은 ViT-L (`04` D6) |
| 미래 질의 토큰으로 병렬 예측 | 학습 가능한 질의(쿼리) 방식 (DETR류, 비디오 JEPA 예측기류) | 논문은 출처를 따로 적지 않음 |
| 블록 인과 어텐션 | big_vision / PaliGemma의 prefix-LM 마스크 | 코드가 `make_att_2d_masks`를 big_vision에서 가져왔다고 주석에 적음 |
| 월드 모델 분기 드롭아웃 α | 모달리티 드롭아웃 계열 | 배치 단위로 분기 전체를 끔 (DDP에서 전 rank 공유, `04` D9) |
| 3단계 영상→로봇 학습 | LAPA류 영상 사전학습 + 로봇 후학습 | 논문은 선행 단계를 "scalable path"로만 기술 |
| 데이터 | InternData-A1 [50], AgiBotWorld [51], RoboTwin 2.0 [41], LIBERO-90 [38], EgoDex [52] | 아래 §5 라이선스 주의 |
| 예측 공간 비교 | Cosmos VAE [42] | 재구성 지향 잠재의 대표로 사용 |
| 벤치마크 | LIBERO [38], LIBERO-Plus [39], RoboTwin 2.0 [41] | 기준선 값은 각 공식 리더보드·원논문에서 가져옴 |

## 4. GitHub와 관련 저장소 (2026-11-22 확인)

| 저장소 | 관계 | 라이선스 | 별 | 마지막 푸시 | 비고 |
|---|---|---|---|---|---|
| [RainyRobo/PLaW-VLA](https://github.com/RainyRobo/PLaW-VLA) | **공식 구현** | Apache-2.0 (+ NOTICE, MIT 일부, Gemma 약관 사본) | 8 | 2026-10-13 | 커밋 1개(`5fb8f32`), 얕은 복제 7.2 MB. 사전학습 체크포인트는 "Release planned"(미공개) |
| [프로젝트 페이지](https://rainyrobo.github.io/PLaW-VLA) | 공식 | (페이지 소스는 별도 저장소) | | | 실로봇 막대 값은 그림 4와 같음. Model 링크는 비활성 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 백본·학습 코드의 모체 | Apache-2.0 | 14,732 | 2026-08-24 | 공식 코드가 openpi에서 파생(파일 머리에 명시) |
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | 고정 인코더 | MIT | 4,913 | 2026-10-29 | 코드는 HF `facebook/vjepa2-vitl-fpc64-256`를 직접 로드 |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 벤치마크 (서브모듈) | MIT | 2,565 | 2025-03-15 | |
| [RoboTwin-Platform/RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) | 벤치마크 (서브모듈) | MIT | 3,074 | 2026-09-24 | 시뮬 자산은 별도 약관 |
| [sylvestf/LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | 벤치마크 (서브모듈, 선택) | **라이선스 없음** | 502 | 2026-01-21 | 공식 코드 docs가 "코드 라이선스 미선언"을 명시 |
| [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA) | 비교 대상 | Apache-2.0 | 636 | 2026-09-29 | |

HF 데이터셋 `RainyBot/libero_v3_eef`(LIBERO EEF 형식 변환본)는 README가 "Available"이라 적었으나 이 분석에서 받지 않았습니다(크기 미확인).

## 5. 데이터 라이선스 주의 (공식 `docs/licenses.md`)

Stage I–II 데이터 중 InternData-A1과 AgiBotWorld는 **CC BY-NC-SA 4.0**, EgoDex는 **CC BY-NC-ND**입니다. 코드 저장소의 Apache-2.0이 가중치나 변환 데이터의 권리를 주지 않으며, 사전학습 체크포인트가 나와도 비상업 조건이 따라올 수 있습니다. 이 점은 모델을 가져다 쓸 때 사전에 볼 항목입니다.
