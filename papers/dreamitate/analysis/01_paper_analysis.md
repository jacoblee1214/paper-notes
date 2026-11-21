# 01 논문 분석: Dreamitate (arXiv:2406.16862)

- 논문: Junbang Liang, Ruoshi Liu, Ege Ozguroglu, Sruthi Sudhakar, Achal Dave, Pavel Tokmakov, Shuran Song, Carl Vondrick. *Dreamitate: Real-World Visuomotor Policy Learning via Video Generation.* (`paper/paper_v1.pdf`, 17쪽)
- 버전: arXiv에는 v1(2024-06-24)만 있습니다. 프로젝트 페이지가 가리키는 `dreamitate_arxiv_v2.pdf`(49.7 MB)는 본문 Table 2 수치가 v1과 같아서 분석은 v1 기준으로 하고, 차이는 `02`와 `README`에 적었습니다.
- 학회: CoRL 2024 (PMLR v270, "Proceedings of The 8th Conference on Robot Learning", PMLR 270:3943–3960, 2025 표기). 자세한 근거는 README 참고.

## 1. 한 줄 요약

영상 확산 모델(SVD)을 사람이 도구(3D 프린트 도구, CAD 모델 있음)로 작업하는 영상으로 미세조정하고, 새 장면 이미지에서 "작업을 수행하는 스테레오 영상"을 생성한 뒤, 생성 영상 속 도구 궤적을 MegaPose로 6D 추적해 로봇 end-effector 명령으로 쓴다. 정책 = 영상 생성 + 3D 추적입니다 (Abstract, §1; Eq. 1 `a_t = T(v̂_t)`, `{v̂_t} = f_θ(v_0)`).

## 2. 방법 (§3)

| 구성 | 내용 | 출처 |
|---|---|---|
| 생성 모델 | 인터넷 영상으로 사전학습된 SVD를 시작점으로 사용. 인코더·디코더는 고정, spatial/temporal attention만 미세조정 | §3.2, 부록 A |
| 학습 목적 | 두 스테레오 뷰의 프레임 MSE 합 (Eq. 2) | §3.2 Eq. 2 |
| 스테레오 | 25프레임 = 뷰1 13프레임 + 뷰2 12프레임. 프레임별 이미지 임베딩을 시점에 맞게 바꿈. 첫 프레임은 입력과 같아서 버림 | 부록 A |
| 추적 | 알려진 CAD로 생성 영상의 도구를 MegaPose로 6D 추적. 768×448 해상도, Realsense 기본 내부 파라미터 | §3.3 |
| 실행 | 도구 pose (카메라 기준 SE(3))를 로봇 end-effector 목표로 실행. 로봇 종류에 독립적 | §3.3 |
| 과제별 모델 | 과제마다 f_θ를 따로 학습 | §3.2 |

핵심 주장: 도구를 공유하면 사람 손과 로봇 manipulator의 embodiment 차이를 메울 수 있고, 학습 데이터도 원격조작 없이 사람 영상으로 모을 수 있습니다 (§1, §3.1).

## 3. 실험 설정 (§4.1, Table 1)

| 과제 | 학습 물체 | 데모 수 | 테스트 물체 | 테스트 시도 수 |
|---|---|---|---|---|
| Rotation | 31 (실물 14 + 폼 17) | 371 | 10 | 40 |
| Scooping | 17 bowls, 8 particles | 368 | 8 bowls, 4 particles | 40 |
| Sweeping | 6 particles | 356 | 6 particles | 40 |
| Push-Shape | 26 letters | 727 | 8 shapes | 32 |

비교 대상은 Diffusion Policy (ResNet-18, 스테레오 입력, 200 epoch, 12 step 예측, 같은 학습 데이터). 테스트 물체는 학습 물체와 겹치지 않습니다 (Table 1 caption).

## 4. 결과 (§4.2–4.5, Table 2)

| 과제 | Ours | Diffusion Policy | 지표 |
|---|---|---|---|
| Rotation | 37/40 (92.5%) | 22/40 (55%) | 성공률 |
| Scooping | 34/40 (85%) | 22/40 (55%) | 성공률 |
| Sweeping | 37/40 (92.5%) | 5/40 (12.5%) | 성공률 |
| Push-Shape (32 시도) | mIoU 0.731, 회전 오차 8.0° | mIoU 0.550, 회전 오차 48.2° | 평균 |

- §4.6 (Figure 8): 회전 과제에서 데모를 2/3, 1/3로 줄였을 때 Diffusion Policy 성능이 크게 떨어지고, 이 방법은 1/3에서도 높은 성공률을 유지한다고 주장합니다. 그림 수치는 텍스트에 없어서 재계산하지 않았습니다.
- 재계산한 신뢰구간과 정확 검정은 `03_analysis.md`에 있습니다.

## 5. 한계 (§5)

- 생성 영상에서 추적 가능한 도구만 다룰 수 있습니다. 엔드이펙터가 가려지면 실패합니다.
- 강체 도구에 의존하므로 정밀한 제어가 필요한 작업에는 한계가 있습니다.
- 영상 모델이 무거워 실시간 폐루프 제어가 어렵습니다 (§5가 [42, 43] 가속 연구를 언급).

## 6. 읽으면서 본 점

- 작은 대상(scooping의 scooper)이 추적과 생성 모두에 어려운 과제라고 저자 스스로 밝힙니다 (§4.3).
- Diffusion Policy 기준선은 "스테레오 입력, 같은 데이터, ResNet-18"로 맞췄지만, 시도마다 실패 원인(접촉 실패, 잘못된 그릇 선택)을 정성적으로만 설명합니다.
- 과제별 학습 데이터 규모가 작습니다 (데모 356–727개). 일반화의 근거는 대부분 사전학습 영상 모델입니다.
