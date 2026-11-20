# 01. 논문 분석: TesserAct (arXiv:2504.20995, ICCV 2025)

분석 대상은 **arXiv v1 하나**입니다(abs 페이지 `arxiv.org/abs/2504.20995` 확인, 2026-11-20. 제출 이력에 v1만 있고 `v2`, `v3`는 "not found"). 학회판은 CVF Open Access ICCV 2025 PDF이고, 제목이 *Learning 4D Embodied World Models*로 바뀌었습니다. 차이는 §6에 있습니다. 원문은 `paper/paper.pdf`(arXiv v1, 31.6 MB, `.gitignore`), 텍스트는 `paper/paper.txt`(`pdftotext -layout`)입니다. 학회판은 `paper/cvf_iccv2025.pdf`에 있습니다.

## 0. 메타

| 항목 | 내용 |
|---|---|
| 제목 | TesserAct: Learning 4D Embodied World Models (arXiv), Learning 4D Embodied World Models (ICCV 2025 CVF판) |
| 저자 | Haoyu Zhen, Qiao Sun, Hongxin Zhang, Junyan Li (UMass Amherst), Siyuan Zhou (HKUST), Yilun Du (Harvard), Chuang Gan (UMass Amherst) |
| arXiv | 2504.20995, v1 제출 2025-04-29, 분류 cs.CV (abs 페이지에 cs.RO도 표시). 댓글·저널 참조 없음 |
| 학회 | ICCV 2025. 근거는 `02` §1에 정리 (CVF Open Access PDF, 저장소 README "accepted to ICCV 2025", 2025-06-25) |
| 라이선스 | **CC BY 4.0** (abs 페이지 "view license" 링크 `creativecommons.org/licenses/by/4.0/`, 2026-11-20 확인) |
| 프로젝트 페이지 | https://TesserActWorld.github.io |
| 공식 코드 | https://github.com/UMass-Embodied-AGI/TesserAct (MIT, 정적 읽기만, `repo/`) |
| 가중치 | https://huggingface.co/anyeZHY/tesseract (MIT 태그, 다운로드 0으로 표시, 받지 않음) |
| 분석일 | 2026-11-20 |

## 1. 문제와 핵심 아이디어

비디오 세계모델은 2D 픽셀에서 동작해서 깊이와 6-DoF 자세가 필요한 조작 계획에 한계가 있습니다(§1). 이 논문은 **RGB, 깊이(D), 법선(N) 세 비디오를 함께 생성**하고, 이를 4D 장면(시간에 따라 움직이는 3D 점군·메쉬)으로 바꾸는 것을 제안합니다. 생성 모델은 CogVideoX-5B를 미세조정하고(§4.2), 4D 장면은 생성된 깊이와 법선을 최적화하고 광학 흐름으로 시간 일관성을 강제해서 얻습니다(§4.3, 식 3·5·6·7). 행동은 4D 점군을 입력으로 하는 역동역학 모델(PointNet + MLP, 7-DoF)로 예측합니다(§4.4).

핵심 주장은 세 가지입니다.
1. RGB-DN 공동 생성이 깊이와 법선의 정확도를 높인다(표 2).
2. 생성된 4D 장면이 2D 비디오 세계모델보다 행동 계획 성공률을 높인다(표 3).
3. 제안한 일관성·정규화 손실이 4D 재구성을 개선한다(그림 3, 정성적).

## 2. 방법

### 2.1 데이터 (§4.1, 표 1)

| 출처 | 유형 | 깊이 | 법선 | 개수 |
|---|---|---|---|---|
| RLBench 20 과제 × 1000 인스턴스 × 4 시점 | 합성 | 시뮬레이터 실측 | DSINE의 `depth2normal` (§4.1) | 80k |
| RT-1 Fractal | 실세계 | RollingDepth (상대 깊이) | Temporal Marigold-LCM | 80k |
| Bridge | 실세계 | RollingDepth | Temporal Marigold-LCM | 25k |
| Something-Something v2 | 사람 손 | RollingDepth | Temporal Marigold-LCM | 100k |

실세계 합계는 205k입니다(`03` §4). 본문은 "약 200k"라고 씁니다.

### 2.2 모델 (§4.2, 그림 2)

- 기반은 CogVideoX-5B입니다. 3D VAE는 고정하고, RGB·깊이·법선마다 입력 투영기 3개를 둡니다. 출력 쪽은 RGB 출력 투영기, Conv3D(입력 잠재와 예측 RGB를 결합), DN 출력 투영기입니다. 새 모듈은 0으로 초기화해서 시작 시 RGB 출력이 CogVideoX와 같게 합니다.
- 텍스트 조건은 `[행동 지시] + [로봇 이름]`입니다(예: "pick up apple google robot").
- 학습 손실은 식 (4)의 노이즈 예측 MSE입니다.

### 2.3 4D 재구성 (§4.3, 식 3–7)

1. 법선을 이용한 깊이 정규화(식 3, 로그 깊이 기반 Bilateral 법선 적분 계열, 참고 [9][16]).
2. 프레임 간 광학 흐름(RAFT, 식 5)으로 정적 영역과 동적 영역을 나누고 배경 영역은 이전 프레임과 같은 깊이가 되게 하는 **일관성 손실** L_c.
3. 생성된 깊이를 따르게 하는 **정규화 손실** L_r(식 6).
4. 전체 목적 함수는 식 (7) = L_s + L_c + L_r을 프레임 순서대로 최적화합니다.

### 2.4 행동 계획 (§4.4, §5.2)

- 현재 4D 점군 s_i와 예측된 다음 상태 s_{i+1}, 지시 T를 입력으로 행동 a_i = ID(s_i, s_{i+1}, T)를 예측합니다.
- PointNet으로 점군을 인코딩하고 텍스트 임베딩과 합쳐 4층 MLP로 7-DoF 행동을 냅니다(Supp. §1.3). 점은 8192개를 샘플하고, 생성물에 맞춰 이미지와 점군 좌표에 상대 크기 20%의 가우시안 잡음을 넣습니다.
- 과제마다 시연 500개로 역동역학을 학습합니다(§5.2.1). 키프레임 단위 예측이며 최대 13 프레임, 해상도 512×512입니다.

## 3. 실험 (본문 수치, 재계산은 `03`)

| 표 | 내용 | 요약 |
|---|---|---|
| 표 2 | 4D 장면 생성 (RGB 지표, 깊이 AbsRel·δ, 법선 평균·중앙·11.25°, 점군 Chamfer L1) | 실세계 400개, 합성 200개, 샘플당 10회. 실세계 Chamfer 0.2030(최저), 합성 0.0811(최저). RGB 지표는 CogVideoX가 실세계에서 더 좋음 |
| 표 3 | RLBench 9개 과제 행동 계획 성공률 | 과제당 100 에피소드. 4DWM 평균 64.9%, UniPi* 60.7%, Image-BC 10.6%. 과제 9개 중 7개에서 UniPi*보다 높음 |
| 표 4 | RLBench 새 시점 합성, SoM 대비 | PSNR 12.99 대 10.94, SSIM 42.62 대 24.02, LPIPS 60.51 대 73.82(낮을수록 좋음), 시간 약 1분 대 약 2시간 |
| 그림 3 | 일관성·정규화 손실 효과 | 정성적 비교만 있음. 정량 제거 실험 없음 |

## 4. 저자가 밝힌 한계 (§7)

- RGB-DN 표현은 세계의 **단일 표면**만 담습니다. 여러 RGB-DN 시점을 생성해 통합하는 것은 향후 과제로 남겼습니다.
- 논문 본문에는 학습 데이터와 평가에 쓴 세부 목록 외의 한계 논의가 거의 없습니다. 실세계 평가의 기준값 문제는 §5에서 다룹니다.

## 5. 주장과 증거의 틈

공식 코드가 있으므로 코드 대조는 `04`에 있습니다(D1–D14). 여기서는 논문 안에서 보이는 틈만 적습니다. 숫자 재계산은 `03`입니다.

1. **실세계 평가의 기준값은 학습 라벨과 같은 추정기입니다.** 실세계 400개의 깊이와 법선은 "Sec. 4.1처럼 추정한 값"입니다(§5.1.1). 즉 RollingDepth와 Marigold가 만든 값을 정답으로 씁니다. 베이스라인 OpenSora와 CogVideoX도 같은 두 추정기로 사후에 깊이와 법선을 얻습니다. 그래서 "더 정확하다"는 주장은 **추정기와의 일치도**를 재는 것이고, 실제 기하 정확도는 아닙니다.
2. **합성 데이터의 법선 정답이 두 가지로 적혀 있습니다.** §4.1은 법선을 DSINE의 `depth2normal`로 계산한다고 쓰고, §5.1.1은 "ground truth depth and normal maps are directly accessible"이라고 씁니다. 깊이에서 계산한 법선은 시뮬레이터 법선이 아닙니다.
3. **공정 비교가 아닌 부분이 있습니다.** 표 2의 CogVideoX와 OpenSora 베이스라인은 **LoRA**로 미세조정했고, TesserAct는 전체 미세조정(SFT)입니다(§5.1.1, §4.2, 저장소 `*_sft`). 차이가 표현 때문인지 학습 방식 때문인지 나뉘지 않습니다.
4. **제거 실험이 없습니다.** 일관성 손실과 정규화 손실의 효과는 그림 3의 정성적 비교뿐입니다. 식 (5)·(6)의 효과를 수치로 확인할 표가 없습니다.
5. **행동 계획의 효과가 한 가지 요인에 묶이지 않습니다.** 표 3에서 TesserAct는 (a) 4D 점군 입력 역동역학, (b) 생성 데이터 RGB-DN 학습, (c) 과제별 시연 500개를 함께 씁니다. UniPi*는 2D 역동역학을 씁니다. (a)만 바꾼 대조가 없어서 4D 표현의 기여를 분리할 수 없습니다.
6. **베이스라인 데이터 예산이 적혀 있지 않습니다.** Image-BC와 UniPi*가 몇 개의 시연을 봤는지 표 3에 없습니다.
7. **통계 불확실성이 없습니다.** 표 2는 10회 평균만 있고 분산과 샘플 수준 신뢰구간이 없습니다. 표 3은 과제당 100 에피소드 단일 실행이고, 시드 반복이 없습니다. 표 4의 시간 측정 하드웨어가 적혀 있지 않습니다.
8. **수식 오류 몇 개.** 식 (3)의 두 번째 항은 ∂_u로 적혀 있는데 ∂_v여야 합니다(바로 위 정의는 ∂_v). λ_g1, λ_g2는 부록 표 5에만 나오고 본문 식 어디에도 정의되지 않습니다. 부록은 "Eq. 12"를 참조하지만 본문 총손실은 식 (7)입니다.

## 6. 버전 차이 (arXiv v1 ↔ ICCV 2025 CVF판)

arXiv 이력에는 v1만 있어서 같은 논문의 학회판과 대조했습니다(단어 수준 diff, `paper/cvf_iccv2025.pdf`).

- **제목**이 *TesserAct: Learning 4D Embodied World Models*에서 *Learning 4D Embodied World Models*로 바뀌었습니다.
- **표 2·표 3·표 4의 수치는 동일합니다.** 표 3의 각 행도 같습니다.
- 본문 변화는 **§3.2 법선 적분 설명 문단 보강**(식 3 앞의 설명 추가), **감사의 글**(NSF award 2404386·2441250과 ONR MURI 지원 명시, 베이스라인 코드 기여자 한 명 추가), **GeoWizard 인용 추가**, 참고문헌 번호 이동(RT-1 [6]→[5], UniPi [15]→[14] 등)입니다.
- CVF판 첫 쪽에는 "This ICCV paper is the Open Access version, provided by the Computer Vision Foundation" 워터마크가 있습니다. 기업 저작권 문구는 없습니다(`§7`).
- 식 (3)의 ∂_u 오류는 두 판 모두에 있습니다(미수정).

## 7. 권리 확인 (요약, 자세한 판단은 README)

- arXiv abs 페이지의 라이선스 링크는 CC BY 4.0입니다. 확인 날짜 2026-11-20.
- PDF 텍스트에서 `©`, `copyright`, `All rights reserved`, `NVIDIA`, `Google`(인용 외), `Meta`, `Inc.` 등을 검색했습니다. 기업 저작권 문구는 없습니다. 소속은 UMass Amherst, HKUST, Harvard뿐입니다.
- 사례 이미지와 데이터셋 화면(Franka, WidowX, Google Robot, RT-1, Bridge 프레임)은 제3자 데이터입니다. 그림을 공개물에 넣지 않는 것이 안전해서 **이번 공개물에는 논문 그림을 넣지 않았습니다**.
