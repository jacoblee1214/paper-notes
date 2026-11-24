# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

W2A는 새 부품을 만들기보다 이미 있는 부품 네 개를 "교사 잠재 → PCA → Q-Former 읽기 → 코사인 정렬" 순서로 이은 구성입니다. 각 부품의 출처는 논문 인용 번호로 적습니다.

| 구성 요소 | 출처 | W2A에서의 쓰임 |
|---|---|---|
| 흐름 정합과 DiT 정책 | Flow Matching [15], DiT [13], Diffusion Policy [46] | 행동 DiT(표 7)와 식 3–7 |
| 비디오 교사 | Cosmos-Predict2 [41] + LoRA [42] | 로봇 영상에 적응한 교사. 14번째 블록, τ_v = 0.5 (표 8) |
| 비디오-행동 두 단계 레시피 | VPP [6], mimic-video [7] | 1단계 영상 모델 적응 → 2단계 행동 디코더. W2A는 교사를 추론에서 뺍니다 |
| 공동 모델링 계열 WAM | UVA [8], Motus [11], Cosmos Policy [12], LingBot-VA [38], DreamZero [39], GigaWorld-Policy [40] | 추론에 비디오 분기를 남기는 계열. 비교 대상으로만 언급 |
| 추론 비용을 줄이는 WAM | Fast-WAM [10], UVA [8], GigaWorld-Policy [40] | 부록 J가 "보완적"이라고 설명. 지연 비교(그림 6)에는 없음 |
| 정렬 모듈 | Q-Former (BLIP-2) [16] | 행동 잠재에서 교사 격자로 읽는 쿼리 모듈. 논문은 "현재 구현에서는 구조화된 정보 읽기"라고 씀 (부록 B.4, E) |
| 표현 증류 | Hinton 외 [50], FitNets 계열 [51–54], Robot-DIFT [58] | 부록 E가 "중간 표현 증류"로 위치시킴 |
| 압축 | PCA [14] | 식 8. K = 482 |
| 웜스타트 | [43] | 이전 청크의 남은 부분을 τ_ws = 0.5로 교란해 재사용 (§4.3) |
| 벤치마크 | LIBERO [44], CALVIN [45] | 표 1, 2 |

**가장 가까운 대안.**
- *Fast-WAM* [10]: 학습 때 비디오 공동 학습을 하고 추론 때 미래 생성을 없앱니다. "비디오 공동 학습의 효과가 상상의 효과보다 크다"는 같은 방향의 주장이지만, 비디오 분기를 학습 손실로 유지하는 방식이고 W2A는 별도 교사의 잠재를 정렬 손실로 씁니다. 지연 수치 비교는 W2A에 없고(그림 6은 π0, π0.5, OpenVLA, mimic-video만), 부록 J가 "보완적"이라고만 적습니다.
- *UVA* [8]: 공동 비디오-행동 잠재를 학습하고 추론 때 비디오 디코딩을 건너뜁니다.
- *VPP* [6]: 비디오 확산 모델의 중간 표현을 정책 조건으로 씁니다(추론에서도 사용). CALVIN 표 2에서 W2A와 평균 길이 4.33 대 4.30으로 가깝습니다.
- *mimic-video* [7]: 영상 모델을 일부만 디노이징한 중간 표현을 행동 디코더에 씁니다. 추론에 비디오 모델이 남습니다.

## 2. 코드와 저장소 (GitHub 표)

2026-11-24에 `gh api`로 확인했습니다.

| 항목 | 위치 | 상태·라이선스 | 비고 |
|---|---|---|---|
| 공식 코드 (자리표시) | github.com/misaka0502/W2A_code | 공개, 파일은 README 한 개(93 B, "coming soon..."), 라이선스 없음, 별 0, 마지막 푸시 2026-05-13, 커밋 `661af62` | 코드 없음. 복제하지 않음 |
| 프로젝트 페이지 저장소 | github.com/misaka0502/W2A | 공개, 정적 사이트(Nerfies 템플릿 기반), 라이선스 없음, 별 0, 마지막 푸시 2026-10-12 | 페이지 본문은 논문 표와 같은 수치. 템플릿의 "Code" 버튼은 링크가 비어 있음 |
| 프로젝트 페이지 | misaka0502.github.io/W2A | 공개 | 개요 영상, 표 1–4, 그림, BibTeX(`booktitle = Conference on Robot Learning`, 2026) |
| 논문 | arXiv:2611.02600 (v1) | CC BY 4.0 (abs 페이지 "view license" 링크 확인) | PDF 23.2 MB |
| 비디오 교사 코드 | github.com/nvidia-cosmos/cosmos-predict2 | Apache-2.0, 별 793, 마지막 푸시 2025-10-29 | 교사의 기반 모델 [41] |
| mimic-video | github.com/mimic-video/mimic-video | Apache-2.0, 별 319, 마지막 푸시 2026-10-19 | 지연 비교 대상 [7] |
| Fast-WAM | github.com/yuantianyuan01/FastWAM | GitHub API는 NOASSERTION, 별 1670, 마지막 푸시 2026-08-20 | 가장 가까운 비교 대상 [10] |
| VPP | github.com/roboterax/video-prediction-policy | MIT, 별 419, 마지막 푸시 2025-05-17 | CALVIN 비교 대상 [6] |
| DreamZero | github.com/dreamzero0/dreamzero | Apache-2.0, 별 2782, 마지막 푸시 2026-04-19 | LIBERO 비교 대상 [39] |
| UVA | github.com/ShuangLI59/unified_video_action | MIT, 별 419, 마지막 푸시 2026-08-21 | [8] |
| LIBERO | github.com/Lifelong-Robot-Learning/LIBERO | MIT, 별 2567 | 벤치마크 [44] |
| CALVIN | github.com/mees/calvin | MIT, 별 1006 | 벤치마크 [45] |
| Q-Former 원 구현 | github.com/salesforce/LAVIS | BSD-3-Clause, 별 11255 | BLIP-2 [16] |

- 체크포인트, 데이터, 실로봇 시연 750개 공개 여부: 논문과 프로젝트 페이지 어디에도 링크가 없습니다.
- 공식 저장소가 비어 있어서 **복제하지 않았고** 코드 대조(D 항목)는 없습니다. 대신 `04_claim_verification.md`에서 논문 주장(C 항목)을 대조합니다.

## 4. 참고: 표 1 기준선 값의 출처

W2A 표 1의 기준선 행은 인용 값인지 재실행인지 표에 적혀 있지 않습니다. 확인한 것(2026-11-24, arXiv PDF에서 직접 검색):

| 행 | W2A 표 1 | 원 논문에서 확인한 것 |
|---|---|---|
| Diffusion Policy, Octo, OpenVLA | Spatial/Object/Goal은 78.3/92.5/68.3, 78.9/85.7/84.6, 84.7/88.4/79.2 | mimic-video 논문 표 II에 같은 값이 있음. Long 열(50.5, 51.1, 53.7)은 별도 출처 |
| π0.5 | 98.8/98.2/98.0/92.4 | Fast-WAM 논문 표 2에서도 같은 값(평균 96.85)이 쓰임 |
| mimic-video | 94.2/96.8/90.6/**84.9**, 평균 91.6 | mimic-video 논문(arXiv v2) 표 II는 **세 스위트(Spatial, Object, Goal)만** 보고하고 평균 93.9. Long 84.9는 이 논문 PDF에서 찾지 못함 |
| DreamZero-5B | 90.4/94.8/87.2/82.2, 평균 88.7 | DreamZero 논문(arXiv:2602.15922)의 PDF에서 LIBERO 결과를 찾지 못함. 5B 변형은 그 논문에서 크기 비교(DROID 계열 실험)로만 나옴 |

즉 표 1의 mimic-video Long 값과 DreamZero-5B 행은 원 논문 표로는 대조되지 않습니다. 저자가 직접 학습·평가한 값일 수 있지만 논문에는 그렇게 적혀 있지 않아서, 평가 횟수와 시드 같은 비교 조건을 알 수 없습니다(`03` §2.1, `04` C5). 저자 확인이 필요합니다.
