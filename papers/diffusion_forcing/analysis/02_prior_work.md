# 02. 선행연구, 공식 코드, 이 저장소와의 관계

## 1. 계보

Diffusion Forcing(DF)은 두 줄기를 합칩니다. 한 줄기는 **다음 토큰 예측**입니다. 교사 강제(teacher forcing, Williams & Zipser 1989, 참고문헌 [65])로 학습하고, 가변 길이 생성과 조건 생성이 쉽습니다. 다른 줄기는 **전체 시퀀스 확산**입니다. 확산 모델은 샘플링 중 궤적을 목표에 맞게 유도하기 쉽습니다(Diffuser, Janner et al. 2022, [37]).

| 줄기 | 대표 연구 (논문 참고문헌 번호) | 이 논문이 가져온 것 | 이 논문이 다르게 한 것 |
|---|---|---|---|
| 다음 토큰, 교사 강제 | Williams & Zipser 1989 [65] | 인과 구조와 가변 길이 | 잡음 섞인 문맥으로 학습하고 샘플링해서 롤아웃 발산을 막음 |
| 전체 시퀀스 확산 | Diffuser [37] | 궤적 전체를 확산으로 샘플링, 안내 | 모든 토큰에 같은 잡음을 주는 대신 토큰마다 독립 잡음 |
| 정책으로서의 확산 | Diffusion Policy [10] | 시각 운동 정책의 확산 행동 | 기억을 잠재 상태로 가짐 (§4.4) |
| 시계열 확산 | TimeGrad [50], ScoreGrad [68], Transformer-MAF [51] | CRPS 평가 설정 | 같은 구조를 여러 시계열에 씀 (§4.5) |
| 프레임별 잡음 확산 | Rolling diffusion [52], AR-diffusion [67] | 프레임마다 다른 잡음 수준 | 논문 부록 B 말미에 따르면 이 둘은 안정화와 조건 안내 제거는 가능하지만, 인과 불확실성 모델링(§3.3)은 못 한다고 봄. 학습 한 번으로 샘플링 스케줄을 바꾸는 점이 다름 |
| SNR 가중 | Min-SNR [24] | 손실 가중 | 잡음 시퀀스에 맞게 누적 SNR 가중 (부록 D.1) |

**가까운 대안.** 논문이 직접 비교한 것은 다음 프레임 교사 강제 기준선과 인과 전체 시퀀스 확산 기준선입니다(§4.1). 논문은 그 밖에 AR-diffusion과 Rolling Diffusion을 부록 B에서 비교합니다. 정리 3.1(부분 시퀀스 가능도의 하한)은 재계산하지 않았습니다(03 §6).

**후속 연구.** 웹 검색으로 2026년 후속 논문 "Diffusion ReRoll: Revisable Denoising for Robotic Sequential Prediction"(arXiv:2607.19919)이 토큰별 잡음 관점을 이어받은 것을 확인했습니다. 제목과 요약 수준으로만 봤고 본문은 읽지 않았습니다.

## 2. 공식 코드와 관련 저장소

| 항목 | 링크 | 라이선스 | 별 / 포크 | 마지막 push (GitHub API) | 상태 |
|---|---|---|---|---|---|
| 공식 코드 | github.com/buoyancy99/diffusion-forcing | MIT (`LICENSE`, Copyright 2024 Boyuan Chen). API 자동 판독은 NOASSERTION | 1308 / 73 | 2026-07-06 (생성 2024-06-26) | **얕은 복제함**: `main` `475e0bc`(2025-02-22), `paper` `dd65516`(2025-02-17). `repo/`에 두고 정적으로만 읽음 |
| 프로젝트 페이지 | boyuan.space/diffusion-forcing | 해당 없음 | 해당 없음 | 해당 없음 | 논문 초록과 TL;DR, 코드·논문·영상 링크. 학회 표기 없음 |
| 후속 저장소 (Diffusion Forcing v2 / history-guidance) | github.com/buoyancy99/history-guidance | 라이선스 필드 없음 | 0 / 0 | 2025-08-25 | README가 "고품질 영상 생성"용 후속이라고 소개. 복제하지 않음 |
| 연구 템플릿 | github.com/buoyancy99/research-template | README는 MIT, API 판독 NOASSERTION | 172 / 17 | 2025-03-04 | 공식 코드 README가 이 템플릿을 포크했다고 씀. 복제하지 않음 |

비공식 재구현은 웹 검색 한 번으로는 찾지 못했습니다. GitHub 토픽 검색은 하지 않았습니다.

**공식 코드 두 브랜치의 역할.** `main`은 트랜스포머 기반 재구현(v1.5)이고, 논문 실험(미로 계획, 시계열, 로봇)은 `paper` 브랜치에 있습니다. README가 직접 이렇게 적었습니다. 세부는 04 §D1을 보십시오.

**논문과 저장소 메타데이터.** README의 BibTeX는 `journal={Advances in Neural Information Processing Systems}`, `volume={37}`, 쪽 24081–24125, `year={2025}`입니다. 학회 연도는 2024가 맞습니다(04 D14).

## 4. 이 연구의 위치 (요약)

1. 학습 목적은 DF가 새로 낸 것입니다(독립 잡음 + 부분 시퀀스 하한).
2. 샘플링 쪽 기여(스케줄 행렬, 안정화, MCG)는 확산 샘플링 기법과 결합된 것입니다.
3. 실험 근거는 미로, 시계열, 로봇 소규모, 영상 정성 결과입니다. 대규모 영상이나 로봇 정량 근거는 적습니다.
