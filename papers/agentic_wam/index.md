---
title: "AW-0 읽기: 코드로 계획하는 World Action Model, 그런데 이득은 어디서 오는가"
date: 2026-10-29
tags: [robotics, world-action-model, vla, code-as-policies, diffusion, paper-review]
paper: https://arxiv.org/abs/2610.30136
code: https://github.com/ZhuoyangLiu2005/agentic-wam
project: https://agentic-wam.github.io/
summary: "코드·미래 영상·행동을 한 번에 denoise하는 AW-0를 표 재계산, RoboTwin 공식 리더보드 대조, 공개 코드 대조로 읽었습니다."
---

> **논문:** Liu, Niu, Zhang, Shao, …, Fan, Darrell. *Agentic World Action Models.* arXiv:2610.30136 (2026). UC Berkeley, NVIDIA, University of Pennsylvania.
> 논문 그림은 CC BY 4.0에 따라 출처를 밝히고 실었습니다(© 저자). 나머지 그래프는 논문 표와 RoboTwin 공식 리더보드 데이터로 직접 그렸습니다. 코드 대조는 공식 저장소 커밋 `275818e`(2026-10-27) 기준입니다.

## 한 줄 요약

World Action Model(WAM)은 미래 영상과 행동을 함께 생성하는 정책입니다. AW-0는 여기에 **코드 전문가**를 붙여서, 스킬 호출 프로그램, 미래 영상, 행동을 하나의 Mixture-of-Transformers 안에서 같은 루프로 denoise합니다. 학습용 프로그램 라벨은 Gemini와 Claude Code가 자동으로 만들고, 반복되는 루틴은 스킬 50개로 정리했습니다.

![Fig. 1](images/paper_fig1_overview.png)
*논문 Fig. 1: 사전학습 데이터, 자동 코드 라벨 파이프라인, 세 전문가 구조, 하이브리드 실행.*

## 1. 구조: 코드가 영상을 이끌고, 행동은 둘 다 읽는다

![Fig. 2](images/paper_fig2_framework.png)
*논문 Fig. 2: 코드는 마스크 확산(이산), video와 action은 flow matching(연속). attention은 코드 → 영상 → 행동 순서다.*

- **코드 전문가:** 동결된 Qwen3-VL ViT로 현재 장면을 보고, 지시문을 읽어서 프로그램을 씁니다. 다른 스트림은 읽지 않습니다.
- **video 전문가:** Cosmos-Predict2.5 기반이고, 코드를 읽으며 미래 프레임을 만듭니다.
- **action 전문가:** 코드와 영상을 모두 읽고 32 step 행동 chunk를 만듭니다.
- **기본 모드:** 코드는 실행되지 않고 계획 조건으로만 쓰입니다.
- **하이브리드 모드:** 생성된 코드가 문법·의존성·whitelist 게이트를 통과하면 Code-as-Policies처럼 실행합니다.

## 2. 숫자는 정확하다

부록 Table 11(50과제 × 6방법 × 2설정, 600칸)에서 평균을 다시 내면 Table 1과 모두 일치합니다(최대 차 0.04). 600칸이 모두 2% 단위여서 과제당 50회 시행과도 맞습니다. 본문과 그림 평균 36개도 재계산과 같습니다. 하이브리드 모드의 채널별 step 비율(Table 12)도 내부적으로 정합합니다. "전체 = 성공률 × 성공 에피소드 + (1 − 성공률) × 실패 에피소드"로 역산한 성공률이 63.12%와 29.24%로, Table 1의 63.1과 29.2와 같습니다.

## 3. 그런데 RoboTwin의 π0.5가 공식 리더보드보다 낮다

![RoboTwin](images/robotwin_headline.png)
*Table 1과 RoboTwin 2.0 공식 리더보드의 π0.5(빗금). 학습 데이터 조건(50과제 × clean 시연 50개)은 같다.*

RoboTwin 2.0 공식 리더보드에는 RoboTwin 팀이 같은 학습 데이터 조건으로 낸 π0.5 결과가 있습니다. 하나의 정책을 50과제에 함께 학습한 결과이고, 값은 clean 70.66, random 46.0입니다. 논문 공개 두 달 전인 2026-08-10에 등록되었습니다. 반면 논문이 재현한 π0.5는 54.9와 27.7입니다. 과제별로 보면 논문 값이 더 낮은 과제가 clean 36개, random 42개입니다.

리더보드 값을 기준으로 다시 비교하면 다음과 같습니다.

- AW-0는 clean에서 10.1점 높습니다.
- random에서는 25.4점, 평균에서는 7.6점 낮습니다.
- 하이브리드 모드의 random 29.2도 46.0보다 16.8점 낮습니다.
- 논문 공개 전에 등록된 27개 항목과 비교하면 AW-0의 평균은 13위입니다.

따라서 "평균이 가장 높다"와 "하이브리드가 random에서 모든 기준선보다 높다"는 **논문이 직접 재현한 기준선 안에서만** 성립합니다. 반면 clean에서 WLA보다 16점 높다는 결과는 과제 단위로 봐도 견고합니다(31승 12패).

## 4. 코드 이득의 대부분은 "계획 스트림 + ViT"에서 온다

![Ablation](images/ablation_ladder.png)
*실제 로봇 8과제 평균. 논문의 Fig. 5·7과 Table 3·4에 흩어진 변형을 한 줄로 세웠다.*

ablation을 한 줄로 세우면 이야기가 조금 달라집니다.

- 코드 전문가가 없을 때 32.9입니다.
- 코드 대신 **자연어 subgoal**을 넣으면 63.6이고, 코드를 넣으면 71.1입니다.
- 전체 이득 38.3점 중 "코드 대 자연어"의 몫은 7.5점(20%)입니다.
- 자연어 subgoal 변형만으로도 가장 강한 기준선(DreamZero 46.0)보다 17.6점 높습니다.

공개 코드를 보면 이유가 하나 더 있습니다. "코드 없음" 변형은 코드 전문가와 함께 **Qwen3-VL ViT 시각 입력까지** 사라집니다. 이 구조에서 ViT prefix는 코드 전문가에만 붙기 때문입니다. 남은 video와 action 전문가는 VAE 잠재만 봅니다. 그래서 32.9 → 71.1에는 "의미 수준 시각 입력이 있느냐"의 효과가 섞여 있습니다. 실제로 코드 전문가의 입력만 ViT에서 VAE로 바꿔도 15.6점이 떨어집니다(Table 3).

주석 없는 코드(62.8), 스킬 없는 코드(63.3), 자연어 subgoal(63.6)과 전체 코드(71.1)의 차이는 8과제 평균으로는 일관됩니다. 그러나 과제별로는 모두 표본 오차 상한 안에 있습니다. 시드별 값이 보고되지 않아 이 이상은 판단할 수 없습니다.

## 5. 하이브리드 실행과 "미리 보는 영상"

- **하이브리드의 random 이득(20.6 → 29.2)은 다섯 과제에 몰려 있습니다.** 이득의 65%가 Stack Blocks Two, Press Stapler 등에서 나오고, 과제 단위로는 24승 19패입니다. 공개 설정에서 두 모델은 학습 데이터만 다른 것이 아닙니다. 학습 step(200k vs 15k), 추론 단계(20 vs 40), 코드 디코딩 규칙도 다릅니다.
- **action이 보는 미래 영상은 꽤 흐립니다.** 논문 설정(4단계 joint denoising)에서 action이 마지막으로 읽는 video는 noise 수준 σ = 0.625입니다. "predictive guidance"는 선명한 미래가 아니라 덜 denoise된 video 전문가의 내부 특징입니다.
- **Noise 재표본화 기법은 그대로 쓸 만합니다.** 사전학습 video 분포에서는 표본의 95.5%가 σ > 0.9입니다. 그래서 하나의 혼합 분포에서 noise를 뽑고, video 손실과 action 손실에 서로 다른 중요도 가중치를 줍니다. forward를 늘리지 않고도 action이 덜 흐린 미래를 보게 하는 방법입니다. 가중치의 상한이 2라서, 논문의 clip 4.0은 실제로는 작동하지 않습니다.

## 6. 가져갈 것

1. **계획 스트림부터 넣고, 코드는 나중에 넣습니다.** 자연어 subgoal만으로 이득의 80%를 얻었습니다. 코드 라벨은 비용이 크므로 두 번째 단계로 미룹니다.
2. **ablation 표에 "같이 빠지는 것"을 적습니다.** 모듈 하나를 끌 때 시각 인코더 같은 입력 경로가 함께 꺼지는지 코드로 확인합니다.
3. **기준선 옆에 공식 리더보드 값을 같이 적습니다.** 직접 재현한 기준선이 공개 값보다 10점 이상 낮으면 원인을 먼저 찾습니다.
4. **video와 action을 함께 학습할 때는 손실별로 noise 분포를 나눕니다.** 추론 단계 수에 따라 action이 실제로 읽는 σ를 로그로 남깁니다.

---

*재계산 스크립트, 논문 ↔ 코드 대조표(D1–D11), 주장별 판정, 공개용 리포트 PDF는 함께 공개한 분석 폴더에 있습니다.*

## 자료

- [통합 리포트 PDF](report/report_public.pdf)
- [01. 논문 분석: Agentic World Action Models (AW-0)](analysis/01_paper_analysis.md)
- [02. 선행연구: AW-0는 어디에서 왔는가](analysis/02_prior_work.md)
- [03. 재계산과 해석](analysis/03_analysis.md)
- [04. 논문 ↔ 코드 대조와 주장 검증](analysis/04_code_verification.md)
