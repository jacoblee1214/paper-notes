# 02. 선행연구와 관련 저장소

이 문서는 StressDream 의 구성 요소가 어디서 왔는지, 가까운 대안이 무엇인지, 코드와 프로젝트 자료가 어디에 있는지를 정리합니다. 대괄호 번호는 논문 참고문헌 번호(`paper/paper.txt` 의 References)입니다.

## 1. 선행연구 계보 (구성 요소별 출처)

| StressDream 의 요소 | 출처 (논문 번호) | 관계 |
|---|---|---|
| 행동 조건부 영상 세계모델 | Vista [11] (구동), Ctrl-World [12] (조작, SVD 기반), Cosmos [1], Wan [2], GEM [76], GAIA-2 [79] | 조종 대상 WM 은 Vista 와 Ctrl-World. 둘 다 확산 기반이고 초기 잡음이 결과를 정하는 결정적 샘플러(A.2) |
| 정책 평가에 WM | WorldGym [15], WorldEval [21], Veo 기반 Gemini Robotics 평가 [16], Cosmos Policy [22], Ctrl-World [12] | 기존은 공칭 상상을 씀(§1). StressDream 은 같은 WM 에 비관적(최악 가능성) 상상을 추가 |
| 정책 개선에 WM | VLAW [13] (가중 회귀), World-Gymnast [17] (WM 안 RL), DreamerV3 [71], DINO-WM [26] (샘플링 기반 계획) | §6.2 는 VLAW 의 가중 회귀 틀을 쓰되 가중을 조종 결과로 정함 |
| 초기 잡음 최적화 | DNO [19] (Tang et al., ICML 2025), 모션 DNO [30], ReNO [31] (Eyring et al., NeurIPS 2024), 노름 유도 탐색 [18], 희귀 개념 생성 [36], 잡음 하이퍼네트워크 [44], 붕괴 복구 [37] | 같은 틀(고정된 생성 모델, 잡음만 최적화)을 대규모 영상 WM 에 적용. 이전 연구는 대부분 1단계 증류 생성기에서 역전파 |
| 근사 기울기 | "A noise is worth diffusion guidance" [45] (Ahn et al., ICLR 2026) | 디노이저 야코비안의 비례·대각 근사와 멀티스텝 점수 증류. 식 (7)과 B.4의 근거 |
| 타이피컬 셋, 잡음 규제 | Cover [33], Betancourt [34], Nalisnick [35] (타이피컬 셋 OOD), 노름 항 [18, 19, 31], 스펙트럼 항 [38–42] (Durall; Hwang et al. v2 에서 추가) | 노름 항은 선행 연구와 같고, 블록 등방성·스펙트럼 항을 더해 세 항으로 확장 |
| VLM 보상, 검증기 | Qwen-VL [32], Qwen3-VL [54], Wolf [51] (드라이빙 캡셔닝), X-CLIP [52], 보상 모델 Robometer [58], RoboDopamine [57], 시각 보상 모델링 [121] | VLM 의 단일 토큰 로그 확률을 미분 가능한 보상으로 사용 |
| Best-of-N, 분류기 안내 | Best-of-N(표본 중 최고 점수 선택, 논문의 기준선), 분류기 안내 [46, 47] | 비교 기준선. Classifier Guidance 는 원샷 깨끗한 latent 근사 [47] 로 구현(C.1) |
| 평가 지표 | WorldModelBench [55] (지시 따르기 0–3, 물리 0–5, 상식 0–2), WorldLens [131], Gemini-3.0 [120], 사람 판정 | 드라이빙은 최적화에 쓰지 않은 지표, 조작은 사람 라벨 |
| 안전, 강건 계획 | 잠재 도달가능성 안전 [73], AnySafe [74], UNISafe [81], 강건 RL [28], 위험 인식 로보틱스 [29] | 이 논문의 이중 문제(식 4)는 최악 사례 계획의 영상 WM 판. 이전 안전 필터는 과제별 실패 분류기를 학습 |
| 정책 | π0.5 [20] (π0.5-DROID, 가중 흐름 매칭 [13]), DROID [53], openpi | 정책 개선의 대상 |

## 2. 가까운 대안과 차이

| 대안 | 어떻게 다른가 |
|---|---|
| 무작위 표본(Best-of-N) | 계산량 N 에서 이득은 표본 한 방향 기대 최댓값 수준(`03` V10). 조종은 기울기를 쓰므로 같은 N 에서 더 높은 보상. 단, 계산량 일치 정의는 `03` K1 |
| 분류기 안내 | 디노이징 경로 자체를 바꿔 OOD 를 만듦(논문 Dubins 와 표 4). 초기 잡음 최적화는 모델을 건드리지 않음 |
| 보상 기울기 미세조정, DPO류 | 보상마다 모델을 바꿈(A.4). 조종은 추론 시점 프롬프트로 지정 |
| SMC·입자 기반 | 수 분짜리 영상 생성을 입자 수만큼 반복. 조종은 1차 기울기 |
| 학습된 실패 분류기(잠재 안전 필터 [73, 74, 81]) | 과제별 분류기와 데이터 필요. 이 논문은 VLM 프롬프트로 사건을 지정(일반적이지만 VLM 품질에 의존) |
| WM 입력 공격(적대적 조종) | 같은 모델의 잡음·조건을 최적화한다는 점은 비슷하나 목적이 반대(공격자 대 평가자). 이 논문은 "그럴듯함" 제약을 두고 공격적 이탈은 막는다고 주장 |

## 4. 코드와 자료 (GitHub 표, 확인 2026-11-25)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **공식 코드 (StressDream)** | https://github.com/CMU-IntentLab/StressDream | MIT | 별 27, 포크 1, 커밋 6, 마지막 push 2026-06-02, 최신 `b16fa06` | `repo/` 얕은 복제(정적 읽기만). Dubins·Vista·Ctrl-World 데모. 논문 v2 에는 코드 링크가 없고 v1 부록과 프로젝트 페이지에는 있음 |
| 프로젝트 페이지 | https://stressdream.github.io/ | - | - | 영상 결과, Dubins 실험과 타이피컬 셋 설명 하위 페이지. arXiv·코드 링크 |
| 허브 체크포인트 | https://huggingface.co/junwon-seo/StressDream | 표기 없음 | 최종 수정 2026-11-05 | Ctrl-World 과제별 체크포인트 6개(각 9.28 GB), 과제별 궤적 10개. 내려받지 않음 |
| Vista | https://github.com/OpenDriveLab/Vista | Apache-2.0 | 별 913, 마지막 push 2025-07-02 | 가중치 vista.safetensors 약 10 GB(허브) |
| Ctrl-World | https://github.com/Robert-gyj/Ctrl-World | MIT | 별 586, 마지막 push 2026-04-08 | 프로젝트 페이지 ctrl-world.github.io |
| openpi (π0.5-DROID) | https://github.com/Physical-Intelligence/openpi | Apache-2.0 | 별 14,748, 마지막 push 2026-08-24 | 정책 개선의 대상 정책. 정책 개선 코드는 StressDream 저장소에 없음 |
| 데이터: PhysicalAI-Autonomous-Vehicles | https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles | NVIDIA 라이선스(허브 표기 "other") | 최종 수정 2026-11-13 | 대용량. 내려받지 않음 |
| 데이터: Nexar 충돌 예측 | https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction | 미확인 | - | 내려받지 않음 |
| VLM: Qwen3-VL-4B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct | Apache-2.0 | - | 조작 보상. 약 8.9 GB |
| VLM: Qwen2.5-VL(Wolf 미세조정) | https://huggingface.co/WoWolf/Qwen2_5vl-7b-fm-tuned | 허브 표기 "other" | - | 드라이빙 보상(옵션). 약 47 GB(허브 저장 용량) |
| VLM: X-CLIP base-patch32 | https://huggingface.co/microsoft/xclip-base-patch32 | MIT | - | 드라이빙 기본 보상. 약 1.6 GB(허브 저장 용량) |

원 논문의 코드 링크는 v1 부록에 `github.com/CMU-IntentLab/StressDream`, 프로젝트 페이지는 v1 `junwon.me/StressDream`, v2 `stressdream.github.io` 입니다(`01` §6).
