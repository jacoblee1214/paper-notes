# 02. 선행연구와 계보

번호 [n]은 논문 v2의 참고문헌 번호입니다. GitHub 정보는 2026-11-09에 `gh api`로 확인했습니다. BA-T 코드만 clone해서 읽었고, 나머지 저장소는 메타데이터와 라이선스 파일만 봤습니다. 선행 논문 본문은 따로 열어 검증하지 않았습니다.

## 1. 계보도

```
[고전 BA·SLAM: 명시적 변수, 풀이기]
  희소 특징 BA [1, 9, 22, 24, 29] · 직접법·광도 BA [2, 12–14]
        │ 미분 가능한 BA를 신경망 안에
        ▼
  BA-Net [33] (특징 공간 BA를 unroll)     RAFT [34] (상관 볼륨 + GRU로 광류 반복 정제)
        └───────────────┬───────────────┘
                        ▼
  DROID-SLAM [35] (RAFT식 갱신 + 조밀 BA 층, 명시적 Gauss-Newton)  →  DPVO (희소 패치판)
                                                         … 표 4의 고전 BA 비교 대상

[feed-forward 3D 복원: 풀이기 없이 한 번에]
  DUSt3R [42] (두 장 → 공유 좌표계 point map, 12층 cross-attn 디코더 2개, 227M)
    → MASt3R [18] (매칭 헤드)   → MUSt3R [5], 순차 확장 [38, 41]
    → 시점별 local point map: ViSTA-SLAM [49] (같은 1저자, 113M), [26]
  다시점 학습: VGGT [39] (605M) · MapAnything [17] (171M) · Depth Anything 3 [19] (765M) · SLAM-Former [48]

[잠재 표현의 반복 정제]
  ReSplat [44] (Gaussian splat 잠재 표현을 recurrent 갱신)

          ▼ 두 갈래를 합침: "BA의 정보 흐름" + "feed-forward 토큰"
  ★ BA-T: ViSTA의 인코더·local point map 설정 위에, 깊은 디코더 대신
          잔차 → 카메라 → geometry 순서로 연결한 Transformer 층 하나를 반복
```

**차별점 요약:** DROID-SLAM 계열은 학습된 특징으로 잔차를 만들되 **갱신은 명시적 BA 풀이기**가 합니다. DUSt3R 계열은 풀이기 없이 **깊은 attention 스택이 암묵적으로** 일관성을 맞춥니다. BA-T는 그 중간입니다. 풀이기도 명시적 변수도 없지만, attention 연결을 BA의 정보 흐름(잔차를 카메라에 모으고, 카메라로 geometry를 고침)대로 제한합니다. 부록 F.1은 BA-Net(명시적 변수 unroll), RAFT(광류장 정제), ReSplat(렌더링 변수 정제)과 갱신 대상이 다르다고 구분합니다.

## 2. 구성 요소별 출처

| 역할 | 출처 | 이 논문에서의 쓰임 |
|---|---|---|
| BA 정규방정식과 Schur 보완 | 고전 BA [1, 16, 23] | 식 2–3. 갱신 **순서**만 가져옴 |
| 인코더, local point map 설정, 학습 데이터 조합 | ViSTA-SLAM [49] | 사전학습 ViT-L로 초기화, 같은 6개 데이터셋 |
| 패치 임베딩, ViT 블록, RoPE2D, point map 헤드 | DUSt3R [42] / CroCo (Naver) | 코드에 그대로 들어 있음(CC BY-NC-SA 4.0 파일) |
| 카메라 조건 변환 | DiT [25]의 adaLN | `ConditionModulationBlock` + `ModLN` |
| soft correspondence | attention 점수 = 매칭 확률 (MASt3R·RAFT 계열의 상관과 같은 발상) | 식 5, 8 heads |
| 반복 감독 Σα^{K−k}L^(k) | RAFT [34]의 시퀀스 손실(γ^{N−i})과 같은 형태 | 식 11, α = 0.8 (RAFT는 γ = 0.8) |
| 고전 BA 비교 | DROID-SLAM [35] | 표 4, Fig. 8 |
| 다시점 pose graph 최적화 (코드에만) | 신뢰도 가중 PGO | `ba_t/pose_graph.py` (`04` D11) |

반복 감독이 RAFT의 시퀀스 손실과 같은 형태이고 감쇠율까지 0.8로 같다는 점은 논문에 적혀 있지 않습니다(RAFT의 γ = 0.8은 내 사전 지식).

### 2.1 GitHub

| 저장소 | 역할 | 라이선스 | ★ | 마지막 push | 확인한 커밋 |
|---|---|---|---|---|---|
| [zhangganlin/BA-T](https://github.com/zhangganlin/BA-T) | **공식 코드** (추론 전용, 2·다시점 데모, PGO) | Apache-2.0 + CC BY-NC-SA 4.0 (Naver 파생 파일) | 49 | 2026-10-22 | `406d791` (2026-10-21, shallow clone) |
| HF [zhangganlin/BA-T](https://huggingface.co/zhangganlin/BA-T) | 가중치 `bat.pth`(기본, 추가 미세조정), `bat_paper.pth`(논문 수치) | CC BY-NC-SA 4.0 (HF 태그) | – | 2026-10-22 | `ac53e30`, 각 1.378 GB |
| [zhangganlin/vista-slam](https://github.com/zhangganlin/vista-slam) | ViSTA-SLAM [49], 인코더·기준선 출처 | Apache-2.0 + CC BY-NC-SA 4.0 | 293 | 2026-04-05 | `b13ac44` |
| [naver/dust3r](https://github.com/naver/dust3r) | DUSt3R [42], 코드 블록 출처 | CC BY-NC-SA 4.0 | 7,368 | 2025-09-24 | `4c24a6e` |
| [naver/mast3r](https://github.com/naver/mast3r) | MASt3R [18] | CC BY-NC-SA 4.0 (NOASSERTION 표시) | 3,161 | 2025-06-30 | `f5209af` |
| [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | VGGT [39] | VGGT License (자체 약관) | 14,610 | 2026-05-19 | `a288dd0` |
| [facebookresearch/map-anything](https://github.com/facebookresearch/map-anything) | MapAnything [17] | Apache-2.0 | 3,897 | 2026-08-07 | `3d10cf7` |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | DA3 [19] | Apache-2.0 | 6,659 | 2026-07-27 | `3d835ec` |
| [princeton-vl/DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | DROID-SLAM [35], 표 4 비교 | BSD-3-Clause | 2,725 | 2025-05-05 | `2dfd39f` |
| [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT) | RAFT [34] | BSD-3-Clause | 4,123 | 2025-08-24 | `2888e15` |
| [princeton-vl/DPVO](https://github.com/princeton-vl/DPVO) | DROID 계열 희소판(논문 미인용, 참고) | MIT | 1,148 | 2026-09-20 | `0ac95b6` |
| [frobelbest/BANet](https://github.com/frobelbest/BANet) | BA-Net [33] | 표시 없음 | 314 | 2025-04-20 | `4b6e6f3` |
| ReSplat [44] | 잠재 Gaussian 반복 정제 | – | – | – | 공식 저장소를 찾지 못함 |

MASt3R의 라이선스는 GitHub API가 NOASSERTION으로 표시하지만 DUSt3R과 같은 CC BY-NC-SA 4.0으로 알려져 있습니다(LICENSE 본문은 열어 보지 않음).

## 3. 이 저장소의 다른 논문과의 관계

BA-T는 로봇 학습 논문이 아니라 3D 복원 모델입니다. 아래 관계는 "조작·real2sim 파이프라인의 기하 앞단(front-end)으로 쓸 수 있는가"라는 관점에서만 성립하고, 직접 비교 대상은 없습니다.

| 논문 | 관계 |
|---|---|
| **RoboSimStudio** (`robosimstudio/`) | Real2Sim2Real 파일럿(§4.3)에서 코딩 에이전트가 **보정 카메라도 CAD도 없는 스마트폰 영상 하나**로 장면을 만들었습니다. 이 단계의 장면 기하·카메라 추정을 BA-T 같은 feed-forward 복원기로 시작할 수 있습니다. 하지만 BA-T의 출력은 224×224 point map, **쌍마다 임의 스케일**, 실내·정적 학습 분포입니다. 물리 시뮬레이션에 필요한 메트릭 스케일과 물체 단위 메쉬는 주지 않으므로, 스케일 기준(알려진 물체 크기, 깊이 센서)과 메쉬화 단계를 따로 붙여야 합니다. RoboSimStudio의 병목은 기하보다 물성·접촉 모델 식별이라, BA-T가 해결하는 문제와 겹치는 부분은 작습니다. |
| **ZeroDex** (`zerodex/`) | 보정된 RGB 카메라 4대의 시점 쌍 삼각측량과 재투영 합의가 핵심입니다. BA-T는 보정 없는 시점 쌍의 상대 포즈를 줄 수 있어 "보정 생략" 대안처럼 보이지만, **정밀도가 맞지 않습니다.** AUC@τ = E[max(0, 1 − e/τ)]이므로 표 2의 회전 AUC@5 = 0.683은 모든 쌍의 오차가 같다고 쳐도 약 1.6°이고, 5°를 넘는 쌍은 0으로 잘려 들어갑니다. 표 6의 4-view 절대 회전 오차는 16.39°입니다. 삼각측량용 외부 파라미터에는 서브-도 수준이 필요하므로, BA-T는 체커보드 보정을 대신할 수 없습니다. 쓸 수 있다면 카메라가 움직인 뒤 보정이 깨졌는지 감지하는 정도입니다(내 판단). |
| **CHORD** (`chord/`) | 사람 영상 복원 끝단(App. C)에서 Gaussian splat과 함께 **DROID-SLAM**으로 공동 정제합니다. BA-T 표 4는 두 장 입력에서 DROID-SLAM의 BA보다 BA-T가 강건하다고 보고하므로, few-view 구간의 초기화로 넣을 후보입니다. 다만 표 4는 DROID-SLAM이 불리한 두 장 조건이고, 긴 영상의 전역 일관성은 BA-T가 다루지 않습니다. |
| 반복 정제·잔차 설계 일반 | VLA나 제어 정책의 "반복 정제 헤드"(예: diffusion·flow 행동 헤드의 반복 단계)에 **명시적 잔차 토큰**을 두는 설계 원칙으로 옮길 수 있습니다. 표 3 (b)→(h)가 그 근거입니다. |

## 4. 논문이 다루지 않았지만 관련 있는 연구 (내 판단)

| 주제 | 관련성 | 근거 수준 |
|---|---|---|
| **Deep Equilibrium Models (DEQ)** | 가중치를 공유한 층을 반복 적용하는 구조는 고정점 반복입니다. BA-T는 K = 4–8로 unroll해 학습하지만, README는 공개 가중치를 4–24회 반복으로 학습했다고 밝힙니다. 반복을 늘려도 발산하지 않는 이유(코드의 post-LN 갱신, `04` D5)를 고정점 관점에서 분석할 수 있습니다. | 내 해석 |
| **Universal Transformer / looped transformer** | 층 공유와 반복 깊이 조절의 일반 계보입니다. BA-T의 차별점은 반복 자체가 아니라 반복 안의 연결 구조라는 점이 표 3 (b)와 (h)로 분리됩니다. | 내 사전 지식, 서지 확인 필요 |
| **Learned optimizers / LISTA류 unrolling** | "풀이기 단계를 신경망 층으로 바꾼다"는 계보입니다. BA-T는 반복 구조만 가져오고 풀이기 수식은 버렸다는 점에서 LISTA보다 느슨합니다. | 내 사전 지식 |
| **DPVO** | DROID-SLAM의 희소 패치판으로, 비교했다면 표 4보다 공정한 고전 기준선이었을 수 있습니다(희소 패치가 BA-T의 토큰 단위와 비슷함). | 내 해석 |

**한계:** 4절과 2절의 RAFT γ 값은 내 사전 지식에 근거합니다. 게시 전에 원문을 확인해야 합니다.
