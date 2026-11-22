# 02. 선행 연구와 관련 자료

이 문서의 계보는 논문 2절(관련 연구), 부록 E.1, 인용 목록에서 가져왔고, 논문이 직접 인용하지 않은 연결은 "우리 판단"으로 표시했습니다.

## 1. 계보

```
잠재 행동 모델                    미래 예측을 정책에 연결                 효율적 WAM
Schmidt & Jiang 2312.10812 [18]   FLARE 2505.15659 [24]                    Fast-WAM 2603.16666 [28]
Genie 2402.15391 [16]             Mantis 2511.16175 [6]  (보조 감독)        GigaWorld-Policy 2603.17240 [11]
LAPA 2410.11758 [19]              VLA-JEPA 2602.10098 [27] (보조 감독)
CoMo 2505.17006 [20]              World Guidance 2602.22010 [26]
UniVLA 2505.06111 [33]            Being-H0.7 2605.00078 [25]
AdaWorld 2503.18938 [34]          DIAL 2603.29844 [7]
DreamDojo 2602.06949 [35]         LDA-1B 2602.12215 [32]  (DINO 잠재 + 확산 결합)
Garrido 외 2601.05230 [36]  ->  LaWAM: 디코더를 추론 인터페이스로 남기고 û_T를 행동 전문가의 교차 어텐션 문맥으로
                           픽셀 WAM: Cosmos-Policy [9], Motus [10], LingBot-VA [12], Mimic-video [8], WoVR [29], π0.7 [30] (시각 부분 목표)
```

### 1.1 LaWAM이 서 있는 자리 (논문 2절, E.1)

1. **잠재 행동 모델 계열.** LAPA, UniVLA, Genie는 잠재 행동 공간 자체(이산·연속, 정규화, 전이)를 연구하고 **디코더는 사전학습 뒤 버립니다** (1절, 2절). LaWAM은 디코더를 남깁니다. 논문은 이 관찰을 Garrido 외 [36]이 동시에 했다고 인정합니다 (2절 마지막 문단).
2. **보조 감독 계열.** Mantis [6]와 VLA-JEPA [27]는 학습 질의로 미래 특징을 예측해 질의 표현을 풍부하게 만들고 미래 예측기는 추론에서 쓰지 않습니다. 논문은 이를 그림 9(b)로 대비하고, 표 4의 "Future Prediction" 변형이 이 방식에 해당한다고 설명합니다 (4.5절).
3. **픽셀 WAM 계열.** 미래 영상을 생성해 조건으로 쓰거나 공동 생성합니다. 지연과 파라미터가 큽니다 (표 1: Motus 3,231 ms, Cosmos-Policy 1,413 ms, LingBot-VA 4,482 ms). Fast-WAM [28]과 GigaWorld-Policy [11]는 미래 예측을 보조 또는 행동 중심으로 줄여 비용을 낮추는 대신 "예측한 동역학이 테스트 때 명시적 조건 신호로 작동하는 역할이 약해진다"고 논문은 평가합니다 (2절).
4. **잠재 공간의 세계 모델.** LDA-1B [32]도 DINO 잠재 공간에서 동역학을 모델링하지만 미래 시각 상태와 행동 청크를 확산 방식으로 함께 노이즈 제거합니다. LaWAM은 비반복 결정적 디코딩입니다 (2절). Nilaksh 외 [31]는 사전학습된 시각 잠재가 픽셀 재구성 공간보다 정책에 유리한 동역학 표현이라는 동시기 분석으로 인용됩니다.

### 1.2 우리 판단

- LaWAM의 실질적 설계 선택은 세 가지입니다. (a) 잠재 행동을 정책이 **회귀**로 예측(점 추정, 불확실성 없음), (b) 미래를 청크 **끝점 하나**로만 표현, (c) 부분 목표를 교차 어텐션 **문맥**으로만 주입. 세 가지 모두 논문 본문 절제에서 독립적으로 시험되지 않았습니다. (a)의 대안(분포 예측), (b)의 대안(여러 중간 프레임), (c)의 대안(입력 연결)의 비교는 없습니다. 저장소 이슈 12에서 저자는 (c)에 대해 "VLM 의미 특징과 DINO 동역학 특징의 표현 차이 때문에 공유 교차 어텐션이 학습을 어렵게 할 수 있어 경로를 분리했다"고 동기를 설명했고, 비교 결과는 제시하지 않았습니다(읽은 범위, `04` §5).
- 표 4에서 가장 큰 단일 효과는 "LaWM을 쓰지 않는 기본 VLA(91.0) 대 Stage-2 없는 변형(96.4)"의 5.4 pp이지만, 이 차이에는 미래 예측 감독, 잠재 행동 증류, 부분 목표 조건, 지식 격리가 모두 섞여 있습니다. 부분 목표를 **조건으로 넣는 것** 하나의 효과는 1.8 pp입니다 (`03` §6).

## 2. 구성 요소별 출처와 가까운 대안

| 구성 요소 | 출처 | LaWAM에서의 쓰임 | 가까운 대안 |
|---|---|---|---|
| 동결 시각 인코더 | DINOv3 [17] (ViT-B/16 증류판) | LAM 특징과 부분 목표 공간 (3.1절, E.2) | V-JEPA 2 [46]는 인코더 구조 참고 (E.2: "V-JEPA2-style spatiotemporal design"). 이슈 8의 제3자는 VGGT-1B로 교체를 시도 |
| 잠재 행동 모델 | Genie [16], LAPA [19] | 인코더가 (u, u_T) → z, 디코더가 (u, z) → ũ_T (식 2) | Garrido 외 [36]: 같은 관찰. 디코더 조건화는 Genie의 덧셈 대신 adaLN (E.2) |
| adaLN 조건화 | DiT 계열 | z를 디코더 블록에 주입, 게이트 영초기화 (공식 코드 `lam_decoder.py`) | Genie의 덧셈 주입 |
| 보조 상태 예측 | UniVLA 계열의 상태 정렬과 같은 발상 (우리 판단) | z가 형태 정보를 담도록 (3.2절) | 없음 |
| VLM 백본 | Qwen3-VL-2B 첫 16층 (E.3) | 문맥과 질의 토큰 | StarVLA 프레임워크의 Qwen-GR00T 구성 |
| 행동 전문가 | GR00T N1/N1.6 [5]의 Alternate-DiT, 조건부 플로 매칭 | 부분 목표와 VLM 문맥에 번갈아 교차 어텐션 | 입력 연결 (논문에 비교 없음) |
| 지식 격리 | Driess 외 [37] | û_T에 stop-gradient (3.3절) | 없음 |
| 물리 시간 정렬 | 연속 시간 위치 부호화 | T = round(τh), 시각 t_i = i/h (E.4) | 인덱스만 쓰는 부호화 (그림 10에서 비교) |
| 사전학습 데이터 | EgoDex [38], Ego4D [39], LEGO [40], AgiBot World [41], RoboMind [42], RoboCOIN [43], Open X-Embodiment [44], DROID [45] | LAM: 로봇 3,000 시간 + 사람 1,500 시간. 정책: 로봇만 | |

## 4. 공식 저장소와 자료

2026-11-22 조회 (GitHub API, HF API). 코드는 얕은 복제를 `repo/`에 두고 **정적으로만** 읽었습니다(실행, 설치, import 안 함).

| 항목 | 위치 | 라이선스 | 별 / 포크 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| 공식 코드 | github.com/RLinf/LaWAM, 기본 브랜치 main, 커밋 `c12168d` (2026-09-17, "docs: add project news") | **LICENSE 파일 없음.** GitHub API 라이선스 null. `pyproject.toml`은 MIT, README는 "StarVLA의 MIT 라이선스를 유지"라고 적음 | 167 / 13 | 2026-10-29 (gh-pages 브랜치) | `repo/` 정적 읽기 |
| 프로젝트 페이지 | rlinf.github.io/LaWAM/ (gh-pages) | 표기 없음 | | 2026-10-29 | 읽음 |
| 블로그 | nemo-1024.github.io/blogs/lawam/ | 표기 없음 | | | 링크만 확인 |
| HF 체크포인트 (LAM) | huggingface.co/jialei02/lawam_lam | HF 메타 MIT | | 2026-06-29 | 2.90 GB (`pytorch_model.pt`). **받지 않음**. 설정 YAML(2 KB)만 읽음 |
| HF 체크포인트 (사전학습) | jialei02/lawam_pretrain | MIT | | 2026-06-29 | 7.17 GB. 받지 않음. `config.yaml`(1.8 KB)만 읽음 |
| HF 체크포인트 (LIBERO SFT) | jialei02/lawam_libero_sft_release | MIT | | 2026-06-29 | 7.17 GB. 받지 않음 |
| HF 체크포인트 (RoboTwin SFT) | jialei02/lawam_robotwin_sft_release | MIT | | 2026-06-29 | 7.17 GB. 받지 않음 |
| HF 데이터 (LIBERO) | jialei02/libero_merged_no_noops_20hz | MIT | | 2026-06-29 | 1.97 GB, 파일 17개. 받지 않음 |
| HF 데이터 (RoboTwin) | jialei02/robotwin_merged | MIT | | 2026-06-29 | 78.99 GB, 파일 468개. 받지 않음 |
| 기반 코드 | github.com/starVLA/starVLA | NOASSERTION (GitHub API) | 3,971 / 524 | 2026-11-18 | 확인만 |
| 공통 인프라 | github.com/RLinf/RLinf | Apache-2.0 | 5,690 / 855 | 2026-11-21 | 확인만 (프로젝트 페이지 도메인 rlinf.github.io) |
| LeRobot 통합 | huggingface/lerobot `docs/source/lawam.mdx` | LeRobot 라이선스 | | | 파일 존재 확인 (8,456 바이트). README의 "공식 통합" 주장은 문서 파일 존재 수준에서 확인 |
| 관련 기준선 코드 | Robbyant/lingbot-va (Apache-2.0, 별 1,989), yuantianyuan01/FastWAM (NOASSERTION, 별 1,670), ginwind/VLA-JEPA (Apache-2.0, 별 636), OpenDriveLab/UniVLA (Apache-2.0, 별 1,145), LatentActionPretraining/LAPA (MIT, 별 587) | | | | 메타데이터만 |

- README의 "출시 3개월 만에 인용 23회" 문구는 Semantic Scholar 링크를 근거로 하며 이 폴더에서 확인하지 않았습니다.
- 이슈는 12개(열린 것 1개, 닫힌 것 11개)이고, 재현과 관련된 것은 `04` §5에 정리했습니다. 이슈 본문은 제3자 보고이므로 미검증으로 표시합니다.

## 5. 학회 확인

- `conferences/corl2026/papers.json` (corl.org 공식 채택 목록에서 2026-11-14 수집): key 2606.15768, forum_id mw8BgE8ADo, **presentation: poster**, also_poster: false, session "Poster post-p2", 2026-11-10(화) 5:00–6:30 PM, 포스터 번호 1, in_official: true, match: title_exact.
- arXiv 댓글: "Accepted at CoRL 2026. Project website: …". 프로젝트 페이지 상단에 "CoRL 2026" 표기.
- 결론: **CoRL 2026 포스터 (공식 목록 확인).** 웹 검색은 쓰지 않았습니다.
