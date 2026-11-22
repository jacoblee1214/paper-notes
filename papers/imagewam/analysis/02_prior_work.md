# 02. 선행연구, 코드, 관련 분석

## 1. 학회와 라이선스 확인

| 항목 | 결과 | 출처 |
|---|---|---|
| 학회 | **CoRL 2026 포스터** | corl.org 공식 채택 목록(이 저장소 수집본, 2026-11-14): 포스터 post-p3 세션, 11/11 10:30–12:00, 포스터 12번. 공식 저장소 설명도 "[CoRL 2026]" |
| arXiv | v1 하나 (2026-06-17), 주석에 프로젝트 페이지 주소만 | arXiv API와 abs 페이지(2026-11-22) |
| 논문 라이선스 | arXiv abs 페이지의 "view license"가 `arxiv.org/licenses/nonexclusive-distrib/1.0/`을 가리킴 (arXiv 비독점 배포 라이선스, CC 아님) | arXiv abs 페이지 |
| PDF 저작권 표시 | ©, Copyright, All rights reserved 문구 없음. 소속에 Tencent Robotics X가 있으나 회사 고지문은 없음 | `pdftotext` 검사 |
| 그림 게시 | **불가**: CC BY가 아니므로 공개본에는 논문 그림을 넣지 않고 우리가 그린 그래프만 씁니다 | 위 라이선스 |

## 2. 계보

ImageWAM의 아이디어를 세 갈래의 합으로 읽을 수 있습니다.

1. **행동 조건을 영상 모델의 내부 상태에서 꺼내는 흐름.** 비디오 예측 정책(VPP), Cosmos Policy, LingBot-VA(인과 비디오 월드 모델), Motus, **Fast-WAM**. 이 중 Fast-WAM이 가장 가깝습니다. 미래 영상 토큰을 학습 때만 쓰고(공동 학습) 추론 때는 현재 관측과 지시문의 KV 캐시만 행동 전문가가 읽습니다. 논문 §3.2도 이 변형을 "Fast-WAM식"이라 부릅니다.
2. **이미지 편집을 로봇에 쓰는 흐름.** SuSIE(편집 확산으로 부분 목표 생성, 논문 인용 [59])가 대표입니다. 차이는 SuSIE류가 **편집된 이미지를 만들어 별도 정책에 넘기는** 반면 ImageWAM은 편집 모델 내부 캐시를 행동 전문가가 읽는다는 점입니다.
3. **MoT식 다중 전문가.** 공유 self-attention 위에 모달리티별 전문가를 두는 구조(BAGEL 계열의 BagelVLA, 구조 서술은 부록 §5.1). 행동 DiT를 백본 DiT 가중치의 보간으로 초기화하는 방식은 Fast-WAM과 LingBot-VA가 쓴 것과 같다고 부록에 적혀 있습니다.

### 구성 요소별 출처

| 구성 요소 | 출처 | 비고 |
|---|---|---|
| 행동 DiT와 MoT 어텐션 마스크(행동은 접두부를 한 방향으로 읽음) | Fast-WAM, LingBot-VA 구조 | 논문 부록 §5.1, [3, 13] 인용 |
| 접두부 KV 캐시와 행동만 반복 denoise | Fast-WAM | 코드의 `prefill_*_cache` 경로 |
| 편집 백본 | OmniGen2 [84], Ovis-U1 [85], FLUX.2 klein [86] | 세 가지 모두 공개 가중치 |
| 정지 비디오 대신 끝점 프레임 | 이 논문(편집 설정의 단순화) | 코드는 16프레임 뒤 프레임 하나 |
| 시간 이동 샘플링과 종형 손실 가중 | Fast-WAM 코드와 같은 `scheduler_continuous.py` 계열 | 논문에 없음 (`04` D3) |

### 가까운 대안과의 차이

| 대안 | 같은 점 | 다른 점 |
|---|---|---|
| Fast-WAM | 접두부 KV 캐시 + 행동 전문가, 영상 공동 학습 | 백본이 비디오 DiT인 대신 이미지 편집 DiT, 예측 대상이 여러 프레임 대신 끝점 프레임 1개 |
| LingBot-VA | 영상 WAM, 행동 DiT 보간 초기화 | 자기회귀 비디오, 인과 구조, 대규모 정책 사전학습 |
| Motus | 통합 잠재 행동 월드 모델 | 정책 사전학습 사용 |
| BagelVLA, UniVLA | 통합 이해·생성 모델에서 행동 | 이해 분기도 학습, 사전학습 사용 |
| SuSIE (편집 확산 부분 목표) | 편집 모델로 목표 이미지 | 목표 이미지를 만들어 별도 정책에 넘김 |

## 4. GitHub와 자료

| 저장소·자료 | 관계 | 라이선스 | 별 / 마지막 푸시 | 기준 커밋·크기 | 상태 |
|---|---|---|---|---|---|
| [yuyangalin/ImageWAM](https://github.com/yuyangalin/ImageWAM) | **공식 코드** (PyTorch, uv, Hydra, DeepSpeed) | MIT (`LICENSE` 파일). GitHub API는 NOASSERTION. 제3자 코드는 원 라이선스 유지 문구 포함 | 193 / 2026-07-30 | `5d4a341ed20a95cdb08f0293f3d44778b9a9e05a`, 소스 약 11 MB | 얕은 복제, 정적 읽기만 (`repo/`) |
| [zhangwenyao1.github.io/ImageWAM](https://zhangwenyao1.github.io/ImageWAM/) | 프로젝트 페이지 | 해당 없음 | | HTML 14 KB | 받아서 링크 확인 |
| Zhangwenyao1/ImageWAM | 같은 이름의 저자 계정 저장소(별 0, 2026-06-18, 119 MB, 라이선스 없음) | 없음 | 0 / 2026-06-18 | 공식 저장소와 별개로 보임 | 복제하지 않음 |
| Zhangwenyao1/EasyWAM | 저자 계정의 WAM 학습·평가 통합 프레임워크(설명: "A Unified and Efficient Framework for Training and Evaluating World Action Models") | 없음 | 0 / 2026-09-02 | 2.3 MB | 읽지 않음 |
| HF `yuyangalin/ImageWAM-FLUX.2-4B-LIBERO`, `-4B-RoboTwin`, `-9B-LIBERO` | 공개 체크포인트(FLUX.2 klein 기반) | 모델 카드에 라이선스 표기 없음 | | `model.pt` 9.04 GB, 9.04 GB, 57.57 GB | 받지 않음 (크기만 확인) |
| HF `yuyangalin/ImageWAM-FLUX.2-4B-InternData-A1-EE`, `-4B-RoboTwin-C2R-FT` | InternData-A1로 사전학습한 체크포인트와 그 RoboTwin 미세조정본(2026-07-29) | 표기 없음 | | 각 9.04 GB | 받지 않음 |
| [black-forest-labs/flux2](https://github.com/black-forest-labs/flux2) | 의존성(고정 커밋 `50fe516…`) | Apache-2.0 | 2,889 | | 사용 안 함 |
| [VectorSpaceLab/OmniGen2](https://github.com/VectorSpaceLab/OmniGen2), 저자 포크 yuyangalin/OmniGen2 | 의존성(패치 한 개 추가된 포크) | Apache-2.0 (원본) | 4,121 | | 사용 안 함 |
| HF FLUX.2-klein-base-4B / 9B | 백본 가중치 | 4B Apache-2.0, 9B는 "other"(게이트됨, 조건 확인 필요) | | | 받지 않음 |
| [yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) | 기준선과 데이터 전처리의 출처 | MIT | 1,670 / 2026-08-20 | | 기준선과 전처리 데이터의 출처 |

- 공식 저장소 README는 FLUX.2(권장), OmniGen2, Ovis-U1 세 변형과 LIBERO, LIBERO-Plus, RoboTwin 학습·평가 스크립트를 제공합니다. 실로봇 코드는 "공개 예정" 체크리스트에 미완료로 남아 있습니다(`04` D12).
- 저장소의 가장 최근 변경(2026-07-30)은 InternData-A1 사전학습 체크포인트 소개입니다. 논문 v1에는 없는 결과입니다(`03` §4).
- 같은 저자들의 이전 연구: DreamVLA(저자 Wenyao Zhang, 별 378), VLA-JEPA, MaskWAM, 분리된 순·역동역학 사전학습 등이 논문 참고문헌에 있습니다([8, 11, 15, 39, 51]).
