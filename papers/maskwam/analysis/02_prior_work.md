# 02. 선행연구와 관련 코드

## 1. 계보

MaskWAM은 네 갈래를 한 모델에 묶습니다(논문 §1–2, 참고문헌 번호는 논문 기준).

| 갈래 | 대표 | MaskWAM에서의 위치 |
|---|---|---|
| 비디오 확산 기반 정책·WAM | UniPi [13], GR-2 [11], VPP [18], UWM [67], DreamGen [21], Motus [4], LingBot-VA [25, 26], Cosmos Policy [23], Fast-WAM [59], mimic-video [40] | 백본 계열. 미래 RGB 복원만 감독하고 과제 관련 영역을 따로 강조하지 않는다는 것이 논문의 출발 문제 |
| 부분 디노이징으로 지연 줄이기 | mimic-video [40], Fast-WAM [59] | 식 3 이후의 추론 설계가 [40]과 비슷하다고 논문이 밝힘(§3.3). 영상 스트림에 1스텝만 적용 |
| 시각 프롬프트 | MOKA [33, 34], RT-Trajectory [15], RoboGround [19], KITE [45], Point What You Mean [57], ControlVLA [27] | 점, 상자, 마스크를 정책 입력으로 주는 방식. 논문은 "프레임마다 마스크가 필요한 방식 [27]"과 다르게 첫 프레임에만 준다고 구분 |
| 중간 표현 예측 | 키포인트 [16, 20, 49, 60], 궤적 [3, 17, 24], 객체 흐름 [50, 54], 어포던스 [1, 22, 38, 39], 마스크 [36] | 마스크는 "고립된 목표"로 다룬 Mask World Model [36]과 대비해, 입력과 출력을 함께 쓴다고 서술 |

가장 가까운 선행은 **Mask World Model [36]**(arXiv:2604.19683)입니다. 논문은 §4(RoboTwin 문단)에서 Mask-only 변형이 이와 "개념적으로 일치한다"고 쓰고, 두 방법의 직접 비교는 하지 않습니다.

## 2. 구성 요소별 출처

| 요소 | 출처 | 비고 |
|---|---|---|
| 비디오 백본, VAE | Wan 2.2 [48] | 5B 모델(TI2V), VAE 시간 4배·공간 16배 압축. 코드 설정 `Wan2.2-TI2V-5B` |
| MoT 결합 어텐션 | Mixture-of-Transformers [29], Motus [4], Fast-WAM [59] | 비디오는 행동을 보지 못하고 행동은 비디오와 행동을 본다 |
| 채널 연결로 모달 추가 | 코드 주석이 "Aether-style"로 지칭 (코드 `mask_wan_mot_sim` 문서열) | 논문 본문에는 이 인용이 없음 |
| 타임스텝 분리, 부분 디노이징 | mimic-video [40] | 논문 §3.3 |
| 분할과 마스크 추적 | SAM 3 [8], Qwen3-VL [41] (언어 → 대상 객체 파싱) | 주석 파이프라인(부록 B.1)과 배포 시 첫 프레임 마스크 |
| 언어 인코더 | T5 [42] | 동결 |
| 기준선 π0-mask | π0 [6]에 SigLIP 임베딩 덧셈(식 4) | 저자 설계(부록 A.3) |

## 4. GitHub와 코드

| 항목 | 주소 | 라이선스 | 상태 (2026-11-24 확인) | 비고 |
|---|---|---|---|---|
| 공식 코드 (프로젝트 페이지 Code 버튼) | `github.com/hanyangyu1021/maskwam` | MIT (LICENSE), Wan2.2 가져온 부분은 Apache-2.0, MaskWAM-Seg는 RoboInter-Tools(MIT) 기반 | ★51, 포크 2, 열린 이슈 3, 커밋 7개, 마지막 push 2026-10-13 | 얕은 복제 `repo/` 커밋 `657f444` (2026-10-13 "real-world+libero"), 정적 읽기만 |
| 프로젝트 페이지 | `hanyangyu1021.github.io/maskwam.github.io/` | 페이지 하단에 CC BY-SA 4.0 링크 | 200 | 저장소 `hanyangyu1021/maskwam.github.io`는 ★0, 마지막 push 2026-09-20, 라이선스 없음. "Model" 버튼은 `#`로 연결(링크 없음) |
| 실로봇 체크포인트 | `modelscope.ai/models/hanyangyu1021/MaskWAM-real-ckpt` | Apache-2.0 (LICENSE 파일) | `model.safetensors` 11,281,252,992 바이트(약 10.5 GiB) + `config.json`(1.4 KB), 다운로드 33회 | 설정 파일만 읽음, 가중치는 받지 않음 |
| LIBERO 체크포인트 | `modelscope.ai/models/hanyangyu1021/MaskWAM-libero-ckpt` | Apache-2.0 | 같은 크기(약 10.5 GiB), 다운로드 48회 | 설정 파일만 읽음 |
| 실로봇 데이터 | `modelscope.ai/datasets/hanyangyu1021/MaskWAM-realdata` | Apache-2.0 | 하위 폴더 bottles, bowls, mugs, pink_bowls와 각 recovery. 크기 미공개 | 받지 않음 |
| LIBERO 마스크 데이터 | `modelscope.ai/datasets/hanyangyu1021/MaskWAM-libero` | Apache-2.0 | 네 스위트 `*_robot`과 정규화 통계(452 B). 크기 미공개 | 받지 않음 |
| 기반 백본 | `Wan-Video/Wan2.2` (가중치 `Wan-AI/Wan2.2-TI2V-5B`) | Apache-2.0 | ★18.9k, 마지막 push 2026-09-21 | 논문의 Wan 2.2 |
| 분할 | `facebookresearch/sam3` | NOASSERTION (API) | ★12.4k, 마지막 push 2026-11-17 | 가중치와 소스는 이 저장소에 없고 사용자가 따로 설치 |
| 직접 기준선 | `github.com/yuantianyuan01/FastWAM` | MIT | 이 논문의 FastWAM 행(기준선) | 표 2의 값은 Fast-WAM 논문 표 3과 다르고 학습·평가 설정은 적혀 있지 않음(`03` K5) |
| 벤치마크 | `Lifelong-Robot-Learning/LIBERO` (MIT, ★2.6k), `RoboTwin-Platform/RoboTwin` (MIT, ★3.1k) | | 마지막 push 각 2025-03-15, 2026-09-24 | |
| 주석 도구 기반 | `Sage-PhysicalAI/RoboInter` (MIT, ★192) | | | 코드의 MaskWAM-Seg가 RoboInter-Tools 흐름을 가져옴 |

공식 저장소의 열린 이슈 세 건: #2 가중치 링크 문의(2026-06-25, 이후 체크포인트 공개), #3 RoboTwin 학습 설정 문의(2026-08-01: 6과제를 과제별, 6과제 통합, 50과제 통합 중 어느 것으로 학습하는지, 공개된 Fast-WAM 코드로 논문 수치를 맞추지 못함, 답변 없음), #4 그림 제작 도구 문의. 코드에는 RoboTwin 학습·평가 설정이 없습니다(`04` D9).

## 5. 학회 확인

CoRL 2026 포스터. 저장소의 `conferences/corl2026/papers.json`(corl.org 공식 채택 목록, 2026-11-14 수집)에 `in_official: true`, 포스터 14번, 11/11 10:30–12:00으로 있습니다. 프로젝트 페이지도 CoRL 2026을 표기합니다. 웹 검색은 쓰지 않았습니다.
