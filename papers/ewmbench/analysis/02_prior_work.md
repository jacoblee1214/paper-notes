# 02. 선행연구와 계보

## 1. 계보

- **영상 생성의 기반:** DDPM(Ho 2020), DDIM(Song 2020), Stable Video Diffusion(Blattmann 2023), VideoCrafter1(Chen 2023), AnimateDiff(Guo 2023), I2VGen-XL(Zhang 2023), DynamiCrafter(Xing 2025), Open-Sora(Zheng 2024), Open-Sora 2.0(Peng 2025), HunyuanVideo(Kong 2024), Vidu(Bao 2024). 논문 §2.1이 나열합니다.
- **로봇 쪽 영상 예측·세계모델:** Video generation models as world simulators(Brooks 2024), UniSim(Yang 2023), RoboDreamer(Zhou 2024), GR-2(Cheang 2024), EVA(Chi 2024), EnerVerse(Huang 2025). 논문 §2.1 인용 범위입니다.
- **영상 평가 벤치마크(일반 목적):** FID(Heusel 2017), IS(Salimans 2016), FVD(Unterthiner 2019), VBench(Huang 2024), VBench++(Huang 2024b), EvalCrafter(Liu 2024), T2V-CompBench(Sun 2024), TC-Bench(Feng 2024), T2VBench(Ji 2024), VMBench(Ling 2025), PhyGenBench(Meng 2024), Physics-IQ(Motamed 2025). 논문 표 1이 이 중 열 개를 여덟 차원으로 비교합니다.
- **EWMBench의 위치:** 표 1에서 EWMBench만 궤적 동역학(TrajD), 궤적 타당성(TrajP), 장면 일관성(SceneC), 다양성(DivM), 시간 인과(TempC), 상호작용 논리(InterL), 화질(ImgC), 의미 유사도(SemS)를 모두 체크합니다. 이 체크표는 저자가 만든 비교이므로, 각 벤치마크 원문으로 검증하지 않았습니다.

## 2. 구성 요소별 출처

| 구성 요소 | 출처 (논문 인용) | 이 논문에서의 역할 |
|---|---|---|
| 장면 특징 | DINOv2 (Oquab 2023) → 미세조정 | SceneC. 가중치 HF 공개(`dinov2-EWMBench-v0.1.pth`), 학습 코드는 없음 |
| 기준 특징 | DINO ViT-B/16 (Zhang 2022, VBench가 씀) | 비교 대상(부록 A.3.1, 그림 8–9) |
| 궤적 검출 | YOLO-World (Cheng 2024) → 미세조정, BoT-SORT (Aharon 2022) | 말단 장치 검출과 추적 (부록 A.1) |
| 궤적 거리 | Hausdorff (Serra 1998), DTW (Müller 2007), Wasserstein (Villani 2009) | HSD, NDTW, DYN |
| 그리디 다양성 선택 | Vince 2002 (greedy framework) | 과제별 궤적 10개 선택 (부록 A.2.2) |
| 캡션 MLLM | Qwen2.5-VL (코드 기준). 논문 본문은 "video MLLM"으로만 씀 | General, Events, Overall_Constraints |
| 의미 유사도 | CLIP (ViT-B/32 다양성, ViT-B/16 CLIPScore), BLEU | Diversity, CLIP, BLEU |
| 기준 모델 | OpenSora 2.0, LTX, COSMOS-7B, Kling-1.6, Hailuo I2V-01-live, LTX_FT, EnerVerse | 비교 7개 (§4) |

## 4. GitHub와 자료 (확인 2026-11-22)

| 자료 | 링크 | 라이선스 | 별 / 포크 / 이슈 | 활동 | 비고 |
|---|---|---|---|---|---|
| **공식 코드 (EWMBench)** | https://github.com/AgibotTech/EWMBench | README는 CC BY-NC-SA 4.0. 저장소에 LICENSE 파일 없음(GitHub API license null) | 132 / 10 / 2 | 생성 2025-05-14, 마지막 push 2025-06-13 (`3a5531c`, "Update __init__.py") | `repo/`에 `--depth 1`로 얇게 clone, **정적 읽기만**. 크기 약 1.9 MB (clone 기준). 커밋 3개 확인 |
| 데이터셋 (HF) | https://huggingface.co/datasets/agibot-world/EWMBench | cc-by-nc-sa-4.0 | 다운로드 318 (HF 표시) | 마지막 수정 2025-05-16 | `gt_dataset.tar`, `generated_samples.tar`. 전체 크기 약 270 MB (HF mainSize 메타데이터). **받지 않음** |
| 가중치 (HF) | https://huggingface.co/agibot-world/EWMBench-model | cc-by-nc-sa-4.0 | - | 마지막 수정 2025-05-16 | `dinov2-EWMBench-v0.1.pth`, `yoloworld-EWMBench-v0.1.pt`. 크기 미확인. **받지 않음** |
| Qwen2.5-VL-7B-Instruct | huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct (README 링크) | 미확인 | - | - | 캡션 MLLM. 크기는 이 모델 카드를 확인하지 않아 미확인(7B bf16이면 15 GB 안팎으로 추정). **받지 않음** |
| CLIP 가중치 | ViT-B/16 (HF openai/clip-vit-base-patch16), ViT-B/32 (openaipublic CDN) | MIT (CLIP 저장소, 미확인) | - | - | README 링크. **받지 않음** |
| 프로젝트 페이지 | 논문에 URL 명시 안 됨. arXiv 댓글 "Website: this https URL"의 주소는 확인하지 못함 | - | - | - | 프롬프트 전문 위치라고 논문 §3.4가 가리킴. 찾지 못함 |
| VBench | github.com/Vchitect/VBench | 미확인 | 미확인 | 미확인 | 사람 순위 비교 대상(그림 6). 점수 재계산 안 함 |
| 학회 공식 목록 | bmvc2025.bmva.org/proceedings/736/ | - | - | - | BMVC 2025 proceedings 목록에서 확인 (발표 형식 미표기) |

- 공식 코드의 README는 CUDA 11.8과 Python 3.10을 요구합니다. requirements.txt는 torch 2.4.0, mmcv 2.2.0, mmdet 3.3.0, xformers 0.0.27.post2를 고정합니다.
- 저장소의 `submodel/dinov2`는 git submodule이라 얇은 clone에는 없습니다. `.gitmodules`는 upstream facebookresearch/dinov2를 가리킵니다.
- 저장소 README의 인용 항목은 `journal={arXiv preprint arXiv:2505.09694}`이고 BMVC 2025는 적혀 있지 않습니다.
