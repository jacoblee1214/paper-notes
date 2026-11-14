# Nemotron 3.5 Super VL·Lightning 기술 보고서 (NVIDIA, 2026-10-27) 분석 노트

*Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI*
(NVIDIA 기술 보고서, arXiv 없음)를 공개 체크포인트의 `config.json`·safetensors 헤더 대조, 표·블로그 수치 재계산,
ReplaySSM·MOPD numpy toy로 분석한 자료입니다. 모델 가중치는 받지 않았고 모델을 실행하지 않았습니다.

- 보고서 PDF: https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3.5-Super-and-Lightning-Technical-Report.pdf
- 모델: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 (확인 리비전 `fb0ee0b`),
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 (`da5b265`)
- 평가 레시피: https://github.com/NVIDIA-NeMo/Gym `nemotron_recipes/super-3.5/` (레시피가 고정한 평가 커밋 `1e66890`)

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 보고서 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행 보고서 계보, 구성 요소별 출처, 공개 자료(GitHub·Hugging Face) 표 |
| `analysis/03_analysis.md` | 활성 파라미터 해부, 요청당 상태 메모리, ReplaySSM·MOPD 확인, 벤치마크 재해석 |
| `analysis/04_code_verification.md` | 보고서 ↔ 공개물 대조(D1–D12), 수치 재계산(C1–C11, V1–V3), 구현 가능성 |
| `analysis/fetch_hf_meta.py` | Hugging Face config·색인·safetensors 헤더 재수집 (가중치는 받지 않음) |
| `analysis/param_check.py` | 구조·파라미터·메모리 계산. 원자료가 없으면 `hf_meta_summary.json`(헤더 요약)을 씀 |
| `analysis/table_check.py` | 보고서 표·본문·블로그 수치 재계산 |
| `analysis/mechanism_toy.py` | ReplaySSM 닫힌 식·트래픽·FP16 반올림 오차, MOPD proximal teacher 하한 (numpy, 약 10초) |
| `analysis/plot_figures.py` | 리포트 그래프 네 장 |
| `analysis/extra_mamba_decay.py`, `analysis/extra_mamba_decay_results.json` | 공개 `A_log`·`dt_bias`(Mamba 40개 층, 헤더와 텐서 80개 range 읽기, 3.9 MB, 원자료는 공개 저장소에 넣지 않음)로 헤드별 감쇠 분포를 만들고 FP16 RN·SR·replay toy를 다시 돌림 (`03` 추가 검증, 약 8분) |
| `analysis/extra_nvfp4_scale.py`, `analysis/extra_nvfp4_scale_results.json` | shard 하나의 expert 가중치(132 MB range 읽기)를 NVFP4로 에뮬레이션해 Max와 MSE 블록 scale을 비교 (`03` 추가 검증). 원자료는 저장하지 않음 |
| `analysis/extra_nvfp4_scale_layers.py`, `analysis/extra_nvfp4_scale_layers_results.json` | 층 6개(3, 30, 48, 65, 79, 87)에서 expert 8개씩 range로 받아(530 MB) 층 12와 합쳐 120개 텐서에서 Max와 MSE scale 비교 (`03` 추가 검증 2026-11-14). 원자료는 저장하지 않음 |

## 재실행

```bash
pip install numpy matplotlib
python3 analysis/param_check.py && python3 analysis/table_check.py && python3 analysis/mechanism_toy.py
cd analysis && python3 plot_figures.py
python3 -I analysis/extra_mamba_decay.py   # Mamba 감쇠 실제 분포 (40 샤드 헤더 + 텐서 80개, 3.9 MB, 약 8분)
python3 analysis/extra_nvfp4_scale.py   # shard 9의 expert 텐서 24개를 HTTP Range로 받음 (약 130 s, 132 MB)
```

## 출처와 라이선스

- 보고서는 "© 2026 NVIDIA. All rights reserved."이고 CC 라이선스가 없습니다. 그래서 이 저장소에는 **보고서 그림과 원문 PDF를 넣지 않았습니다.**
- 모델 가중치는 OpenMDW-1.1, NVIDIA-NeMo 코드는 Apache-2.0입니다.
- 그래프는 보고서 표의 수치, 공개 config, safetensors 헤더로 직접 그렸습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
