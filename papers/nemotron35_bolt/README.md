# Nemotron 3.5 Bolt 74B-A9B 24GB (arXiv:2610.33039) 분석 노트

*Nemotron 3.5 Bolt 74B-A9B 24GB: A Hybrid Vision-Language Model for Local AI* (NVIDIA, 2026-10-15)를
공개 config·GGUF 헤더 대조, 파라미터·메모리 산술, 디코드 대역폭과 초안 검증 비용 toy, 표 재계산으로 분석한 자료입니다.
학습 코드는 공개되지 않아 학습 재현은 하지 않았고, 가중치는 받지 않았습니다(GGUF는 HTTP Range로 헤더만).

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 계보, 구성 요소별 출처, Hugging Face·GitHub 정리 |
| `analysis/03_analysis.md` | 파라미터·메모리 산술, 토큰당 읽는 바이트, 초안 블록 검증 비용, 2비트의 비용 |
| `analysis/04_code_verification.md` | 논문 ↔ 공개 config·GGUF·llama.cpp 대조(D1–D10), 수치 재계산(C1–C11), 구현 가능성 |
| `analysis/gguf_header_check.py` | GGUF 헤더만 받아 텐서별 형식·바이트 집계 |
| `analysis/config_check.py` | config로 파라미터, KV·SSM 상태, 24 GiB 맥 메모리 여유 계산 |
| `analysis/spec_decode_toy.py` | 디코드 대역폭과 MoE 초안 검증 비용 모델 |
| `analysis/table_check.py` | 표·본문 수치 재계산 |
| `analysis/plot_figures.py` | 리포트 그래프 세 장 |

## 재실행

```bash
pip install numpy matplotlib
mkdir -p paper/hf
curl -L -o paper/hf/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16_config.json \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-BF16/resolve/e09a4cedb1787db1760375ee5539f222cae0c672/config.json
python3 analysis/gguf_header_check.py   # GGUF 다섯 개의 앞부분(합계 약 92 MB)을 .scratch/에 받음
python3 analysis/config_check.py && python3 analysis/spec_decode_toy.py
python3 analysis/table_check.py && python3 analysis/plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다(헤더 다운로드 시간 제외). 결과 JSON이 함께 들어 있으므로 다운로드 없이 `table_check.py`와 `plot_figures.py`만 다시 돌려도 됩니다.

## 출처와 라이선스

- 논문은 arXiv abs 페이지 기준 **CC BY 4.0**이지만, PDF에 "© 2026 NVIDIA. All rights reserved." 문구가 함께 있습니다. 권리 표기가 충돌하므로 **논문 그림은 이 저장소에 넣지 않았습니다.** 그림은 [연구 페이지](https://research.nvidia.com/labs/nemotron/Nemotron-3.5-Bolt-24GB/)와 논문에서 볼 수 있습니다.
- 세 그래프는 논문 표 수치와 공개 GGUF 헤더 집계로 직접 그렸습니다.
- 대조에 쓴 공개 산출물: Hugging Face `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-*`(OpenMDW-1.1; BF16 `e09a4ce`, 24GB `cf2effc`, GGUF `e0fecb6`, LiLiCorr `449d241`·`182d298`), `unsloth/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF`(`34841e3`), `ggml-org/llama.cpp`(`889f2ec`). 모델 파일과 config는 이 저장소에 넣지 않았습니다.
- 원문 PDF는 이 저장소에 넣지 않았습니다. 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
