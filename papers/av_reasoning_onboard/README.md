# Running AV Reasoning Models Within Onboard Compute and Latency Limits 분석 노트

NVIDIA 기술 블로그(2026-11-10, 접근 2026-11-12)의 Alpamayo 2 Edge와 AlpaJudge 주장을 공개 config, 공개 코드, 레시피 단계 시간과 대조하고 수치를 다시 계산한 자료입니다.
블로그는 동료심사를 받지 않았습니다. 가중치는 받지 않았고 공개 코드는 실행하지 않았습니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 |
| `analysis/01_paper_analysis.md` | 블로그 주장, 방법, 수치, 빈칸 |
| `analysis/02_prior_work.md` | 계보, GitHub·HF 목록과 라이선스 |
| `analysis/03_analysis.md` | 파라미터·메모리·지연·대역폭·추측 디코딩 재계산 |
| `analysis/04_claim_verification.md` | 주장 대조, 정적 코드 대조, 구현 가능성 판정 |
| `analysis/config_check.py` | 공개 config로 파라미터와 메모리 산술 |
| `analysis/latency_toy.py` | 대역폭 상한, 지연 모델, 추측 디코딩 몬테카를로 |
| `analysis/rf_score_check.py` | AlpaJudge RF 점수 재계산 (공개 노트북의 기록값 사용) |
| `analysis/plot_figures.py` | 그래프 세 장 |

## 재실행

스크립트 입력(공개 config 두 개와 노트북 하나)은 제3자 파일이라 이 저장소에 넣지 않았습니다. 먼저 받아 두세요.

```bash
mkdir -p paper/hf paper/repos/alpajudge
curl -L -o paper/hf/nvidia_Alpamayo2-Edge_config.json https://huggingface.co/nvidia/Alpamayo2-Edge/resolve/main/config.json
curl -L -o paper/hf/nvidia_alpajudge_config.json https://huggingface.co/nvidia/alpajudge/resolve/main/config.json
curl -L -o paper/repos/alpajudge/alpajudge_pai.ipynb https://raw.githubusercontent.com/NVlabs/alpajudge/607ae9d/examples/alpajudge_pai.ipynb
```

```bash
pip install numpy matplotlib
python3 analysis/config_check.py
python3 analysis/latency_toy.py
python3 analysis/rf_score_check.py
python3 analysis/plot_figures.py
```

모두 CPU에서 몇 초 안에 끝납니다.

## 출처와 라이선스

- 블로그는 NVIDIA 저작물로 보고 그림을 싣지 않았습니다. 그래프는 이 저장소 스크립트가 만든 것입니다.
- 공개 config는 Hugging Face `nvidia/Alpamayo2-Edge`, `nvidia/alpajudge`(OpenMDW-1.1)에서, 노트북은 NVlabs/alpajudge(Apache-2.0, 커밋 607ae9d)에서 받습니다. 이 저장소에는 넣지 않았습니다(분석 시점의 Alpamayo2-Edge config는 main 리비전이 바뀌었을 수 있습니다).
- 원문 HTML과 모델 가중치는 이 저장소에 넣지 않았습니다.
