# LAPA (arXiv:2410.11758, ICLR 2025) 분석 노트

*Latent Action Pretraining from Videos*를 표 재계산, 통계 검정, 공식 코드 정적 대조로 분석한 자료입니다. 분석 대상은 arXiv v2(2025-05-15)입니다. 재현 실험은 하지 않았습니다.

## 파일

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (논문 그림은 넣지 않음, 우리 그래프만) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이, 주장과 근거의 간격 |
| `analysis/02_prior_work.md` | 선행연구 계보, 공식 저장소 표 (GitHub 메타데이터 2026-11-21) |
| `analysis/03_analysis.md` | 재계산, 통계 검정, 일관성 검사 (V1–V11) |
| `analysis/04_code_verification.md` | 공식 코드와 논문의 대조 (D1–D12), 구현 가능성 판정 |
| `analysis/recompute.py`, `recompute_results.json` | 표 값 파싱, 재계산, 검정 (numpy, scipy, 수 초) |
| `analysis/plot_figures.py` | 그래프 세 장 (`report/figures/`) |
| `paper/paper.txt` | 논문 본문 pdftotext -layout 결과 (스크립트 입력) |

## 핵심 결과

1. Language Table에서 라벨 없는 사전학습은 기준선 대부분보다 유의하게 낫지만, Cross-task Seen에서 VPT와는 차이가 없습니다.
2. "+6.22%" 우위는 54회 짝지은 차이로 95% 구간 −5.8 ~ +18.4, 부호 검정 p = 0.33입니다.
3. 표 1의 ±는 과제 5개의 표준오차입니다. 평균 네 칸이 부록 표와 0.2–0.4포인트 다릅니다.
4. "30배 효율"은 H100 대 A100 속도 가정에 따라 26–40배입니다.
5. 공식 코드(MIT)는 핵심 방법과 맞습니다. 차이는 배치와 스텝, 텍스트 보조 손실 0.01, 그리퍼 이산화입니다.

## 재실행

```bash
python3 analysis/recompute.py      # 재계산 (paper/paper.txt 필요)
python3 analysis/plot_figures.py   # 그래프 (report/figures/에 저장)
```

## 라이선스

- 논문: CC BY 4.0 (arXiv 페이지 "view license"). 이 노트의 그래프는 우리가 그렸습니다.
- 분석 텍스트와 스크립트: 저장소 라이선스를 따릅니다.
