# UVA (arXiv:2503.00200, RSS 2025) 분석 노트

*Unified Video Action Model* (arXiv v3, 2025-04-24)을 표 재계산, 통계 검정, CPU 토이로 분석한 자료입니다. 논문 원문 PDF는 넣지 않았습니다(arXiv:2503.00200에서 받을 수 있습니다). 논문 그림은 한 장만 넣었고 출처를 적었습니다.

## 구현 가능성 (요약)

- **원본 학습은 이 맥에서 불가능합니다.** 공식 설정이 여러 GPU를 가정하고, 학습 예산이 논문에 없습니다.
- **PushT 체크포인트 추론 평가는 조건부로 가능합니다.** 체크포인트 다운로드, 설치, 장치 코드 수정이 필요합니다.
- **실로봇 재현은 불가능합니다.** 로봇 하드웨어가 필요합니다.

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 논문 그림 한 장 포함) |
| `analysis/01_paper_analysis.md` | 논문 구조, 방법, 실험 수치, 버전 차이 |
| `analysis/02_prior_work.md` | 선행 계보, 공식 저장소 표 |
| `analysis/03_analysis.md` | 재계산, 일관성 검사, 토이 결과 |
| `analysis/04_code_verification.md` | 공식 코드 대조, 라이선스, 구현 가능성 |
| `analysis/recompute.py` | 표 재계산과 검정 (표준 라이브러리) |
| `analysis/toy_best_of_n.py` | 토이: 후보 N개 중 전 모델 예측으로 고르기 (표준 라이브러리, 수 초) |
| `report/figures/` | 그래프 두 장(우리가 그림)과 논문 그림 한 장(© 저자, CC BY 4.0) |

## 재실행

```bash
python3 analysis/recompute.py
python3 analysis/toy_best_of_n.py
```

## 출처와 라이선스

- 논문은 arXiv에 CC BY 4.0으로 공개되어 있습니다(arXiv 라이선스 페이지 확인). 원문 PDF 저작권 표시는 찾지 못했습니다.
- 논문 그림은 저작자 표기와 함께 실었습니다("© 저자, CC BY 4.0, 출처 arXiv:2503.00200"). 실로봇 사진과 데이터셋 프레임이 든 그림은 넣지 않았습니다.
- 공식 코드는 `github.com/ShuangLI59/unified_video_action`에 있으며 MIT 라이선스입니다. 이 저장소에는 코드를 넣지 않았고, 실행하거나 설치하지 않았습니다.
- 가중치와 데이터셋은 받지 않았습니다.
