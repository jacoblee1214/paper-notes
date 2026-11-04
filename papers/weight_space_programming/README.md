# Sparks of Weight-Space Programming 분석 노트

*Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*
(Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 블로그 글 2026-11-02)를 저자들이 공개한 실험 로그로 수치를 다시 계산하고,
실험 1·2의 가중치를 numpy로 직접 읽어 보며 분석한 자료입니다. LLM 실험 자체는 다시 돌리지 않았습니다.
원문: https://rikiyat.github.io/blog/weight-space-programming/

| 경로 | 내용 |
|---|---|
| `report/report_public.pdf` | 통합 리포트 (게시용, 블로그 그림 제외) |
| `analysis/01_paper_analysis.md` | Abstract → Method → Experiments → Limitations |
| `analysis/02_prior_work.md` | 선행연구 계보, 실험별 기존 기법, 관련 GitHub 표 |
| `analysis/03_analysis.md` | 실험 1·2 가중치 접기, 실험 3 확률, 실험 4 편집 크기, 실험 5 씨앗 편차, 실험 6 예산 곡선, 실험 8 짝 비교와 실행 선택 |
| `analysis/04_claim_verification.md` | 주장 검증 C1–C14, 재현 정보 체크리스트, 구현 가능성 |
| `analysis/verify_logs.py` | 공개 로그로 블로그 수치 재계산 |
| `analysis/weight_reading_toy.py` | 실험 1·2 가중치를 numpy로 접어 보기(forward pass 없음) |
| `analysis/plot_figures.py` | 리포트 그래프 |

## 재실행

```bash
pip install numpy scipy matplotlib
git clone https://github.com/RikiyaT/lm_understand-logs logs_official   # 커밋 4d09ba6 기준, 약 370 MB
python3 analysis/verify_logs.py logs_official
python3 analysis/weight_reading_toy.py logs_official
python3 analysis/plot_figures.py
```

모두 CPU에서 1분 안에 끝납니다(실험 8 부트스트랩 포함).

## 출처와 라이선스

- 블로그 페이지에는 라이선스 표기가 없습니다. 그래서 이 저장소에는 **블로그 그림과 본문 사본을 넣지 않았습니다.**
- `report/figures/*.png`는 저자들의 공개 로그(RikiyaT/lm_understand-logs, 로그·데이터·신경망 CC BY 4.0)로 직접 계산해 그린 그래프입니다. 원 데이터 출처: Takehi, Teoh, Zhang, Khattab (2026).
- 분석 문서와 스크립트는 이 저장소 작성자가 만들었습니다.
