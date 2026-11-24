# 02. 선행연구와 관련 자료

> 원문 참고문헌 번호는 논문 v2의 [n]을 따릅니다. 외부 논문과 저장소는 이름과 공개 메타데이터만 확인했고 본문을 직접 읽지 않은 것은 "미확인"으로 적었습니다.

## 1. 한 줄 위치

MEM은 VLA의 기억을 **두 척도로 나눠** 다룹니다. 단기는 영상 인코더(ViT 안의 시공간 분리 어텐션)로, 장기는 자연어 요약으로 표현하고, 둘을 π0.6 위에서 함께 사전학습합니다(Sec. I, III). 저자들이 내세우는 차이는 (1) 한 표현으로 모든 척도를 감당하는 기존 방식(고유수용 기억, 점 궤적, 키프레임, 자연어 단독)과 달리 척도별로 표현을 나눈 것, (2) 모델이 기억을 갱신하는 시점과 압축 정도를 스스로 정하게 학습한 것, (3) 사전학습 규모에서 검증한 것입니다.

## 2. 구성 요소별 출처

| 구성 요소 | 논문에서 쓴 방식 | 선행 출처 (논문 [n]) |
|---|---|---|
| 공간-시간 분리 어텐션 | 공간 어텐션 + 4층마다 시간 어텐션, 패치 위치별 인과 마스크 | Bertasius 외 TimeSformer [3], ViViT [1], VideoMamba [25] (Sec. II, III-C) |
| 과거 토큰 삭제로 VLA 입력 길이 고정 | 상위 층에서 과거 시점 토큰 버림 | 논문 서술에 직접 출처 없음 |
| 시간 위치 임베딩 | 사인, e(0) = 0 | 표준 사인 임베딩 변형 (출처 없음) |
| 기억 길이 확장(사후학습) | 5초 사전학습 → 54초 사후학습 | 위치 보간 Chen 외 [10] |
| 고수준·저수준 분리, 하위 과제 지시 | π_HL이 하위 과제 l을 내고 π_LL이 행동 | π0.5 [36] (Physical Intelligence 외) |
| 언어 기억 | 요약 m_t를 모델이 갱신 | OneTwoVLA [26] (자연어 추론 표현), 요약 라벨은 기성 LLM |
| 기반 모델 | Gemma3-4B + SigLIP(400M), 액션 전문가 860M | π0.6 모델 카드 [35], Gemma 3 [21], KI [16] |
| 행동 | FAST 토큰 + 흐름 정합 | FAST [34], π0 [18] |
| 실시간 추론 | RTC, 학습 시점 RTC | Black 외 [6], [7] |
| 인간 교정 데이터 | 개입 후 시연 | π0.5 [36] |
| 풀링 기억 기준선 | 과거 프레임을 평균 풀링한 토큰 하나 | ContextVLA [20] (Jang 외) |
| 고유수용 기억 기준선 | 과거 관절 상태를 선형 사영 | [53] (고유수용 이력 기억) |
| 사후학습 전용 변형 | 사후학습에서만 인코더 도입 | CronusVLA [24] (v2에서 인용 추가) |
| 비교 대상으로 언급만 한 기억 계열 | 잠재 기억, 키프레임, 점 궤적 | SAM2Act [17], MemoryVLA [39], MemER [40], BPP [29], 점 추적 [9, 55] |


## 4. GitHub와 자료

**공식 코드: 없습니다.** 논문, arXiv 페이지, 프로젝트 페이지(Vercel 점검 화면 때문에 직접 열지 못하고 웹 조회도 한도로 실패해 내용은 미확인), 저자 소속 기관(Physical Intelligence)의 GitHub 조직의 저장소 10개 목록에서 MEM 구현을 찾지 못했습니다. GitHub 저장소 검색과 코드 검색에서도 공식 구현은 나오지 않았습니다(2026-11-24).

| 저장소 | 관계 | 라이선스 | 별 | 마지막 push | 확인 |
|---|---|---|---|---|---|
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 저자 소속 기관의 공개 π0/π0.5 코드. MEM 자체가 아니라 비교용 베이스(SigLIP So400m/14, Gemma 2B, 이미지 224×224) | Apache-2.0 | 14,747 | 2026-08-24 | 커밋 `215abfb`를 얕게 복제해 정적 읽기 (`04` §3) |
| [Physical-Intelligence/real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | RTC [6]의 공개 구현 | MIT | 637 | 2025-12-08 | 메타데이터만 |
| [shihao1895/MemoryVLA](https://github.com/shihao1895/MemoryVLA) | 관련 기억 VLA [39] 공식 | API 표시 없음 | 383 | 2026-06-13 | 메타데이터만 |
| [Sage-PhysicalAI/CronusVLA](https://github.com/Sage-PhysicalAI/CronusVLA) | 사후학습 전용 비교 변형의 모티브 [24] | MIT | 116 | 2026-01-11 | 메타데이터만 |
| [huiwon-jang/ContextVLA](https://github.com/huiwon-jang/ContextVLA) | 풀링 기억 기준선의 모티브 [20] | API 표시 없음 | 20 | 2025-11-05 | 메타데이터만 |
| [sam2act/sam2act](https://github.com/sam2act/sam2act) | 잠재 기억 계열 [17] | API 표시 없음 | 253 | 2026-05-29 | 메타데이터만 |
| [Fanqi-Lin/OneTwoVLA](https://github.com/Fanqi-Lin/OneTwoVLA) | 자연어 추론 표현 [26] | MIT | 243 | 2025-05-30 | 메타데이터만 |

- 프로젝트 페이지: `https://pi.website/research/memory` (논문 표지의 주소). 직접 내용을 읽지 못했습니다.
- 논문 PDF 공개 사본: 프로젝트 사이트의 `Mem.pdf` 링크가 제3자 목록에서 보이지만 읽지 않았습니다(arXiv 판만 사용).
- 가중치와 데이터: 공개 정보를 찾지 못했습니다.
- 이 표의 별 수와 push 일자는 2026-11-24 `gh api` 값입니다.
