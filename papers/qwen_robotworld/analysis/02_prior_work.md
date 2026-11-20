# Qwen-RobotWorld 선행연구와 관련 자료

## 1. 계보: 어디에서 무엇을 가져왔나

| 구성 요소 | 출처 (논문 인용) | 이 논문의 변형 |
|---|---|---|
| MMDiT 이중 스트림 블록 | Esser 외 2024 (SD3, ICML) | 이해 스트림에 동결 Qwen2.5-VL 은닉 상태, 생성 스트림에 VAE 잠재 (§3.1) |
| 행동 인코더로 MLLM | 논문 자체 주장 (T5 Raffel 2020, CLIP Radford 2021 대비) | 마지막 층 은닉 상태를 조건으로 사용 (§3.1) |
| VAE | Wan 2025 (Wan-VAE) | 같은 구조 |
| 위치 인코딩 | RoFormer (Su 2024), 시각 RoPE (Heo 2024), Scalable RoPE (Wan 2025) | 시간 16 / 높이 56 / 너비 56 비대칭 분할 (§3.2) |
| 흐름 정합 목적 | Lipman 2023, Liu 2022 | 로그정규 timestep, 길이 적응 이동 |
| 다중 구간 조건 | 가장 가까운 선행: 첫 프레임 조건 (TI2V) | Scene2Robot: 장면, 로봇 참조, 생성 3구간, 인덱스 기반 timestep 0 (§3.3) |
| 사람→로봇 쌍 데이터 | MANO (Romero 2017), MuJoCo IK | 사람 손 인페인팅 + 로봇 렌더 (§2.2.5) |
| 장면 데이터 | VLNVerse (Lin 2025), Isaac Sim (NVIDIA 2022), InternData-A1 (Tian 2025) | 내비게이션과 시뮬 사진실감 쌍 |
| 평가 | EWMBench (Yue 2025), DreamGen Bench (Zhou 2025), PBench (NVIDIA 2025a), WorldModelBench (Li 2025), VBench (Huang 2024) | 자체 보고 점수 |

- **다른 기여의 성격.** 아키텍처 부품은 대부분 기존 기법입니다. 새로운 것은 EWK 데이터 구성과 캡션 파이프라인, 그리고 다중 구간 조건의 적용입니다. 이 둘은 ablation이 없어 효과가 분리되지 않습니다.

## 2. 가까운 대안 (논문 표의 기준선)

| 모델 | 종류 | 논문에서의 위치 | 비고 |
|---|---|---|---|
| Cosmos (Agarwal 2025), Cosmos2.5-14B | 오픈 세계모델 (NVIDIA) | 임베디드 기준선 (표 2–5), RoboTwin-IF 비교 | 표 4에서 Overall 0.802로 Ours 0.804와 차이 0.002 |
| LVP (Chen 2025a) | 오픈 임베디드 영상 계획 | 표 2 EWMBench 2위(4.05), 표 3 DreamGen 2위 | 표 4 Motion 0.991로 자체 모델보다 높음 |
| GigaWorld (Team 2025) | 오픈 | 표 3 GR1-Env IF 0.933로 자체보다 높음 | 본문에 언급 없음 |
| Vidar (Feng 2025), WoW (Chi 2025) | 오픈 | 임베디드 기준선 | 표 5에서 Vidar 7.01 |
| Veo3, Sora2, Wan2.6, Kling26, LTX-2 | 상용 또는 일반 영상 | 일반 기준선 | 자체 모델보다 Veo3, Wan2.6이 WorldModelBench 상위 |

## 4. 코드와 프로젝트 페이지 (조회 결과)

이번 분석에서 직접 조회한 항목입니다. 조회일은 2026-11-20입니다.

| 항목 | 위치 | 상태 | 라이선스 | 최근 push | 비고 |
|---|---|---|---|---|---|
| 논문 공식 저장소 | github.com/QwenLM/Qwen-RobotWorld | 404 (없음) | - | - | 조직 저장소 62개 목록에도 없음 |
| 형제 공식 저장소 | github.com/QwenLM/Qwen-RobotManip | 공개, 189 stars | 없음 | 2026-06-29 | 가중치 공개 계획 없다고 명시 |
| 형제 공식 저장소 | github.com/QwenLM/Qwen-RobotNav | 공개, 202 stars | 없음 | 2026-06-30 | 가중치 공개 계획 없다고 명시 |
| 프로젝트 페이지 | qwen.ai/blog?id=qwen-robotworld | 자동 조회로 본문을 읽지 못함 (미확인) | - | - | 사람이 직접 열어 확인 필요 |
| Hugging Face 모델 | huggingface.co, Qwen 조직 robot 검색 | 0건 | - | - | - |

- 논문에 코드·가중치·데이터 링크가 없습니다(arXiv 페이지, PDF 본문과 참고문헌 확인).
- 따라서 공식 코드 대조 문서(04_code_verification)는 만들 수 없고, 주장 검증 문서(04_claim_verification)로 대체합니다.
- 형제 저장소는 같은 Qwen 팀 이름으로 2026-06-29 전후에 공개되었습니다. 세 보고서 모두 공개 가중치를 약속하지 않습니다.
