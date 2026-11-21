# 02. 선행연구와 저장소 (TaF-Foundation, arXiv:2611.20718)

## 1. 선행연구 계보 (구성 요소별 출처)

논문의 구성 요소를 출처별로 나눕니다. 번호는 논문 참고문헌 번호입니다(`paper/paper.txt`, 9–10쪽).

| 구성 요소 | 출처 | 이 논문이 더한 것 |
|---|---|---|
| 촉각 자기지도 사전학습과 센서별 인코더 + 공유 Transformer | Sparsh [10], T3 [35] | 같은 구조를 렌치 지도로 학습 |
| 교차 센서 정렬 (쌍 기반) | HTT [2], TactX [28] (정렬 대상: 같은 접촉 조건의 촉각 쌍) | 촉각끼리 쌍을 만들지 않고 렌치를 공통 기준으로 둠 |
| 촉각을 힘에 묶는 물리 지도 | AnyTouch 2 [7] (촉각–힘 쌍으로 3차원 힘 예측), UniForce [5] (준정적 평형 제약, 잠재 힘 맵) | 측정 렌치를 직접 지도로 쓰고, 데이터 1억 프레임 규모 |
| 같은 저자 전작 | TaF-VLA [15] (VQ-VAE 코드북 두 개, 대조 정렬, VLA에 통합) | 같은 프로브를 촉각과 힘으로 나누어 모은 방식의 대안으로 동시 기록 장치 |
| 촉각 언어 정렬 | CLTP [23], FG-CLTP [24] | 언어 대신 6차원 렌치 |
| 시각 백본 초기화 | DINOv2 [27] | 인코더 Transformer 초기값 |
| 정책 백본 | ACT [36] | 촉각 토큰을 512차원으로 추가 |
| 시뮬레이션 기준선 | UniVTAC [4] | 시뮬 과제 3개의 공식 정책을 기준선으로 씀 |
| 데이터셋 비교 | Sparsh Force (Table 1, [10]), RCT [8], ToucHD-Force [7] | 센서 24개, 프로브–센서 쌍 1,445개 |
| 기준 렌치 장치 | ATI Axia80-ZC22 (상업 6축 센서, 부록 A.2) | 없음 |

**가장 가까운 대안:** AnyTouch 2(시각 촉각 광학 센서 계열, 힘 지도)와 UniForce(잠재 힘 맵, 힘 라벨 없이 사전학습). 논문 Table 2의 최강 기준선은 단일프레임 힘·토크, 검색 R@1·R@5에서 AnyTouch 2, 시간 힘·토크에서 HTT입니다(03 §1.1).

## 2. 코드와 저장소 (GitHub 표)

조회: 2026-11-21, GitHub API (메타데이터만, 클론·설치·실행 없음).

| 저장소 | 무엇인가 | 공식 여부 | 라이선스 | 스타 | 마지막 push | 커밋 |
|---|---|---|---|---|---|---|
| [mrJiangjm/TaF-Foundation](https://github.com/mrJiangjm/TaF-Foundation) | 프로젝트 페이지 저장소 (GitHub Pages). `index.html`과 `assets/`(그림 3장, 영상 4개)만 있음. **코드와 데이터 없음** | 프로젝트 페이지와 연결되어 있으나 저자 계정 여부는 미확인 | 없음 (API null) | 0 | 2026-11-20 | `1b08adee3` (커밋 7개, 생성 2026-11-16) |
| [ModelScope](https://modelscope.cn) 링크 | 프로젝트 페이지 버튼. 실제 주소 없이 `#top`을 가리킴 | — | — | — | — | — |
| [univtac/UniVTAC](https://github.com/univtac/UniVTAC) | 시뮬 기준선 UniVTAC의 저장소 (논문 ref [4]) | 이름 일치, 저자 대조 안 함 | Apache-2.0 | 196 | 2026-09-07 | `0dafa1026` |
| [GeWu-Lab/AnyTouch2](https://github.com/GeWu-Lab/AnyTouch2) | 기준선 AnyTouch 2 (ref [7]) | 추정 공식 (연구실 명칭 일치) | MIT | 68 | 2026-04-17 | `82c5677d9` |
| [jxbi1010/HTT](https://github.com/jxbi1010/HTT) | 기준선 HTT (ref [2]) | 추정 공식 (저자 이름 일치) | NOASSERTION (확인 불가) | 8 | 2026-09-05 | `2e5b0b10d` |
| [alanzjl/t3](https://github.com/alanzjl/t3) | 기준선 T3 (ref [35]) | 추정 공식 (제1저자 이름 일치) | MIT | 64 | 2024-06-21 | `341177f23` |
| [facebookresearch/sparsh](https://github.com/facebookresearch/sparsh) | 기준선 Sparsh (ref [10]) | 공식 | NOASSERTION, **보관(archived)** | 225 | 2025-02-27 | `fee6a05a9` |
| [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | DINOv2 기준선과 인코더 초기값 (ref [27]) | 공식 | Apache-2.0 | 13560 | 2026-11-05 | — |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | ACT 정책 백본 (ref [36]) | 공식 | MIT | 2352 | 2024-07-23 | — |

- **공식 코드는 없습니다.** TaF-Foundation 본체 저장소를 찾지 못했고(`gh search repos`에서 제목 검색 결과는 프로젝트 페이지 저장소 하나뿐), 프로젝트 페이지도 데이터셋을 "acceptance 이후 공개"라고 적었습니다.
- TactX [28]와 UniForce [5]는 GitHub 검색에서 찾지 못했습니다(0건).
- 따라서 `repo/` 클론은 하지 않았습니다. 04는 코드 대조 대신 주장 대조입니다.
- 기준선 저장소의 체크포인트가 TaF 데이터로 재학습된 것인지는 저장소로 판단할 수 없습니다(V1).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- **DINOv2 초기화와 DINOv2 기준선이 같은 백본입니다.** 따라서 "렌치 지도로 사전학습" 효과는 DINOv2 대비 개선으로 읽을 수 있습니다. 단일프레임 힘 3.032 → 2.321 N입니다(표 2).
- **UniVTAC 시뮬은 외부 벤치마크입니다.** UniVTAC의 시각 촉각 입력 형식 때문에 Board Clean에서는 UniVTAC를 평가하지 않았습니다(표 5 주석).
- **정책 백본은 기존 방법(ACT)을 그대로 씁니다.** 정책 학습률 1e-5와 배치 64는 표 S3에 있습니다. 어떤 ACT 설정을 기준으로 바꾸었는지는 본문에 없습니다.
