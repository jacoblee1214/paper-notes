# 02. 선행 연구와 관련 자료: DeTac

## 1. 계보

| 흐름 | 대표 | DeTac와의 관계 |
|---|---|---|
| 시각 촉각 센서(음영) | GelSight [9], GelSlim [10, 12, 14], DIGIT [30], 색 부호화 조명 센서 | 음영 이미지가 형상 복원의 입력. DeTac의 출력 하나가 이것 |
| 마커 기반 힘·전단 | Yuan 외 2015 [11], GelSight Wedge [13], TacTip 계열 [25, 35–37] | 마커 변위가 힘 추정의 입력. 출력의 다른 하나 |
| 하드웨어로 분리 | UVTac [17](UV 마커 전환) | 한 번에 한 모달리티만 얻고 하드웨어 부담이 큼(2.1절) |
| 직렬 인페인팅 | TacDiff [21](Ou, Chen, Luo, RA-L 2024), MVTS [22] | 마커 마스크 추정 후 인페인팅. DeTac가 직접 겨냥한 기준선 |
| 일반 인페인팅 | TELEA [23], LaMa [54] | 마스크 기반 기준선 |
| 조건부 영상 생성 | Palette [45], InstructPix2Pix [46], ControlNet [42] 등 | 전체 영상 생성(마스크 없음)을 택한 근거(2.2절) |
| 마커·음영 공존 센서 | MagicSkin [15], MuxGel [16] | 하드웨어 쪽 접근. 서론에서 결합 문제의 근거로 인용 |

## 2. 구성 요소별 출처

| DeTac의 요소 | 출처 | 메모 |
|---|---|---|
| 음영 분기 | DDPM [18] 노이즈 예측, 픽셀 공간 U-Net [50], 입력 연결 조건(Palette 방식) | 새 구조가 아니라 조건 입력이 I_mix |
| 마커 분기 | FCN [51] 3층 | 가벼운 분할망 |
| L_sur | 마커 예측으로 만든 고리 가중 L1 | 마스크 가중 손실의 변형. 마커 예측이 고리를 정함 |
| L_con | 복원 전후 차이를 마커 가시도로 사용 | 두 분기를 잇는 결합 항 |
| 데이터 | 정렬 핀 외피, 질감 유닛 패드, 마커 제작 전후 쌍 수집 | 이 논문의 가장 독자적인 부분(3.3절) |
| 정답 마커 | SAM [53] + 사람이 찍은 중심점 | pseudo GT |
| 센서 | MC-Tac [52] | 공개 하드웨어(아래 표) |

## 4. GitHub와 자료 (확인 2026-11-23)

| 자료 | 링크 | 라이선스 | 별 / 마지막 활동 | 비고 |
|---|---|---|---|---|
| **DeTac 공식 코드** | 찾지 못함 | - | - | arXiv 페이지, 본문, `gh search repos`(DeTac, Dual-branch Decoupling 등), 웹 검색에서 확인되지 않음 |
| 프로젝트 페이지 | 찾지 못함 | - | - | arXiv 링크 섹션에도 없음 |
| MC-Tac (센서 하드웨어) | https://github.com/Tacxels/MC-Tac | BSD-3-Clause | 별 22 / 2023-09-21 | 논문 그림 3의 센서. hardware, imgs 폴더(하드웨어 자료) |
| TacDiff 공식 구현(Marker or Markerless?) | https://github.com/gitouni/Marker-or-Markerless | AGPL-3.0 | 별 5 / 2025-08-25 | 가장 가까운 기준선 [21]. 이 저장소 소유자가 논문 저자인지 확인하지 않음 |
| LaMa | https://github.com/advimman/lama | Apache-2.0 | 별 10,410 / 2025-02-05 | 기준선 [54] |
| SAM 3 | https://github.com/facebookresearch/sam3 | NOASSERTION(별도 라이선스) | 별 12,381 / 2026-11-17 | 표 2의 "SAM 3.1" 기준선 [55]. 3.1 버전 여부는 확인하지 않음 |
| SAM (원본) | https://github.com/facebookresearch/segment-anything | Apache-2.0 | 별 55,174 / 2024-09-18 | pseudo GT 생성에 쓴 SAM [53]. 어느 버전을 썼는지 논문에 없음 |
| GelSight SDK | https://github.com/joehjhuang/gs_sdk | GPL-3.0 | 별 39 / 2025-06-02 | 광도 스테레오 [59] |
| UVTac | 공식 저장소를 찾지 못함 | - | - | 기준선 [17] |

표의 수치는 `gh api`로 확인했고 clone은 하지 않았습니다. 공식 코드가 없어서 `analysis/04_claim_verification.md`로 대조합니다.
