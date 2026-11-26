# 02. 선행연구와 관련 저장소

## 1. 선행연구 계보 (구성 요소별 출처)

| 구성 요소 | 논문이 기대는 출처 | 이 논문에서의 위치 |
|---|---|---|
| 이벤트 방식 촉각 센서 | Evetac [12] (Funk 외, IEEE T-RO 2024, 같은 저자 그룹), GelEvent [13] | 고주파 촉각 입력. 논문은 "프레임 방식 GelSight와 이벤트 방식을 함께 쓴 첫 연구"라고 주장(§1, §3) |
| 프레임 방식 비전 촉각 센서 | GelSight [2], DIGIT [3], TacTip [4] | 접촉 기하용 입력(GelSight Mini). 시각촉각 조작 사례 [5–11] |
| 이질 센서·다중 센서 촉각 표현 | Sparsh-X [26] (Digit 360의 촉각 이미지, 오디오, IMU, 압력을 융합), AnyTouch [38] (정적-동적 통합 표현), T3 [39] (센서 사이 전이), Touch100k [25] | 기준선(Sparsh-X의 융합 구조)과 관련 연구 서술. 이들은 표현 학습이고 MiTaS는 정책과 종단 학습(§2) |
| 시각촉각 모방 학습 | Reactive Diffusion Policy [24] (느린-빠른 시각촉각), 촉각 조건 확산 정책 [27, 7] (같은 그룹), 3D-ViTac [18], MimicTouch [9], PolyTouch [46], 성냥 켜기 사례 연구 [21] | 비교 대상은 아니고 맥락. PolyTouch의 교차 어텐션 융합은 표 1에서 기준선으로 다시 구현 |
| 사전학습·마스킹 | VITaL [28], 마스크 멀티모달 학습 [44], 마스킹 관측 [21], 자기지도 대조 사전학습 [30] | MTD(§3.3)의 동기. 사전학습이 아니라 "학습 때만 센서를 쓰고 추론에서 뺀다"는 점이 다름 |
| 다중 감각 융합 | See-Hear-Feel [19], Stage-guided fusion [43], Visual-force-tactile [47], Making Sense of Vision and Touch [41, 42] | "교차 모달 융합이지 촉각 사이 융합은 아니다"라는 위치 지정(§2) |
| 융합 구조 | PolyTouch의 교차 어텐션 [46], BLIP-2 Q-Former [59], attention bottleneck [58], ViT [60] | 표 1의 기준선. 본 구조는 풀링 없는 토큰 644개와 완전 셀프 어텐션 |
| 정책 머리 | 플로 매칭 [49], π0.5 식의 플로 매칭 행동 생성 [48], DiT [54], AdaLN-Zero [54, 55], 생성 모델 서베이 [52] | DiT 머리 10층(§3.2). 센서 ablation에서 머리는 고정 |
| 초기 합성곱 | Early convolutions [50] | 센서별 합성곱 스템 설계의 근거(§3.1) |
| 과제 출처 | IndustReal [61] (Gear), FurnitureBench [62] (Lamp) | 5개 과제 중 둘 |
| 실험 파이프라인 | Nautilus [57] | 데이터 수집과 제어 |

가까운 대안(논문에 비교가 없는 것): 같은 센서를 쓰되 GelSight 프레임률을 높인 시간 해상도 대조(저자 가설은 "프레임 방식은 프레임 사이를 놓친다"이므로 직접 대응), 이벤트 촉각 하나만 쓰는 구조에서 시간 창만 바꾼 절제, 상태 입력(고유수용)을 넣은 같은 구조(논문은 상태를 일부러 뺌), 사전학습된 촉각 인코더(Sparsh 계열)를 얼린 채 쓰는 구성.

## 2. 코드와 저장소 (GitHub 표)

기준일 2026-11-26. 스타와 마지막 push는 GitHub API 값입니다. **공식 코드는 공개되어 있지 않습니다**(프로젝트 페이지 "Code (Coming Soon)").

| 저장소·자료 | 관계 | 라이선스 | 스타 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| 공식 코드 | **없음.** 프로젝트 페이지의 Code 단추는 "Coming Soon". GitHub 검색(`mitas`, `multi-resolution tactile`, `mitas-touch`)에서 구현 저장소를 찾지 못함 | - | - | - | 코드 대조 대신 주장 검증(`04`) |
| [mitas-touch.github.io](https://mitas-touch.github.io) | 프로젝트 페이지(Nerfies 템플릿). 논문, arXiv, Dataset 링크, 영상, 초록, 어텐션 그림, 결과 요약(85, 95, 83, 75, 63%) | - | - | - | 읽음 |
| [huggingface.co/datasets/RKrohn/MiTaS](https://huggingface.co/datasets/RKrohn/MiTaS) | 논문 데이터셋. 과제 폴더 6개(board_wiping, gear_assembly, key_inlock, lamp_install, lamp_install_dual_gelsight, lightbulb_connection)의 HDF5 589개 파일, 저장소 사용량 63,629,257,502 B(약 63.6 GB). 카드 태그 `license:cc-by-4.0`, arXiv 2606.06281 연결 | CC BY 4.0 (카드 태그) | - (likes 2, downloads 2,725) | 2026-11-08 수정 | **메타데이터만 읽음, 받지 않음.** 세션 구조는 `04` §1 |
| [nifunk/evetac_software](https://github.com/nifunk/evetac_software) | Evetac 센서 인터페이스(저자 그룹) | 라이선스 파일 없음 | 5 | 2024-09-09 | 참고 |
| [nifunk/evetac_design](https://github.com/nifunk/evetac_design) | Evetac 설계 파일 | 라이선스 파일 없음 | 12 | 2024-09-09 | 참고 |
| [facebookresearch/sparsh-multisensory-touch](https://github.com/facebookresearch/sparsh-multisensory-touch) | Sparsh-X(기준선 [26])와 Sparsh-Skin | 별도 고지(API: NOASSERTION, "Other") | 56 | 2025-08-20 | 참고. 논문이 기준선을 이 코드로 돌렸는지(CNN 스템으로 바꾼 것 포함)는 서술이 없음 |
| [facebookresearch/sparsh](https://github.com/facebookresearch/sparsh) | 첫 Sparsh | CC BY-NC 4.0 (API: NOASSERTION) | 225 | 2025-02-27 | 참고 |
| [xiaoxiaoxh/reactive_diffusion_policy](https://github.com/xiaoxiaoxh/reactive_diffusion_policy) | Reactive Diffusion Policy [24] | API: NOASSERTION (최상위 LICENSE는 MIT로 알려짐) | 398 | 2026-04-12 | 참고 |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 플로 매칭 행동 생성의 인용 대상 [48] 계열 | Apache-2.0 | 14,766 | 2026-08-24 | 참고 |
| [facebookresearch/DiT](https://github.com/facebookresearch/DiT) | DiT [54], AdaLN-Zero | API: NOASSERTION | 8,668 | 2024-05-31 | 참고 |
| [clvrai/furniture-bench](https://github.com/clvrai/furniture-bench) | Lamp 과제 출처 [62] | MIT | 244 | 2026-11-22 | 참고 |
| [NVlabs/industreallib](https://github.com/NVlabs/industreallib) | Gear 과제 출처 [61] | API: NOASSERTION | 85 | 2024-01-05 | 참고 |
| [salesforce/LAVIS](https://github.com/salesforce/LAVIS) | BLIP-2의 Q-Former [59] | BSD-3-Clause | 11,255 | 2026-09-18 | 참고 |
| Nautilus [57] | 논문의 데이터·제어 파이프라인 | - | - | - | 저장소를 찾지 못함 |

- 체크포인트, 학습 설정 파일, 평가 로그, 정책 코드는 공개되어 있지 않습니다. 데이터셋은 공개되어 있으나 어느 세션이 논문의 "첫 번째·두 번째 데이터셋"인지는 논문과 페이지에 적혀 있지 않습니다(`04` §1).

## 4. 참고: 이 논문의 주장이 기대는 선행 결과

- "프레임 방식과 이벤트 방식 촉각 센서를 결합한 첫 연구"(§1, §3)는 인용 범위와의 비교에 기댑니다. 이 분석은 인용된 문헌 밖으로 검색 범위를 넓히지 않았습니다. 확인할 수 있는 것은 이 논문이 인용한 이벤트 촉각 연구 [12, 13]와 다중 센서 촉각 표현 연구 [26, 38, 39] 중에 "두 방식을 정책에 함께 넣어 비교한" 연구가 목록에 없다는 점까지입니다(`04` C14).
- 같은 그룹의 선행 연구: Evetac [12]의 센서 논문, 촉각 조건 확산 정책 [7], 성냥 켜기 사례 연구 [21], 멀티센서 RL 사전학습 [56]. MiTaS는 이 그룹의 센서 위에서 정책 구조를 평가하는 후속입니다.
