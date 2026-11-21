# 02. 선행연구와 관련 자료

## 1. 계보 (논문이 기대는 것)

| 구성 요소 | 출처 (논문 참고문헌 번호) | 이 논문에서 하는 일 |
|---|---|---|
| 확산 모델 (노이즈 예측 목적) | Sohl-Dickstein, Song 외 [57, 59] | 식 (1)의 기반 |
| 영상 확산 트랜스포머, 3D VAE | CogVideoX [79] | 기반 모델 (5B-I2V) |
| 자기회귀 영상 생성 | History-guided diffusion [58], Genie 2 [50] | 작업 기억의 틀 (최근 프레임을 조건으로) |
| 카메라 제어 영상 | DaS [25], TrajectoryCrafter [82], Wan2.1 [66] | 기준선이자 사전학습 가중치의 출처 (DaS) |
| 점군 기반 3D 일관성 | GEN3C [54], TrajectoryCrafter [82], DaS [25] | 정적 점군 렌더링을 조건으로 씀 |
| 온라인 3D 복원 | CUT3R [69] (추론 중), DUSt3R [70] 계열 | 생성 중 점 지도 복원 (부록 A) |
| 동적 장면 SfM | MegaSaM [44] | 데이터셋 구축 (카메라와 깊이) |
| TSDF 융합 | TSDF fusion [84] (논문 참고문헌 번호, 원저자 표기는 확인 안 함) | 정적 공간 기억의 누적과 동적 제거 |
| 프레임 압축으로 문맥 늘림 | FramePack [87] (Zhang & Agrawala) | 비교 대상으로만 언급 (상보적이라고 주장) |
| 제어 신호 주입 | ControlNet [88] | 조건 DiT의 0 초기화 선형층 방식 |
| 평가 | VBench [36], LPIPS, SSIM, PSNR, AHR (ControlNet 방식) | Table 1, 2 |
| 데이터 | MiraData [39] | 원영상 |
| 주석 | Qwen [77] (Qwen2.5-VL 계열) | target 행동 문장 생성 |


## 2. 공식 코드와 관련 저장소

| 저장소 | 관계 | 라이선스 (GitHub API) | 별 | 마지막 push | 상태 |
|---|---|---|---|---|---|
| [spmem/spmem](https://github.com/spmem/spmem) | 공식 구현 (추론과 학습) | **Apache-2.0** (LICENSE.txt 확인) | 86 | 2026-05-11 (생성 2026-03-18) | 얕은 복제 `repo/`, 커밋 `89e1f70e` (2026-05-11). 정적 읽기만 |
| [spmem.github.io](https://spmem.github.io/) | 프로젝트 페이지 | - | - | - | 헤더에 "CVPR 2024"(학회 표기 불일치). Code 버튼 옆에 "[TODO]" 표기가 있음 |
| ysmikey/spmem_megadata (Hugging Face) | 데이터 (약 80K 클립) | - | - | - | 받지 않음 |
| ysmikey/spmem_ckpt (Hugging Face) | 체크포인트 | - | - | - | 받지 않음 |
| Qwen2.5-VL-7B-Instruct | 추론 중 프롬프트 생성 | (Qwen 라이선스) | - | - | 받지 않음 |
| [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) | 추론 중 깊이와 자세 (`repo/Depth-Anything-3`에 벤더링) | Apache-2.0 | 6,720 | 2026-07-27 | 정적 읽기만 |
| [zai-org/CogVideo](https://github.com/zai-org/CogVideo) (구 THUDM/CogVideo) | 기반 모델 코드 (CogVideoX) | Apache-2.0 | 13,217 | 2025-11-04 | 참고만 |
| [IGL-HKUST/DiffusionAsShader](https://github.com/IGL-HKUST/DiffusionAsShader) | DaS, 기준선이자 사전학습 출처 | Apache-2.0 | 853 | 2025-06-09 | 참고만 |
| [TrajectoryCrafter/TrajectoryCrafter](https://github.com/TrajectoryCrafter/TrajectoryCrafter) | 기준선 | GitHub는 NOASSERTION (LICENSE 파일 직접 확인 안 함) | 897 | 2025-12-17 | 참고만 |
| [CUT3R/CUT3R](https://github.com/CUT3R/CUT3R) | 추론 중 온라인 복원 (논문 부록 A) | GitHub는 NOASSERTION (LICENSE 파일 있음, 직접 확인 안 함) | 1,524 | 2025-08-27 | 참고만. **저장소 코드에는 CUT3R 호출이 없음** (`04` D2) |
| [mega-sam/mega-sam](https://github.com/mega-sam/mega-sam) | 데이터 구축 (논문 §3.4) | Apache-2.0 | 1,372 | 2026-01-05 | 참고만. **저장소에 MegaSaM 코드 없음** (`04` D3) |

별 수와 push 날짜는 2026-11-21 GitHub API 조회값입니다. 라이선스 필드가 NOASSERTION이면 LICENSE 파일을 직접 읽지 않은 것입니다.

## 4. 판단

- 이 논문은 **로봇 논문이 아닙니다.** 행동 입력은 카메라 이동과 텍스트 행동 문장이고, 목표는 영상 품질과 장면 일관성입니다. 로봇 정책 학습이나 조작 과제는 없습니다.
- 그래서 저장소의 로봇 세계 모델 분석과 직접 비교할 실험은 없습니다. 연결은 기억 설계 수준입니다.
