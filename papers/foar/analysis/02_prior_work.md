# 02. 선행연구, 구성 요소의 출처, 공식 저장소

논문 번호 [n]은 원문 참고문헌 번호입니다. 번호를 확인하지 못한 문헌은 저자명과 연도로만 적었습니다.

## 1. 선행연구 계보

FoAR은 세 흐름이 만나는 자리에 있습니다.

1. **시각 기반 모방 학습.** ACT [52]는 행동 청크와 시간 앙상블을, Diffusion Policy [10]는 확산 행동 헤드를 제시했습니다. RISE [44]는 희소 3D 점군과 Transformer를 결합해 실로봇 모방을 단순하게 만들었습니다. FoAR은 RISE를 그대로 상속합니다(§III-A).
2. **힘·토크를 정책 입력으로 쓰는 흐름.** 힘을 입력이나 출력에 넣은 정책이 있습니다. Diffusion Policy를 힘 도메인으로 옮긴 TacDiffusion [46]은 힘 공간에서 확산합니다. 임피던스 제어에 쓸 힘을 예측하는 연구 [46], 어드미턴스 제어에 쓸 목표 힘을 예측하는 Admittance Visuomotor Policy [53], 힘과 위치 하이브리드 제어를 위한 ForceMimic [31] 계열이 있습니다. 적응형 컴플라이언스(ACP, Hou et al., arXiv:2410.09309)는 확산 정책의 강성을 예측합니다. 이들 대부분은 "잡고 있는 물체" 또는 "고정된 도구"를 가정하고, 비접촉 구간의 힘 잡음은 다루지 않습니다(§II-A).
3. **다중 감각 융합.** 오디오(ManiWAV 계열), 촉각(Visuo-tactile Transformers [9], 3D-ViTac [22]), 힘을 Transformer 토큰으로 넣는 방식(RISE force-token, §IV-B의 변형)이 있습니다. 논문은 이들 융합이 비접촉 구간에서 힘 잡음을 키운다고 봅니다(§II-B).

FoAR의 차별점은 **미래 접촉 확률 ϕ로 융합 가중을 정하고, 같은 ϕ로 실행 단계의 행동을 보정하는 것**입니다. 두 번째는 제어기 설계이며, 임피던스 파라미터를 학습하지 않고 위치 명령만 보냅니다. 논문이 "simple position control"이라고 부르는 이유입니다(§III-C).

## 2. 구성 요소별 출처

| 구성 요소 | 출처 | 논문 표시 |
|---|---|---|
| 점군 인코더 (희소 3D, ResNet 얕은 구조) | RISE [44], MinkowskiEngine [11] | 명시 (§III-B, App. A) |
| 힘 Transformer (정현파 시간 부호화) | Transformer [43] | 명시 |
| 확산 행동 헤드 (CNN, 100/20 단계) | Diffusion Policy [10], DDPM [18], DDIM [40] | 명시 |
| 시간 앙상블 (버퍼 두 개) | ACT [52] | 명시 (§III-C) |
| 예측기 백본 ResNet18 | He et al. [16] | 명시 |
| 접촉 라벨 생성 (힘 임계값) | 논문이 자체 제안 | App. A |
| 미래 접촉 확률로 융합 가중 | 논문이 자체 제안 | §III-B 식 (융합) |
| 반응 제어 (Alg. 1) | 논문이 자체 제안. 착안점은 ACP [21], 컴플라이언스 제어 계열 [21, 24] | §III-C |
| 데이터 수집 (햅틱 원격 조작) | RH20T [14] | §IV-A |

## 4. 공식 코드와 관련 저장소

이 표는 2026-11-22에 GitHub API와 저장소 페이지로 확인했습니다.

| 저장소 | 역할 | 라이선스 | 별 | 최근 push | 커밋 (분석 기준) | 상태 |
|---|---|---|---|---|---|---|
| [Alan-Heoooh/FoAR](https://github.com/Alan-Heoooh/FoAR) | 공식 구현 (학습, 평가, 로봇 인터페이스) | LICENSE 파일 없음 (GitHub API license: null) | 70 | 2026-08-07 (API `pushed_at`) | `869abd3` (2025-11-13, "fix typo (issue #6)") | `repo/`에 얕은 clone (정적 읽기만) |
| [tonyfang.net/FoAR](https://tonyfang.net/FoAR/) | 프로젝트 페이지 | 페이지 하단 CC BY-SA 4.0 (웹사이트 템플릿 표기. 논문 라이선스와 다름) | - | - | - | 읽음 |
| [rise-policy/RISE](https://github.com/rise-policy/RISE) | 기반 정책 (IROS 2024) | Other (NOASSERTION) | 163 | 2025-11-29 | 받지 않음 | 확인만 함 |
| [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) | 확산 행동 헤드의 출처 (RSS 2023) | MIT | 4,737 | 2024-12-24 | 받지 않음 | 확인만 함 |
| [tonyzhaozh/act](https://github.com/tonyzhaozh/act) | 시간 앙상블 출처 (RSS 2023) | MIT | 2,352 | 2024-07-23 | 받지 않음 | 확인만 함 |
| [chenxi-wang/MinkowskiEngine](https://github.com/chenxi-wang/MinkowskiEngine) | 저자가 고친 MinkowskiEngine 포크 (설치 문서가 요구) | Other (NOASSERTION) | 7 | 2026-07-01 | 받지 않음 | 설치하지 않음 |

- 공식 저장소에는 LICENSE 파일이 없습니다. GitHub 라이선스 API는 `null`을 돌려줍니다. 공개 재배포 허가를 확인할 수 없으므로 코드는 공개본에 넣지 않았습니다.
- 공식 저장소의 push 날짜(2026-08-07)는 기본 브랜치의 마지막 커밋(2025-11-13)과 다릅니다. 기본 브랜치 밖의 갱신이거나 메타데이터 갱신일 가능성이 있습니다. 분석은 clone한 기본 브랜치 `869abd3`로 했습니다.
- 프로젝트 페이지 BibTeX는 `journal = {arXiv preprint}`이고 연도는 2024입니다. RA-L 게재 정보는 저장소 README의 BibTeX에 있습니다(`doi=10.1109/LRA.2025.3560871`).
