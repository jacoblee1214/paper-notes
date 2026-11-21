# 02. 선행연구와 관련 자료

## 1. 계보

SHAPE는 세 줄기를 합칩니다. 각 줄기에서 어떤 구성 요소를 가져왔는지와 SHAPE가 다르게 한 점을 적습니다(논문 Sec. 2 참조).

| 줄기 | 구성 요소 | 출처 (원문 참고문헌) | SHAPE의 차이 |
|---|---|---|---|
| VLA 이산 토큰 | 행동 청크를 토큰으로 예측 | RT-2류 (Brohan et al., 2023b), OpenVLA (Kim et al., 2024), **FAST** (Pertsch et al., RSS 2025) | 사람과 로봇 청크를 같은 어휘로 쓰고, 토큰을 **보조 목표**로만 씀 (Sec. 3.3) |
| 흐름 매칭 전문가 | 백본 위 연속 행동 전문가 | π0 (Black et al., RSS 2025), π0.5 (Physical Intelligence et al., 2025) | 전문가를 **두 스트림**(EEF, 관절)으로 나누고 주의 마스크로 분리 |
| 그래디언트 분리 | 토큰 손실이 전문가 그래디언트를 막음 | π0.5 knowledge insulation (Driess et al., 2025) | 그래디언트를 **막지 않고** 둘 다 훈련 (Sec. 2, 6.3) |
| 사람 영상의 행동 지도 | EEF 공간에서 사람 손 행동을 지도 | EgoMimic (Kareer et al., 2025), Phantom (Lepert et al., CoRL 2025), π0.5+ego (Kareer et al., RSS 2026), EgoLAP (Zha et al., 2026), Ego-Pi (Kim et al., CVPR Findings 2026), H-RDT (Bi et al., AAAI 2026), Nexus-VLA (Guan et al., arXiv:2610.42021) | 같은 설정에서 토큰 효과를 **분리한 2×2 실험** (Sec. 1, 5.1) |
| 사람 데이터 규모 | 대규모 사람 영상 사전학습 | GR00T N1.7(NVIDIA, 2026)의 사전학습이 인용한 Zheng et al., 2026(20k시간, EgoScale로 보이나 원문으로 확정 못함) | SHAPE의 사람 데이터는 2.8k시간 (Sec. 4) |
| 손 행동 표현 | 관절 대 EEF 공간의 안정성과 일반화 | Feng et al., 2026 (팔과 그리퍼 대규모 연구), Wang et al., 2024a (IK 기반 EEF) | 다지 손에서 두 표현을 **같은 체크포인트로** 비교 (Sec. 5.3) |
| 비디오 보조 목표 | 미래 프레임 예측 | Wan2.1 (Team Wan et al., 2025, arXiv:2503.20314), 백본 Qwen3-VL (Bai et al., 2025, arXiv:2511.21631) | 같은 방식 (A.1) |

**가장 가까운 대안.** 같은 질문(사람 데이터를 어떻게 지도할지)에 답하는 논문은 π0.5+ego(Kareer et al., RSS 2026)입니다. 이 논문은 FAST 토큰과 흐름 매칭을 사람 데이터에 함께 씁니다. 저자들은 두 손실을 분리하지 않았다고 지적합니다(Sec. 2). SHAPE의 차이는 "토큰을 분리된 효과로 측정하는 실험"과 "두 스트림 배치"입니다. 두 논문을 직접 비교하는 수치는 원문에 없습니다.

## 2. 공식 코드와 저장소

검색 날짜 2026-11-21. 공식 코드는 찾지 못했습니다. `repo/` 클론은 하지 않았습니다.

| 항목 | 결과 | 확인 방법 |
|---|---|---|
| 프로젝트 페이지 | https://li-yaping.github.io/SHAPE/ (정적 페이지, 코드 링크 없음, arXiv BibTeX 링크만) | 페이지 HTML의 모든 `href` |
| arXiv 주석 | "Project page: ..." 한 줄. 코드 링크 없음 | abs 페이지 Comments |
| GitHub 검색 | "SHAPE egocentric", "Supervision from Human Action tokens", "2611.24585" 검색 결과 0건 (저장소). 코드 검색 결과는 논문 목록 저장소 4개뿐 | `gh search repos`, `gh api search/repositories`, `gh api search/code` |
| 저자 계정 | `li-yaping/li-yaping.github.io` (개인 홈페이지, Jekyll 템플릿, MIT, 별 0개, 최종 push 2026-11-20). 논문 코드 없음 | `gh api repos/...` |
| 소속 조직 | `sharpa-robotics` 공개 저장소 22개 중 SHAPE나 VLA 학습 코드는 없음. 관련 저장소는 `WM-Craftnet`(CoRL'2026 Spotlight, 별 55, 최종 push 2026-10-29)로 다른 논문입니다. 손 SDK, 촉각 센서 자료 등 하드웨어 저장소가 있습니다 | `gh api orgs/sharpa-robotics/repos` |
| 체크포인트, 데이터 | 공개 언급 없음 | 원문과 프로젝트 페이지 |

**결과:** 공식 코드가 없으므로 `04_claim_verification.md`(주장 C1–Cn)로 검증합니다.

## 4. 사용 시 주의

- 이 파일의 표는 원문과 공개 페이지에서 확인한 사실만 적었습니다. 저자 계정이나 조직에 대한 정보는 공개 저장소 메타데이터만 사용했습니다.
- 원문의 참고문헌 가운데 일부는 arXiv 번호가 없어서, 연도와 학회 이름만 적었습니다.
