"""GR00T N1 (arXiv:2503.14734 v2) 논문 표 숫자를 옮겨 적은 모듈.

출처: paper/paper.pdf 의 Table 2(p.15), Table 3(p.15), Table 4(p.26), Table 5(p.27), Table 7(p.28), 본문 4.4 절.
값은 원문 표를 그대로 옮겼다 (단위: 성공률 %, 실로봇은 부분 점수 포함).
"""

# Table 4: 시뮬 과제별 성공률. (과제명, DP@30, DP@100, DP@300, GR00T@30, GR00T@100, GR00T@300)
ROBOCASA = [
    ("Close Double Door", 1.7, 26.5, 60.8, 0.0, 43.1, 74.5),
    ("Close Drawer", 57.5, 88.2, 94.1, 76.9, 96.1, 99.0),
    ("Close Single Door", 21.7, 46.1, 72.6, 49.1, 67.7, 83.3),
    ("Coffee Press Button", 32.5, 46.1, 91.2, 27.8, 56.9, 85.3),
    ("Coffee Serve Mug", 6.7, 28.4, 66.7, 3.7, 34.3, 72.6),
    ("Coffee Setup Mug", 0.0, 19.6, 32.4, 0.0, 2.0, 22.6),
    ("Open Double Door", 0.0, 9.8, 18.6, 0.0, 12.8, 14.7),
    ("Open Drawer", 15.8, 42.2, 61.8, 9.3, 42.2, 79.4),
    ("Open Single Door", 36.7, 42.2, 57.8, 20.4, 54.9, 58.8),
    ("PnP Cab to Counter", 2.5, 4.9, 9.8, 0.9, 3.9, 19.6),
    ("PnP Counter to Cab", 0.0, 2.9, 10.8, 1.9, 6.9, 36.3),
    ("PnP Counter to Microwave", 0.0, 2.0, 8.8, 0.0, 0.0, 12.8),
    ("PnP Counter to Sink", 0.0, 0.0, 13.7, 0.0, 1.0, 9.8),
    ("PnP Counter to Stove", 0.0, 1.0, 17.7, 0.0, 0.0, 23.5),
    ("PnP Microwave to Counter", 0.0, 2.0, 11.8, 0.0, 0.0, 15.7),
    ("PnP Sink to Counter", 4.2, 8.8, 42.2, 0.0, 5.9, 33.3),
    ("PnP Stove to Counter", 1.7, 2.9, 23.5, 0.0, 0.0, 29.4),
    ("Turn Off Microwave", 63.3, 53.9, 52.0, 47.2, 57.8, 70.6),
    ("Turn Off Sink Faucet", 21.7, 63.7, 72.6, 49.1, 67.7, 72.6),
    ("Turn Off Stove", 5.0, 10.8, 19.6, 4.6, 15.7, 26.5),
    ("Turn On Microwave", 30.0, 51.0, 75.5, 55.6, 73.5, 78.4),
    ("Turn On Sink Faucet", 31.7, 27.5, 63.7, 33.3, 59.8, 62.8),
    ("Turn On Stove", 12.5, 22.6, 36.3, 14.8, 25.5, 55.9),
    ("Turn Sink Spout", 8.3, 11.8, 23.5, 24.1, 42.2, 52.9),
]
DEXMG = [
    ("Can Sort", 82.8, 93.1, 99.4, 94.8, 98.0, 98.0),
    ("Coffee", 35.5, 68.1, 79.7, 44.9, 79.4, 73.5),
    ("Pouring", 37.0, 62.3, 68.8, 54.4, 71.6, 87.3),
    ("Threading", 4.2, 18.3, 27.5, 3.9, 37.3, 60.8),
    ("Three Piece Assembly", 10.0, 32.5, 63.3, 10.8, 43.1, 69.6),
    ("Transport", 7.5, 25.0, 53.3, 7.8, 48.0, 61.8),
    ("Box Cleanup", 30.0, 80.8, 97.5, 33.3, 29.4, 95.1),
    ("Drawer Cleanup", 1.7, 16.7, 52.5, 10.8, 42.2, 55.9),
    ("Lift Tray", 5.0, 25.0, 73.3, 5.8, 77.5, 65.7),
]
GR1 = [
    ("Cutting Board to Pot", 22.6, 37.3, 48.0, 58.8, 57.8, 57.8),
    ("Cutting Board to Basket", 19.6, 42.2, 29.4, 43.1, 61.8, 56.9),
    ("Cutting Board to Tiered Basket", 13.7, 13.7, 18.6, 13.7, 23.5, 34.3),
    ("Cutting Board to Pan", 28.4, 48.0, 57.8, 67.7, 65.7, 68.6),
    ("Cutting Board to Cardboard Box", 11.8, 15.7, 22.6, 31.4, 30.4, 33.3),
    ("Placemat to Bowl", 14.7, 18.6, 23.5, 31.4, 39.2, 39.2),
    ("Placemat to Plate", 15.7, 23.5, 37.3, 33.3, 37.3, 49.0),
    ("Placemat to Basket", 15.7, 25.5, 41.2, 50.0, 46.1, 55.9),
    ("Placemat to Tiered Shelf", 6.9, 5.9, 11.8, 11.8, 21.6, 19.6),
    ("Plate to Pan", 13.7, 17.7, 35.3, 35.3, 48.0, 52.9),
    ("Plate to Cardboard Box", 12.8, 13.7, 27.5, 34.3, 38.2, 32.4),
    ("Plate to Bowl", 15.7, 18.6, 31.4, 41.2, 42.2, 34.3),
    ("Plate to Plate", 25.5, 39.2, 61.8, 72.6, 85.3, 68.6),
    ("Tray to Tiered Shelf", 2.0, 6.9, 15.7, 17.7, 27.5, 14.7),
    ("Tray to Tiered Basket", 12.8, 34.3, 39.2, 33.3, 49.0, 45.1),
    ("Tray to Plate", 26.5, 41.2, 49.0, 53.9, 68.6, 62.8),
    ("Tray to Cardboard Box", 21.6, 37.3, 40.2, 51.0, 55.9, 54.9),
    ("Tray to Pot", 21.6, 48.0, 52.9, 52.0, 59.8, 65.7),
    ("Wine to Cabinet", 43.1, 55.9, 60.8, 57.8, 53.9, 62.8),
    ("Place Bottle to Cabinet", 40.2, 62.8, 60.8, 60.8, 81.4, 74.5),
    ("Place Milk to Microwave", 37.3, 41.2, 51.0, 42.2, 58.8, 49.0),
    ("Potato to Microwave", 17.7, 30.4, 41.2, 30.4, 26.5, 34.3),
    ("Cup to Drawer", 24.5, 32.4, 36.3, 36.3, 44.1, 40.2),
    ("Can to Drawer", 48.0, 74.5, 75.5, 77.5, 76.5, 75.5),
]
# Table 4 하단에 인쇄된 벤치마크 평균: (DP@30, DP@100, DP@300, GR00T@30, GR00T@100, GR00T@300)
T4_PRINTED_AVG = {
    "RoboCasa": (14.7, 25.6, 43.2, 17.4, 32.1, 49.6),
    "DexMG": (23.7, 46.9, 68.4, 29.6, 58.5, 74.2),
    "GR-1": (21.3, 32.7, 40.4, 43.2, 50.0, 49.3),
}
TASKS = {"RoboCasa": ROBOCASA, "DexMG": DEXMG, "GR-1": GR1}

# Table 2 (본문 4.4, 100 demos/task): 열 = 벤치마크, 행 = 방법. 평균 열은 인쇄값.
T2 = {
    "RoboCasa": {"BC-Transformer": 26.3, "Diffusion Policy": 25.6, "GR00T-N1-2B": 32.1},
    "DexMG": {"BC-Transformer": 53.9, "Diffusion Policy": 56.1, "GR00T-N1-2B": 66.5},
    "GR-1": {"BC-Transformer": 16.1, "Diffusion Policy": 32.7, "GR00T-N1-2B": 50.0},
}
T2_AVG = {"BC-Transformer": 26.4, "Diffusion Policy": 33.4, "GR00T-N1-2B": 45.0}
N_TASKS = {"RoboCasa": 24, "DexMG": 9, "GR-1": 24}

# Table 5: 실로봇 과제별 성공률 (%). (그룹, 과제, DP 10%, DP 전체, GR00T 10%, GR00T 전체, 시행 수)
# 시행 수: 과제(행)당 10회, Machinery Packing만 5회 (본문 4.3).
REAL_ROWS = [
    ("PnP-seen", "Tray to Plate", 0.0, 20.0, 40.0, 100.0, 10),
    ("PnP-seen", "Cutting Board to Basket", 0.0, 30.0, 10.0, 100.0, 10),
    ("PnP-seen", "Cutting Board to Pan", 0.0, 60.0, 60.0, 80.0, 10),
    ("PnP-seen", "Plate to Bowl", 0.0, 40.0, 30.0, 100.0, 10),
    ("PnP-seen", "Placemat to Basket", 10.0, 60.0, 40.0, 80.0, 10),
    ("PnP-unseen", "Tray to Plate", 0.0, 20.0, 30.0, 80.0, 10),
    ("PnP-unseen", "Cutting Board to Basket", 10.0, 20.0, 60.0, 60.0, 10),
    ("PnP-unseen", "Cutting Board to Pan", 0.0, 40.0, 40.0, 80.0, 10),
    ("PnP-unseen", "Plate to Bowl", 0.0, 20.0, 10.0, 40.0, 10),
    ("PnP-unseen", "Placemat to Basket", 10.0, 50.0, 30.0, 100.0, 10),
    ("Articulated", "White Drawer", 6.6, 36.4, 26.4, 79.9, 10),
    ("Articulated", "Dark Cabinet", 0.0, 46.2, 86.6, 69.7, 10),
    ("Articulated", "Wooden Chest", 36.4, 33.2, 72.9, 63.2, 10),
    ("Industrial", "Machinery Packing", 20.0, 44.0, 8.0, 56.0, 5),
    ("Industrial", "Mesh Cup Pouring", 0.0, 62.5, 65.0, 67.5, 10),
    ("Industrial", "Cylinder Handover", 0.0, 76.5, 20.0, 86.6, 10),
    ("Coordination", "Coordination Part 1", 45.0, 65.0, 70.0, 80.0, 10),
    ("Coordination", "Coordination Part 2", 10.0, 60.0, 30.0, 85.0, 10),
]
# Table 3 인쇄값: 그룹별 평균 (DP 10%, DP 전체, GR00T 10%, GR00T 전체)
T3_PRINTED = {
    "PnP": (3.0, 36.0, 35.0, 82.0),
    "Articulated": (14.3, 38.6, 62.0, 70.9),
    "Industrial": (6.7, 61.0, 31.0, 70.0),
    "Coordination": (27.5, 62.5, 50.0, 82.5),
    "Average": (10.2, 46.4, 42.6, 76.8),
}
# 본문 4.4 인쇄값: 사전학습 평가 (15회 중 성공 수, 부분 점수 0.5 포함)
PRETRAIN = {"coordinated_bimanual": (11.5, 15, 76.6), "novel_object": (11.0, 15, 73.3)}

# Table 7: 사전학습 데이터 (프레임 수 M, 시간 h). 합계 행은 인쇄값.
T7_ROBOT = [("GR-1 Teleop", 6.4, 88.4), ("DROID (OXE)", 23.1, 428.3), ("RT-1 (OXE)", 3.7, 338.4),
            ("Language Table (OXE)", 7.0, 195.7), ("Bridge-v2 (OXE)", 2.0, 111.1),
            ("MUTEX (OXE)", 0.362, 5.0), ("Plex (OXE)", 0.077, 1.1), ("RoboSet (OXE)", 1.4, 78.9),
            ("Agibot-Alpha", 213.8, 1979.4), ("RH20T-Robot", 4.5, 62.5)]
T7_HUMAN = [("Ego4D", 154.4, 2144.7), ("Ego-Exo4D", 8.9, 123.0), ("Assembly-101", 1.4, 19.3),
            ("HOI4D", 0.892, 12.4), ("HoloAssist", 12.2, 169.6), ("RH20T-Human", 1.2, 16.3),
            ("EPIC-KITCHENS", 2.3, 31.7)]
T7_SIM = (125.5, 1742.6)     # GR-1 Simulation Pre-Training, FPS 20
T7_NEURAL = (23.8, 827.3)    # GR-1 Neural Videos, FPS 8
T7_PRINTED = {"robot": (262.3, 3288.8), "human": (181.3, 2517.0), "sim": (125.5, 1742.6),
              "neural": (23.8, 827.3), "total": (592.9, 8375.7)}
