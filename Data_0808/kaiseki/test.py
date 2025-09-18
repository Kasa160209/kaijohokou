import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 設定一括管理
# =========================
DEFAULTS = {
    "midhip_index": 8,                # OpenPose BODY_25 の MidHip
    "fps": 60,                        # フレームレート
    "pixel_to_meter": 2.0 / 844.0,    # px -> m 変換係数
    "center_x": 1920,                 # 画面中心のx座標
    "bound_half_width": 400,          # 中央からの半幅 固定
    "conf_thresh": 0.5,               # MidHip信頼度の下限
    "choose": "leftmost",             # "leftmost" or "rightmost"
    "direction": "right_to_left",     # "right_to_left" or "left_to_right"
    "show_from_cut_plot": True,       # 切り出し後のプロット
    "show_full_plot": True,           # 全体プロット
}

# ==== 変更できる窓 ファイル名とスタートフレームだけをまとめる ====
CONFIGS = [
    {
        "name": "pa4_pt1_cali",
        "json_dir": r"C:/Users/kasa1/YMMT_kadai/Data_0808/key/pa4_pt1_cali",
        "start_frame": 2600,
    },
    {
        "name": "pa4_pt5_cali",
        "json_dir": r"C:/Users/kasa1/YMMT_kadai/Data_0808/key/pa4_pt5_cali",
        "start_frame": 600,
    },
    {
        "name": "pa4_pt6_cali",
        "json_dir": r"C:/Users/kasa1/YMMT_kadai/Data_0808/key/pa4_pt6_cali",
        "start_frame": 700,
    },
    {
        "name": "pa4_saisyou_cali",
        "json_dir": r"C:/Users/kasa1/YMMT_kadai/Data_0808/key/pa4_saisyou_cali",
        "start_frame": 1200,
    },
]

# =========================
# ユーティリティ
# =========================
def fill_none(arr):
    arr = np.array([v if v is not None else np.nan for v in arr])
    nans = np.isnan(arr)
    if np.all(nans):
        return np.zeros_like(arr)
    if np.sum(~nans) < 2:
        return np.full_like(arr, np.nan)
    interp_vals = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), arr[~nans])
    arr[nans] = interp_vals
    return arr

def choose_person_x(people, midhip_index, conf_thresh, mode="leftmost"):
    candidates = []
    for person in people:
        kp = person["pose_keypoints_2d"]
        x, y, c = kp[midhip_index*3: midhip_index*3+3]
        if c > conf_thresh:
            candidates.append(x)
    if not candidates:
        return None
    return min(candidates) if mode == "leftmost" else max(candidates)

def load_midhip_x_series(json_dir, midhip_index, conf_thresh, choose):
    midhip_x_all = []
    frame_numbers = []
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    for frame_idx, file_name in enumerate(json_files):
        with open(os.path.join(json_dir, file_name), "r") as f:
            data = json.load(f)
        people = data.get("people", [])
        if not people:
            midhip_x_all.append(None)
            frame_numbers.append(frame_idx)
            continue
        x = choose_person_x(people, midhip_index, conf_thresh, choose)
        midhip_x_all.append(x if x is not None else None)
        frame_numbers.append(frame_idx)
    return np.array(frame_numbers), fill_none(midhip_x_all)

def compute_enter_exit(midhip_x_cut, left_bound, right_bound, fps, direction):
    enter_idx = exit_idx = None
    if direction == "right_to_left":
        enter_candidates = np.where(midhip_x_cut <= right_bound)[0]
        exit_candidates  = np.where(midhip_x_cut <= left_bound)[0]
    else:
        enter_candidates = np.where(midhip_x_cut >= left_bound)[0]
        exit_candidates  = np.where(midhip_x_cut >= right_bound)[0]

    if len(enter_candidates) > 0 and len(exit_candidates) > 0:
        enter_idx = enter_candidates[0]
        exit_idx = exit_candidates[0]
        if exit_idx > enter_idx:
            time_s = (exit_idx - enter_idx) / fps
            distance_px = abs(right_bound - left_bound)
            return enter_idx, exit_idx, time_s, distance_px
    return None, None, None, None

def plot_segment(frame_numbers, x_series, left_bound, right_bound, center_x,
                 enter_idx, exit_idx, title):
    plt.figure(figsize=(12,5))
    plt.plot(frame_numbers, x_series, label="MidHip X", color="blue")
    plt.axhline(y=left_bound,  color="red",   linestyle="--", label=f"Left Bound ({left_bound}px)")
    plt.axhline(y=right_bound, color="green", linestyle="--", label=f"Right Bound ({right_bound}px)")
    plt.axhline(y=center_x,    color="gray",  linestyle=":",  label=f"Center ({center_x}px)")
    if enter_idx is not None and exit_idx is not None and exit_idx > enter_idx:
        plt.scatter(frame_numbers[enter_idx], x_series[enter_idx], color="green", s=80, label="Enter")
        plt.scatter(frame_numbers[exit_idx],  x_series[exit_idx],  color="red",   s=80, label="Exit")
    plt.xlabel("Frame")
    plt.ylabel("MidHip X Position (px)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# =========================
# メイン処理
# =========================
def main():
    # プロット用と解析用に必要なデータを保持
    stash = []

    # まずは全データセットでグラフを出す 全体→切り出し の順番で表示
    for cfg in CONFIGS:
        par = {**DEFAULTS, **cfg}
        name = par["name"]
        json_dir = par["json_dir"]
        midhip_index = par["midhip_index"]
        fps = par["fps"]
        center_x = par["center_x"]
        half_w = par["bound_half_width"]
        conf_thresh = par["conf_thresh"]
        choose = par["choose"]
        start_frame = par["start_frame"]

        left_bound  = center_x - half_w
        right_bound = center_x + half_w

        frames_full, midhip_x_full = load_midhip_x_series(json_dir, midhip_index, conf_thresh, choose)

        frames_cut = frames_full[start_frame:]
        midhip_x_cut = midhip_x_full[start_frame:]

        # 解析結果は後回しにするため 今は None を渡して枠線だけ表示
        if par["show_full_plot"]:
            title_full = f"{name} MidHip X trajectory full video"
            plot_segment(frames_full, midhip_x_full, left_bound, right_bound, center_x,
                         None, None, title_full)

        if par["show_from_cut_plot"]:
            title_cut = f"{name} MidHip X trajectory from frame {start_frame}"
            plot_segment(frames_cut, midhip_x_cut, left_bound, right_bound, center_x,
                         None, None, title_cut)

        # 後で解析するために保存
        stash.append({
            "name": name,
            "fps": fps,
            "pixel_to_meter": par["pixel_to_meter"],
            "direction": par["direction"],
            "left_bound": left_bound,
            "right_bound": right_bound,
            "frames_cut": frames_cut,
            "midhip_x_cut": midhip_x_cut,
        })

    # すべてのグラフを出した後で解析をまとめて実行
    for item in stash:
        name = item["name"]
        fps = item["fps"]
        pixel_to_meter = item["pixel_to_meter"]
        direction = item["direction"]
        left_bound = item["left_bound"]
        right_bound = item["right_bound"]
        frames_cut = item["frames_cut"]
        midhip_x_cut = item["midhip_x_cut"]

        enter_idx, exit_idx, time_s, distance_px = compute_enter_exit(
            midhip_x_cut, left_bound, right_bound, fps, direction
        )

        if time_s is not None:
            distance_m = distance_px * pixel_to_meter
            walking_speed = distance_m / time_s if time_s > 0 else np.nan
            print(f"[{name}] 通過距離: {distance_px} px = {distance_m:.3f} m")
            print(f"[{name}] 通過時間: {time_s:.2f} 秒")
            print(f"[{name}] 歩行速度: {walking_speed:.3f} m/s")
        else:
            print(f"[{name}] 範囲通過を検出できませんでした")

if __name__ == "__main__":
    main()
