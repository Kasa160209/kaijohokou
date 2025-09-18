import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pywt
import os

# ====== スケール ======
ACC_SCALE = 9.80665e-4
GYRO_SCALE = 0.01 * np.pi/180

# ====== ベースディレクトリ ======
BASE_DIR = r"C:\Users\kasa1\YMMT_kadai\Data_0808"

# ====== 解析開始時刻の設定 ======
START_TIMES = {
    "pt5": 0.0,
    "pt6": 0.0,
    "pt1": 25.0,
}

# ====== 最初と最後の接地を除外する個数の設定 ======
# データセットごとに head, tail を上書き可能
CONTACT_TRIM = {
    "default": {"head": 4, "tail": 4},
    # 例: 個別に変えたい場合は以下を編集
     "pt1": {"head": 5, "tail": 4},
    # "pt5": {"head": 4, "tail": 5},
    # "pt6": {"head": 3, "tail": 3},
}

# ====== ファイル設定 ======
hand_files = {
    "pt5": {
        "right": rf"{BASE_DIR}\20250521\IMU\mem\rhand\mem_rhand_pa4_pt5.csv",
        "left":  rf"{BASE_DIR}\20250521\IMU\mem\lhand\mem_lhand_pa4_pt5.csv"
    },
    "pt6": {
        "right": rf"{BASE_DIR}\20250521\IMU\mem\rhand\mem_rhand_pa4_pt6.csv",
        "left":  rf"{BASE_DIR}\20250521\IMU\mem\lhand\mem_lhand_pa4_pt6.csv"
    },
    "pt1": {
        "right": rf"{BASE_DIR}\20250521\IMU\mem\rhand\mem_rhand_pa4_pt1.csv",
        "left":  rf"{BASE_DIR}\20250521\IMU\mem\lhand\mem_lhand_pa4_pt1.csv"
    }
}

foot_files = {
    "pt5": {"file": rf"{BASE_DIR}\20250521\IMU\mem\paw\mem_paw_pa4_pt5.csv", "start_ts": 56062410},
    "pt6": {"file": rf"{BASE_DIR}\20250521\IMU\mem\paw\mem_paw_pa4_pt6.csv", "start_ts": 56355630},
    "pt1": {"file": rf"{BASE_DIR}\20250521\IMU\mem\paw\mem_paw_pa4_pt1.csv", "start_ts": 55484830},
}

# ====== 出力設定 ======
SHOW_BOXPLOT  = True
SHOW_FOOTCHECK = True    # 確認用グラフを出す
SHOW_OVERLAYS = False    # 周期重ね描き

# ====== ベース区間（回転・スケール・ゼロ化の5秒窓。start_time補正後の相対時間で指定） ======
offset_ranges = {k: (0.0, 5.0) for k in START_TIMES.keys()}

# ====== フィルタ ======
def butter_filter(data, cutoff, fs, btype='low', order=4):
    nyq = 0.5 * fs
    norm = cutoff / nyq
    b, a = butter(order, norm, btype=btype)
    return filtfilt(b, a, data)

# ====== データ読み込み（start_timeを原点にシフトしてt>=0のみ残す） ======
def load_imu(file_path, start_ts, start_time):
    col_names = ["type","timestamp","acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]
    df = pd.read_csv(file_path, header=None, names=col_names, engine="python")
    df = df[df["type"]=="ags"].reset_index(drop=True)
    df["abs_sec"] = (df["timestamp"].astype(float) - start_ts) * 0.001
    df["time_sec"] = df["abs_sec"] - float(start_time)
    df = df[df["time_sec"] >= 0].reset_index(drop=True)
    for col in ["acc_x","acc_y","acc_z"]:
        df[col] = df[col].astype(float) * ACC_SCALE
    for col in ["gyr_x","gyr_y","gyr_z"]:
        df[col] = df[col].astype(float) * GYRO_SCALE
    df["acc_norm"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
    df["gyr_norm"] = np.sqrt(df["gyr_x"]**2 + df["gyr_y"]**2 + df["gyr_z"]**2)
    return df

# ====== Rodrigues 回転補助 ======
def _rotation_matrix_from_to(u, v):
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    c = float(np.clip(np.dot(u, v), -1.0, 1.0))
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        cand = np.array([0.0, 1.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        v2 = cand - cand.dot(u) * u
        v2 = v2 / (np.linalg.norm(v2) + 1e-12)
        K = np.array([[0, -v2[2], v2[1]],
                      [v2[2], 0, -v2[0]],
                      [-v2[1], v2[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    angle = np.arccos(c)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

# ====== LRL/RLR用 回転＋9.81スケール＋ゼロ化 ======
def rotate_scale_offset_for_lr(df, offset_range, g_true=9.80665, pre_lpf_hz=2.0, zero_offset=True):
    if offset_range is None or len(df) == 0:
        return df.copy(deep=True)
    df2 = df.copy(deep=True)
    start, end = offset_range
    mask = (df2["time_sec"] >= start) & (df2["time_sec"] <= end)
    if mask.sum() == 0:
        return df2

    dt = np.median(np.diff(df2["time_sec"]))
    fs = 1.0/dt if dt > 0 else 100.0

    A = df2[["acc_x","acc_y","acc_z"]].to_numpy()
    if pre_lpf_hz is not None and pre_lpf_hz > 0:
        A_lp = np.empty_like(A)
        for k in range(3):
            A_lp[:,k] = butter_filter(A[:,k], pre_lpf_hz, fs, 'low')
        g_vec = A_lp[mask.values, :].mean(axis=0)
    else:
        g_vec = A[mask.values, :].mean(axis=0)

    if np.linalg.norm(g_vec) < 1e-12:
        if zero_offset:
            for col in ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]:
                df2[col] = df2[col] - df2.loc[mask, col].mean()
        df2["acc_norm"] = np.sqrt(df2["acc_x"]**2 + df2["acc_y"]**2 + df2["acc_z"]**2)
        df2["gyr_norm"] = np.sqrt(df2["gyr_x"]**2 + df2["gyr_y"]**2 + df2["gyr_z"]**2)
        return df2

    R = _rotation_matrix_from_to(g_vec, np.array([1.0, 0.0, 0.0]))
    A_rot = (R @ A.T).T
    gm = np.mean(A_rot[mask.values, 0])
    s = 1.0 if abs(gm) < 1e-12 else g_true / gm
    A_rs = A_rot * s

    G = df2[["gyr_x","gyr_y","gyr_z"]].to_numpy()
    G_rot = (R @ G.T).T

    df2["acc_x"], df2["acc_y"], df2["acc_z"] = A_rs[:,0], A_rs[:,1], A_rs[:,2]
    df2["gyr_x"], df2["gyr_y"], df2["gyr_z"] = G_rot[:,0], G_rot[:,1], G_rot[:,2]

    if zero_offset:
        for col in ["acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]:
            df2[col] = df2[col] - df2.loc[mask, col].mean()

    df2["acc_norm"] = np.sqrt(df2["acc_x"]**2 + df2["acc_y"]**2 + df2["acc_z"]**2)
    df2["gyr_norm"] = np.sqrt(df2["gyr_x"]**2 + df2["gyr_y"]**2 + df2["gyr_z"]**2)
    return df2

# ====== 接地トリム適用ユーティリティ ======
def apply_contact_trim(peaks, head, tail):
    head = max(0, int(head))
    tail = max(0, int(tail))
    if len(peaks) <= head + tail:
        return np.array([], dtype=int)
    return peaks[head: len(peaks) - tail]

# ====== 歩行周期検出 + 左右判定 + 確認用4段表示 ======
def detect_gait_cycles(df_foot, dt, fs, label="", df_r_hand=None, df_l_hand=None, trim_head=4, trim_tail=4):
    # 2Hz LPF
    df_2Hz = df_foot.copy()
    for col in ["acc_x","acc_y","acc_z"]:
        df_2Hz[col] = butter_filter(df_foot[col], 2, fs)

    # CWT で接地検出
    acc_vert = df_2Hz["acc_x"].values
    acc_vert_integrated = np.cumsum(acc_vert) * dt
    cwt_gaus1, _ = pywt.cwt(acc_vert_integrated, np.arange(1, 50), 'gaus1')
    cwt_signal_1st = cwt_gaus1[9]
    peaks, _ = find_peaks(-cwt_signal_1st, distance=int(0.3*fs), prominence=0.15)

    # 右向き正に反転して pos_y
    acc_y_right = -df_2Hz["acc_y"].values
    vel_y = np.cumsum(acc_y_right)*dt
    pos_y = butter_filter(np.cumsum(vel_y)*dt, 0.5, fs, 'high')

    # 左右ラベル
    labels = []
    search_window = int(1.0 * fs)
    prom = 0.1 * np.std(pos_y)
    for idx in peaks:
        seg = pos_y[idx: idx + search_window]
        if len(seg) < 5:
            labels.append("?")
        else:
            p_pos, _ = find_peaks(seg, prominence=prom)
            p_neg, _ = find_peaks(-seg, prominence=prom)
            cand=[]
            if len(p_pos)>0: cand.append((p_pos[0],"R"))
            if len(p_neg)>0: cand.append((p_neg[0],"L"))
            labels.append(min(cand,key=lambda x:x[0])[1] if cand else "?")

    left_all  = [p for i,p in enumerate(peaks) if labels[i]=='L']
    right_all = [p for i,p in enumerate(peaks) if labels[i]=='R']

    # ここで head, tail の設定に基づいて採用範囲を決定
    used_peaks = apply_contact_trim(peaks, trim_head, trim_tail)
    used_set = set(used_peaks.tolist())
    left_used  = [p for p in left_all if p in used_set]
    right_used = [p for p in right_all if p in used_set]

    # 確認用表示
    if SHOW_FOOTCHECK:
        t = df_foot["time_sec"].values
        fig,axs = plt.subplots(4,1,figsize=(12,10),sharex=True)

        # 上段 CWT
        axs[0].plot(t, cwt_signal_1st, color="deepskyblue", lw=1.4)
        axs[0].scatter(t[peaks], cwt_signal_1st[peaks], c="black", s=28, label="contacts")
        if len(used_peaks)>0:
            axs[0].axvspan(t[used_peaks[0]], t[used_peaks[-1]], color="yellow", alpha=0.25)
        axs[0].set_ylabel("CWT coeff."); axs[0].grid(True); axs[0].legend(loc="upper right")

        # 二段目 pos_y と L R
        axs[1].plot(t, pos_y, color="forestgreen", lw=1.2)
        axs[1].scatter(t[left_all], pos_y[left_all], c="blue", s=30, label="L")
        axs[1].scatter(t[right_all], pos_y[right_all], c="red",  s=30, label="R")
        if len(used_peaks)>0:
            axs[1].axvspan(t[used_peaks[0]], t[used_peaks[-1]], color="yellow", alpha=0.25)
        axs[1].set_ylabel("pos_y"); axs[1].grid(True); axs[1].legend(loc="upper right")

        # 三段目 右手 acc_norm
        if df_r_hand is not None and len(df_r_hand)>0:
            tr = df_r_hand["time_sec"].values
            axs[2].plot(tr, df_r_hand["acc_norm"].values, color="royalblue", lw=1.0, label="Right hand acc_norm")
            if len(used_peaks)>0:
                axs[2].axvspan(t[used_peaks[0]], t[used_peaks[-1]], color="yellow", alpha=0.25)
            axs[2].set_ylabel("RH acc_norm"); axs[2].grid(True); axs[2].legend(loc="upper right")

        # 四段目 左手 acc_norm
        if df_l_hand is not None and len(df_l_hand)>0:
            tl = df_l_hand["time_sec"].values
            axs[3].plot(tl, df_l_hand["acc_norm"].values, color="firebrick", lw=1.0, label="Left hand acc_norm")
            if len(used_peaks)>0:
                axs[3].axvspan(t[used_peaks[0]], t[used_peaks[-1]], color="yellow", alpha=0.25)
            axs[3].set_ylabel("LH acc_norm"); axs[3].grid(True); axs[3].legend(loc="upper right")
            axs[3].set_xlabel("Time (s)")

        plt.suptitle(f"{label} - Foot Contact Detection and Hand acc_norm in Used Window\n(trim head={trim_head}, tail={trim_tail})")
        plt.tight_layout()
        plt.show()

    return left_used, right_used, used_peaks

# ====== 区切り抽出（LRL/RLR用 平均0化） ======
def extract_cycles(df, left_idx, right_idx, col, mode="LRL"):
    segs=[]
    all_idx = sorted(left_idx + right_idx)
    if len(all_idx) < 2 or len(df) == 0:
        return segs
    if mode=="LRL":
        start_idx = min(left_idx) if len(left_idx)>0 else all_idx[0]
    else:
        start_idx = min(right_idx) if len(right_idx)>0 else all_idx[0]

    seq = sorted(all_idx)
    try:
        start_pos = seq.index(start_idx)
    except ValueError:
        start_pos = 0

    for i in range(start_pos, len(seq)-2, 2):
        seg = df[col].iloc[seq[i]:seq[i+2]].values
        seg = seg - np.mean(seg)
        segs.append(seg)
    return segs

# --- acc_norm を周期ごとに xyz から再構成（各軸0平均→norm→normも0平均） ---
def recompute_acc_norm_cycles_from_xyz(df, left_idx, right_idx, mode="LRL"):
    segs_x = extract_cycles(df, left_idx, right_idx, "acc_x", mode=mode)
    segs_y = extract_cycles(df, left_idx, right_idx, "acc_y", mode=mode)
    segs_z = extract_cycles(df, left_idx, right_idx, "acc_z", mode=mode)
    n = min(len(segs_x), len(segs_y), len(segs_z))
    out = []
    for i in range(n):
        L = min(len(segs_x[i]), len(segs_y[i]), len(segs_z[i]))
        if L < 5: 
            continue
        nx = segs_x[i][:L]
        ny = segs_y[i][:L]
        nz = segs_z[i][:L]
        norm_seg = np.sqrt(nx**2 + ny**2 + nz**2)
        norm_seg = norm_seg - np.mean(norm_seg)
        out.append(norm_seg)
    return out

# ====== 指標計算（np.trapezoidでDeprecationWarning回避） ======
def compute_ldlj_acc(signal, dt, fs, cutoff=20):
    if len(signal)<5: return np.nan
    sig_filt = butter_filter(signal, cutoff, fs)
    jerk = np.gradient(sig_filt, dt)
    vel = np.cumsum(sig_filt)*dt
    peak_v = np.max(np.abs(vel))+1e-8
    T = len(sig_filt)*dt
    val = (T**3/peak_v**2)*np.trapezoid(jerk**2, dx=dt)
    return -np.log(val+1e-12)

def compute_spal_acc(signal, dt, fs, cutoff=20):
    if len(signal)<5: return np.nan
    sig_filt = butter_filter(signal, cutoff, fs)
    vel = np.cumsum(sig_filt)*dt
    v_peak = np.max(np.abs(vel))+1e-8
    v_hat = vel/v_peak
    T = len(sig_filt)*dt
    dvhat_dt = np.gradient(v_hat, dt)
    integrand = np.sqrt((1.0/T)**2+dvhat_dt**2)
    val = np.trapezoid(integrand, dx=dt)
    return -np.log(val+1e-12)

def compute_ldlj_gyr(signal, dt, fs, cutoff=20):
    if len(signal)<5: return np.nan
    sig_filt = butter_filter(signal, cutoff, fs)
    jerk = np.gradient(np.gradient(sig_filt,dt),dt)
    peak_w = np.max(np.abs(sig_filt))+1e-8
    T = len(sig_filt)*dt
    val = (T**3/peak_w**2)*np.trapezoid(jerk**2, dx=dt)
    return -np.log(val+1e-12)

def compute_spal_gyr(signal, dt, fs, cutoff=20):
    if len(signal)<5: return np.nan
    sig_filt = butter_filter(signal, cutoff, fs)
    w_peak = np.max(np.abs(sig_filt))+1e-8
    w_hat = sig_filt/w_peak
    T = len(sig_filt)*dt
    dw_dt = np.gradient(w_hat, dt)
    integrand = np.sqrt((1.0/T)**2+dw_dt**2)
    val = np.trapezoid(integrand, dx=dt)
    return -np.log(val+1e-12)

# ====== 解析（LRL/RLR） ======
def analyze_hand_lr(df_lr, left_idx, right_idx, dt, fs):
    results={"LDLJ":{}, "SPAL":{}}

    segsL_accn = recompute_acc_norm_cycles_from_xyz(df_lr, left_idx, right_idx, mode="LRL")
    segsR_accn = recompute_acc_norm_cycles_from_xyz(df_lr, left_idx, right_idx, mode="RLR")
    segsL_gyrn = extract_cycles(df_lr, left_idx, right_idx, "gyr_norm", mode="LRL")
    segsR_gyrn = extract_cycles(df_lr, left_idx, right_idx, "gyr_norm", mode="RLR")

    results["LDLJ"]["acc_norm"] = {
        "LRL": [compute_ldlj_acc(seg,dt,fs) for seg in segsL_accn],
        "RLR": [compute_ldlj_acc(seg,dt,fs) for seg in segsR_accn],
    }
    results["SPAL"]["acc_norm"] = {
        "LRL": [compute_spal_acc(seg,dt,fs) for seg in segsL_accn],
        "RLR": [compute_spal_acc(seg,dt,fs) for seg in segsR_accn],
    }

    results["LDLJ"]["gyr_norm"] = {
        "LRL": [compute_ldlj_gyr(seg,dt,fs) for seg in segsL_gyrn],
        "RLR": [compute_ldlj_gyr(seg,dt,fs) for seg in segsR_gyrn],
    }
    results["SPAL"]["gyr_norm"] = {
        "LRL": [compute_spal_gyr(seg,dt,fs) for seg in segsL_gyrn],
        "RLR": [compute_spal_gyr(seg,dt,fs) for seg in segsR_gyrn],
    }

    return results

# ====== 周期オーバーレイ（任意） ======
def plot_cycle_overlays(plot_payload):
    if not SHOW_OVERLAYS:
        return
    colors = {"LRL":"royalblue", "RLR":"firebrick"}
    conds = ["LRL","RLR"]
    dataset_order = [d for d in START_TIMES.keys() if d in plot_payload]

    for side in ["Right","Left"]:
        for sig in ["acc","gyr"]:
            for axis in [f"{sig}_x", f"{sig}_y", f"{sig}_z", f"{sig}_norm"]:
                fig, axs = plt.subplots(len(dataset_order), len(conds),
                                        figsize=(12, 9), sharex=False)
                for r, label in enumerate(dataset_order):
                    item = plot_payload[label][side]
                    df_lr     = item["df_lr"]
                    left_idx  = item["left_idx"]
                    right_idx = item["right_idx"]

                    if axis == "acc_norm":
                        segsL = recompute_acc_norm_cycles_from_xyz(df_lr, left_idx, right_idx, mode="LRL")
                        segsR = recompute_acc_norm_cycles_from_xyz(df_lr, left_idx, right_idx, mode="RLR")
                    else:
                        segsL = extract_cycles(df_lr, left_idx, right_idx, axis, mode="LRL")
                        segsR = extract_cycles(df_lr, left_idx, right_idx, axis, mode="RLR")

                    all_vals=[]
                    for arr in segsL+segsR:
                        if len(arr)>0:
                            all_vals.extend(arr)
                    if len(all_vals)==0:
                        y_min, y_max = -1.0, 1.0
                    else:
                        y_min, y_max = float(np.min(all_vals)), float(np.max(all_vals))
                        if y_min == y_max:
                            y_min -= 1e-6; y_max += 1e-6

                    for c, cond in enumerate(conds):
                        ax = axs[r, c]
                        segs = segsL if cond=="LRL" else segsR
                        for seg in segs:
                            x = np.linspace(0,100,len(seg))
                            ax.plot(x, seg, color=colors[cond], alpha=0.35, lw=1.2)
                        if r==0:
                            ax.set_title(cond, fontsize=11)
                        if c==0:
                            ax.set_ylabel(label, fontsize=11)
                        ax.set_ylim(y_min, y_max)
                        ax.grid(True, alpha=0.3)

                fig.suptitle(f"Cycle overlays: {side} - {axis}", fontsize=13)
                plt.tight_layout()
                plt.show()

# ====== 箱ひげ（LRL/RLR） ======
def run_and_plot_grouped(all_results):
    if not SHOW_BOXPLOT:
        return
    from matplotlib.patches import Patch

    datasets = [d for d in START_TIMES.keys() if d in all_results]
    colors = {"pt5":"skyblue", "pt6":"blue", "pt1":"orange"}
    metrics = ["SPAL","LDLJ"]
    conds = ["LRL","RLR"]

    base_L = 1.0
    base_R = 3.0
    width = 0.18 if len(datasets) >= 3 else 0.25
    offsets = [ (j - (len(datasets)-1)/2)* (width*1.05) for j in range(len(datasets)) ]

    for side in ["Right","Left"]:
        for axis in ["acc_norm","gyr_norm"]:
            fig = plt.figure(figsize=(12,8))
            for i, metric in enumerate(metrics, 1):
                ax = fig.add_subplot(2,1,i)

                # LRL
                data_L = [all_results[d][side][metric][axis]["LRL"] for d in datasets]
                pos_L  = [base_L + off for off in offsets]
                bpL = ax.boxplot(data_L, positions=pos_L, widths=width,
                                 patch_artist=True, showmeans=True)
                for patch, d in zip(bpL["boxes"], datasets):
                    patch.set_facecolor(colors.get(d,"gray"))

                # RLR
                data_R = [all_results[d][side][metric][axis]["RLR"] for d in datasets]
                pos_R  = [base_R + off for off in offsets]
                bpR = ax.boxplot(data_R, positions=pos_R, widths=width,
                                 patch_artist=True, showmeans=True)
                for patch, d in zip(bpR["boxes"], datasets):
                    patch.set_facecolor(colors.get(d,"gray"))

                ax.set_xticks([base_L, base_R])
                ax.set_xticklabels(conds, fontsize=14)
                ax.set_xlim(base_L-1.0, base_R+1.0)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.set_ylabel(metric)

                handles = [Patch(facecolor=colors.get(d,"gray"), label=d) for d in datasets]
                ax.legend(handles=handles, loc="best")

            fig.suptitle(f"{side} - {axis}")
            plt.tight_layout()
            plt.show()

# ====== メイン ======
def main():
    all_results={}
    plot_payload={}
    for label,files in hand_files.items():
        if label not in foot_files:
            continue

        start_time = START_TIMES.get(label, 0.0)

        # 足
        foot_info = foot_files[label]
        df_foot = load_imu(foot_info["file"], foot_info["start_ts"], start_time)
        if len(df_foot) == 0:
            continue
        dt = np.median(np.diff(df_foot["time_sec"])); fs = 1.0/dt

        # 手
        df_r_raw = load_imu(files["right"], foot_info["start_ts"], start_time)
        df_l_raw = load_imu(files["left"],  foot_info["start_ts"], start_time)

        # LRL/RLR 用 回転＋9.81 スケール＋ゼロ化
        base_range = offset_ranges.get(label, (0.0, 5.0))
        df_r_lr = rotate_scale_offset_for_lr(df_r_raw, base_range, g_true=9.80665, pre_lpf_hz=2.0, zero_offset=True)
        df_l_lr = rotate_scale_offset_for_lr(df_l_raw, base_range, g_true=9.80665, pre_lpf_hz=2.0, zero_offset=True)

        # トリム設定の取得
        trim_conf = CONTACT_TRIM.get(label, CONTACT_TRIM.get("default", {"head":4,"tail":4}))
        head = int(trim_conf.get("head", 4))
        tail = int(trim_conf.get("tail", 4))

        # 足で接地検出
        left_idx, right_idx, used_peaks = detect_gait_cycles(
            df_foot, dt, fs, label=label, df_r_hand=df_r_lr, df_l_hand=df_l_lr,
            trim_head=head, trim_tail=tail
        )

        # 解析
        all_results[label] = {
            "Right": analyze_hand_lr(df_r_lr, left_idx, right_idx, dt, fs),
            "Left":  analyze_hand_lr(df_l_lr, left_idx, right_idx, dt, fs)
        }

        # オーバーレイ描画用
        plot_payload[label] = {
            "Right": {"df_lr": df_r_lr, "left_idx": left_idx, "right_idx": right_idx},
            "Left":  {"df_lr": df_l_lr, "left_idx": left_idx, "right_idx": right_idx}
        }

    run_and_plot_grouped(all_results)
    plot_cycle_overlays(plot_payload)

if __name__ == "__main__":
    main()
