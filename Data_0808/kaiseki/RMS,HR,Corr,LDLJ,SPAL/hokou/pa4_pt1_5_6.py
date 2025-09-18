import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
import pywt, os

# ====== 設定 ======
ACC_SCALE = 9.80665e-4
BASE_DIR = r"C:\Users\kasa1\YMMT_kadai\Data_0808"

# ====== 解析対象データ ======
DATASETS = [
    {"name": "saisyou", "file": f"{BASE_DIR}/20250521/IMU/mem/paw/mem_paw_pa4_saisyou.csv",
     "start_ts": 55156490, "start_time": 0.0, "speed": 0.628},
    {"name": "pt5", "file": f"{BASE_DIR}/20250521/IMU/mem/paw/mem_paw_pa4_pt5.csv",
     "start_ts": 56062410, "start_time": 0.0, "speed": 0.763},
    {"name": "pt6", "file": f"{BASE_DIR}/20250521/IMU/mem/paw/mem_paw_pa4_pt6.csv",
     "start_ts": 56355630, "start_time": 0.0, "speed": 0.807},
    {"name": "pt1", "file": f"{BASE_DIR}/20250521/IMU/mem/paw/mem_paw_pa4_pt1.csv",
     "start_ts": 55484830, "start_time": 25.0, "speed": 0.657},
]

# ====== 解析対象の選択 ======
USE_DATASETS = ["saisyou", "pt5", "pt6", "pt1"]

# === オプション ===
SHOW_CONFIRM = True        # 左右検出の確認用グラフを描く（★各データセットにつき1回のみ表示）
SHOW_BOXPLOT = True        # 箱ひげ図を描く
SHOW_OVERLAYS = True       # 周期波形の重ね合わせを描く

# ★ 新仕様スイッチ：LRL/RLR の norm を「周期ごと再計算＋norm も周期平均を引く」方式にする
USE_CYCLE_RECALC_NORM = True
SUBTRACT_NORM_MEAN = True

# ====== 追加の窓設定 ======
# 1) データセットごとの解析開始時刻を上書きできる窓
START_TIME_OVERRIDE = {
    # 例) "pt1": 25.0
}

# 2) 最初と最後の接地を何個除外するかの窓（左右判定＆周期作成の採用範囲）
CONTACT_TRIM = {
    "default": {"head": 4, "tail": 4},
    # 例) "pt1": {"head": 6, "tail": 6}
}

# ====== フィルタ ======
def butter_filter(data, cutoff, fs, btype='low', order=4):
    nyq = 0.5 * fs
    norm = cutoff / nyq
    b, a = butter(order, norm, btype=btype)
    return filtfilt(b, a, data)

# ====== 周期抽出 ======
def get_LRL_cycles_time(left_indices, right_indices, time_array):
    return [(time_array.iloc[l_start], time_array.iloc[l_end])
            for i, l_start in enumerate(left_indices[:-1])
            if any((l_start < r < left_indices[i+1]) for r in right_indices)
            for l_end in [left_indices[i+1]]]

def get_RLR_cycles_time(right_indices, left_indices, time_array):
    return [(time_array.iloc[r_start], time_array.iloc[r_end])
            for i, r_start in enumerate(right_indices[:-1])
            if any((r_start < l < right_indices[i+1]) for l in left_indices)
            for r_end in [right_indices[i+1]]]

# ====== 周期ごと切り出し ======
def extract_cycles_time(data, cycles, col, subtract_mean=True, npts=100):
    resampled_cycles = []
    for (t_start, t_end) in cycles:
        seg = data.loc[(data["time_sec"] >= t_start) & (data["time_sec"] < t_end), col].values
        if len(seg) < 5:
            continue
        if subtract_mean:
            seg = seg - np.mean(seg)
        t = np.linspace(0, 1, len(seg))
        seg_resampled = np.interp(np.linspace(0, 1, npts), t, seg)
        resampled_cycles.append(seg_resampled)
    return np.array(resampled_cycles)

# ====== 指標計算 ======
def mean_rms(cycles, speed):
    if len(cycles) == 0:
        return np.nan
    return [np.sqrt(np.mean(c**2)) / (speed**2) for c in cycles]

def autocorr_one_cycle(cycles):
    if len(cycles) < 2:
        return []
    vals = []
    for i in range(len(cycles) - 1):
        r, _ = pearsonr(cycles[i], cycles[i+1])
        vals.append(r)
    return vals

def harmonic_ratio_cycles_time(df, cycles, col, n_harmonics=20, axis_type="ap"):
    n = 10
    hr_vals = []
    for (t_start, t_end) in cycles:
        sig = df.loc[(df["time_sec"] >= t_start) & (df["time_sec"] < t_end), col].values
        if len(sig) < 10:
            continue
        sig = sig - np.mean(sig)
        fft_vals = np.fft.rfft(sig)
        amps = np.abs(fft_vals)
        odd  = amps[1:2*n:2]
        even = amps[2:2*n+1:2]
        if axis_type == "ml":
            den = np.sum(even)
            if den > 0:
                hr_vals.append(np.sum(odd) / den)
        else:
            den = np.sum(odd)
            if den > 0:
                hr_vals.append(np.sum(even) / den)
    return hr_vals

# ====== Rodrigues 回転 + 9.81 スケール（LRL/RLR 用） ======
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

def rotate_scale_to_gravity(df, baseline_start, baseline_end, g_true=9.80665, pre_lpf_hz=None):
    mask = (df["time_sec"] >= baseline_start) & (df["time_sec"] <= baseline_end)
    A = df[["acc_x","acc_y","acc_z"]].to_numpy()
    dt = np.median(np.diff(df["time_sec"]))
    fs = 1.0/dt if dt > 0 else 100.0

    if pre_lpf_hz is not None and pre_lpf_hz > 0:
        A_lp = np.empty_like(A)
        for k in range(3):
            A_lp[:,k] = butter_filter(A[:,k], pre_lpf_hz, fs, 'low')
        g_vec = A_lp[mask.values, :].mean(axis=0)
    else:
        g_vec = A[mask.values, :].mean(axis=0)

    if np.linalg.norm(g_vec) < 1e-12:
        df_rs = df.copy()
        df_rs["acc_norm"] = np.sqrt(df_rs["acc_x"]**2 + df_rs["acc_y"]**2 + df_rs["acc_z"]**2)
        return df_rs, np.eye(3), 1.0, np.zeros(3)

    R = _rotation_matrix_from_to(g_vec, np.array([1.0, 0.0, 0.0]))
    A_rot = (R @ A.T).T
    gm = np.mean(A_rot[mask.values, 0])
    s = 1.0 if abs(gm) < 1e-12 else g_true / gm
    A_rs = A_rot * s

    df_rs = df.copy()
    df_rs["acc_x"] = A_rs[:,0]
    df_rs["acc_y"] = A_rs[:,1]
    df_rs["acc_z"] = A_rs[:,2]
    df_rs["acc_norm"] = np.sqrt(df_rs["acc_x"]**2 + df_rs["acc_y"]**2 + df_rs["acc_z"]**2)
    return df_rs, R, s, df_rs.loc[mask, ["acc_x","acc_y","acc_z"]].mean().to_numpy()

def rotate_to_gravity_then_zero(df, baseline_start, baseline_end, g_true=9.80665, pre_lpf_hz=2.0):
    df_rs, _, _, _ = rotate_scale_to_gravity(df, baseline_start, baseline_end, g_true=g_true, pre_lpf_hz=pre_lpf_hz)
    mask = (df_rs["time_sec"] >= baseline_start) & (df_rs["time_sec"] <= baseline_end)
    m = df_rs.loc[mask, ["acc_x","acc_y","acc_z"]].mean().to_numpy()
    df_rs["acc_x"] -= m[0]
    df_rs["acc_y"] -= m[1]
    df_rs["acc_z"] -= m[2]
    df_rs["acc_norm"] = np.sqrt(df_rs["acc_x"]**2 + df_rs["acc_y"]**2 + df_rs["acc_z"]**2)
    return df_rs

def rotate_to_gravity_only(df, baseline_start, baseline_end, g_true=9.80665, pre_lpf_hz=2.0):
    df_rs, _, _, _ = rotate_scale_to_gravity(df, baseline_start, baseline_end, g_true=g_true, pre_lpf_hz=pre_lpf_hz)
    return df_rs

# ====== 接地トリムユーティリティ ======
def apply_contact_trim(peaks, head, tail):
    head = max(0, int(head))
    tail = max(0, int(tail))
    if len(peaks) <= head + tail:
        return np.array([], dtype=int)
    return peaks[head: len(peaks) - tail]

# ====== 確認用グラフ（上段：左右検出 pos_y, 下段：acc_norm） ======
def plot_confirm_panels(dfp, pos_y, peaks, left_all, right_all, used_peaks, title_suffix=""):
    if len(dfp) == 0:
        return
    t = dfp["time_sec"].values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # 上段：pos_y と L/R マーカー + 採用範囲
    ax1.plot(t, pos_y, lw=1.2, label="pos_y")
    ax1.scatter(t[left_all],  pos_y[left_all],  c="blue",  s=28, label="L")
    ax1.scatter(t[right_all], pos_y[right_all], c="red",   s=28, label="R")
    if len(used_peaks) > 0:
        ax1.axvspan(t[used_peaks[0]], t[used_peaks[-1]], color="yellow", alpha=0.25, label="used window")
    ax1.set_ylabel("pos_y")
    ax1.grid(True, alpha=0.4)
    ax1.legend(loc="upper right")

    # 下段：acc_norm のみ + 採用範囲
    if "acc_norm" not in dfp.columns:
        dfp["acc_norm"] = np.sqrt(dfp["acc_x"]**2 + dfp["acc_y"]**2 + dfp["acc_z"]**2)
    ax2.plot(t, dfp["acc_norm"].values, lw=1.0, label="acc_norm")
    if len(used_peaks) > 0:
        ax2.axvspan(t[used_peaks[0]], t[used_peaks[-1]], color="yellow", alpha=0.25)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("acc_norm")
    ax2.grid(True, alpha=0.4)
    ax2.legend(loc="upper right")

    plt.suptitle(f"Foot contact & acc_norm check {title_suffix}")
    plt.tight_layout()
    plt.show()

# ====== データ処理 ======
def process_dataset(ds, mode="LRL", subtract_mean=True, moe_nilssen=True, npts=100, show_confirm=False,
                    use_cycle_recalc_norm=False, subtract_norm_mean=False):
    # --- start_time を上書き可能にする & その時刻以降のみを解析 ---
    start_time_eff = START_TIME_OVERRIDE.get(ds["name"], ds["start_time"])

    col_names=["type","timestamp","acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"]
    df_raw = pd.read_csv(ds["file"], header=None, names=col_names, engine="python")
    df_raw = df_raw[df_raw["type"]=="ags"].reset_index(drop=True)

    # 絶対秒 → 解析原点へシフト → t>=0のみ残す
    abs_sec = (df_raw["timestamp"].astype(float) - ds["start_ts"]) * 0.001
    time_sec = abs_sec - float(start_time_eff)
    keep = time_sec >= 0.0
    df_raw = df_raw.loc[keep].reset_index(drop=True)
    df_raw["time_sec"] = time_sec.loc[keep].values

    for col in ["acc_x","acc_y","acc_z"]:
        df_raw[col] = df_raw[col].astype(float)*ACC_SCALE

    # 基準窓は「シフト後の 0〜2s」
    baseline_start, baseline_end = 0.0, 2.0

    # ====== 前処理ポリシー ======
    if moe_nilssen:
        # LRL/RLR：回転+9.81スケール＋基準窓ゼロ化
        df_rot_zero = rotate_to_gravity_then_zero(df_raw, baseline_start, baseline_end, g_true=9.80665, pre_lpf_hz=2.0)
        df_rot_only = rotate_to_gravity_only(df_raw,  baseline_start, baseline_end, g_true=9.80665, pre_lpf_hz=2.0)
        dfp = df_rot_zero
    else:
        # Before：回転しない 9.81もしない 基準窓の各軸平均のみ引く
        dfp = df_raw.copy()
        mask=(dfp["time_sec"]>=baseline_start)&(dfp["time_sec"]<=baseline_end)
        offset = dfp.loc[mask, ["acc_x","acc_y","acc_z"]].mean().to_numpy()
        dfp["acc_x"] -= offset[0]
        dfp["acc_y"] -= offset[1]
        dfp["acc_z"] -= offset[2]
        dfp["acc_norm"]=np.sqrt(dfp["acc_x"]**2+dfp["acc_y"]**2+dfp["acc_z"]**2)
        dfp["acc_norm"] -= dfp.loc[mask,"acc_norm"].mean()
        df_rot_only = dfp.copy()

    # ====== 接地検出 & 左右判定（dfp を使用） ======
    if len(dfp) < 5:
        return {"Dataset": ds["name"], "cycles_time": [], "pos_y": [], "peaks": np.array([]),
                "labels": [], "used_peaks": np.array([])}

    dt = np.median(np.diff(dfp["time_sec"])); fs = 1.0/dt
    acc_vert = butter_filter(dfp["acc_x"].values, 2, fs)
    acc_vert_integrated = np.cumsum(acc_vert) * dt
    cwt_gaus1,_ = pywt.cwt(acc_vert_integrated, np.arange(1,50), 'gaus1')
    cwt_signal = cwt_gaus1[9]
    peaks,_ = find_peaks(-cwt_signal, distance=int(0.3*fs), prominence=0.15)

    acc_y_right = -dfp["acc_y"].values
    acc_y_filt = butter_filter(acc_y_right, 2, fs)
    vel_y = np.cumsum(acc_y_filt)*dt
    pos_y = butter_filter(np.cumsum(vel_y)*dt, 0.5, fs, 'high')

    labels=[]
    prom = 0.1*np.std(pos_y)
    win = int(1.0*fs)
    for idx in peaks:
        seg = pos_y[idx: idx+win]
        if len(seg)<5:
            labels.append("?"); continue
        ppos,_ = find_peaks(seg, prominence=prom)
        pneg,_ = find_peaks(-seg, prominence=prom)
        cand=[]
        if len(ppos)>0: cand.append((ppos[0],"R"))
        if len(pneg)>0: cand.append((pneg[0],"L"))
        labels.append(min(cand, key=lambda x:x[0])[1] if cand else "?")

    left_all  = [p for i,p in enumerate(peaks) if labels[i]=='L']
    right_all = [p for i,p in enumerate(peaks) if labels[i]=='R']

    # === データセットごとの head/tail トリムを適用 ===
    trim_conf = CONTACT_TRIM.get(ds["name"], CONTACT_TRIM.get("default", {"head":4,"tail":4}))
    used_peaks = apply_contact_trim(peaks, trim_conf.get("head", 4), trim_conf.get("tail", 4))
    used_set = set(used_peaks.tolist())
    left_used  = [p for p in left_all  if p in used_set]
    right_used = [p for p in right_all if p in used_set]

    # ====== 確認用グラフ（★各データセットにつき1回だけ呼び出す想定） ======
    if show_confirm:
        suffix = f"[{ds['name']}] start={start_time_eff:.2f}s, trim(head={trim_conf.get('head',4)}, tail={trim_conf.get('tail',4)})"
        plot_confirm_panels(dfp, pos_y, peaks, left_all, right_all, used_peaks, title_suffix=suffix)

    # ====== 周期作成 ======
    if mode=="LRL":
        cycles = get_LRL_cycles_time(left_used, right_used, dfp["time_sec"])
    elif mode=="RLR":
        cycles = get_RLR_cycles_time(right_used, left_used, dfp["time_sec"])
    else:
        cycles = (get_LRL_cycles_time(left_used, right_used, dfp["time_sec"])
                  + get_RLR_cycles_time(right_used, left_used, dfp["time_sec"]))

    # ====== 出力格納 ======
    res = {
        "Dataset": ds["name"],
        "cycles_time": cycles,
        "pos_y": pos_y,
        "peaks": peaks,
        "labels": labels,
        "used_peaks": used_peaks,
    }

    # --- x,y,z は dfp から ---
    for axis, ax_type in zip(["acc_x","acc_y","acc_z"], ["vert","ml","ap"]):
        cycles_data = extract_cycles_time(dfp, cycles, axis,
                                          subtract_mean=subtract_mean, npts=npts)
        res[f"CYC_DATA_{axis}"] = cycles_data
        res[f"RMS_{axis}"] = mean_rms(cycles_data, ds["speed"])
        res[f"Corr_{axis}"] = autocorr_one_cycle(cycles_data)
        res[f"HR_{axis}"] = harmonic_ratio_cycles_time(dfp, cycles, axis,
                                                       axis_type=ax_type, n_harmonics=20)

    # --- norm の作り方を分岐 ---
    if moe_nilssen and use_cycle_recalc_norm and (mode in ["LRL","RLR"]):
        cyc_x = extract_cycles_time(df_rot_only, cycles, "acc_x", subtract_mean=True,  npts=npts)
        cyc_y = extract_cycles_time(df_rot_only, cycles, "acc_y", subtract_mean=True,  npts=npts)
        cyc_z = extract_cycles_time(df_rot_only, cycles, "acc_z", subtract_mean=True,  npts=npts)

        ncyc = min(len(cyc_x), len(cyc_y), len(cyc_z))
        cyc_norm = []
        mean_norm_each = []
        for i in range(ncyc):
            seg = np.sqrt(cyc_x[i]**2 + cyc_y[i]**2 + cyc_z[i]**2)
            if subtract_norm_mean:
                seg = seg - np.mean(seg)
            cyc_norm.append(seg)
            mean_norm_each.append(float(np.mean(seg)))
        cyc_norm = np.asarray(cyc_norm, dtype=float)

        res["CYC_DATA_acc_norm"] = cyc_norm
        res["RMS_acc_norm"] = mean_rms(cyc_norm, ds["speed"])
        res["Corr_acc_norm"] = autocorr_one_cycle(cyc_norm)
        res["MEAN_acc_norm_per_cycle"] = mean_norm_each
    else:
        if "acc_norm" not in dfp.columns:
            dfp["acc_norm"] = np.sqrt(dfp["acc_x"]**2 + dfp["acc_y"]**2 + dfp["acc_z"]**2)
        cyc_n = extract_cycles_time(dfp, cycles, "acc_norm",
                                    subtract_mean=subtract_mean, npts=npts)
        res["CYC_DATA_acc_norm"] = cyc_n
        res["RMS_acc_norm"] = mean_rms(cyc_n, ds["speed"])
        res["Corr_acc_norm"] = autocorr_one_cycle(cyc_n)

    return res

# ====== 実行 ======
conditions = {
    "LRL":    {"mode":"LRL","subtract_mean":True,"moe_nilssen":True},
    "RLR":    {"mode":"RLR","subtract_mean":True,"moe_nilssen":True},
    "Before": {"mode":"ALL","subtract_mean":False,"moe_nilssen":False}  # Before は回転しない
}

# ★ 確認用グラフは各データセットにつき1回のみ（ここでは LRL の時だけ True にする）
results_dict = {k:[] for k in conditions}
for ds in DATASETS:
    if ds["name"] not in USE_DATASETS:
        continue
    for cond, opts in conditions.items():
        show_once = SHOW_CONFIRM and (cond == "LRL")
        res = process_dataset(
            ds,
            use_cycle_recalc_norm=USE_CYCLE_RECALC_NORM,
            subtract_norm_mean=SUBTRACT_NORM_MEAN,
            **opts,
            show_confirm=show_once
        )
        res["Condition"] = cond
        results_dict[cond].append(res)

df_dict = {k: pd.DataFrame(v).set_index("Dataset") for k,v in results_dict.items()}

# ====== 箱ひげ図（平均値 ▲ も表示） ======
if SHOW_BOXPLOT:
    metrics_groups = {
        "RMS":  [c for c in df_dict["LRL"].columns if c.startswith("RMS")],
        "Corr": [c for c in df_dict["LRL"].columns if c.startswith("Corr")],
        "HR":   [c for c in df_dict["LRL"].columns if c.startswith("HR")]
    }
    for metric, cols in metrics_groups.items():
        fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        n_datasets = len(USE_DATASETS)

        for row_idx, (cond, df) in enumerate(df_dict.items()):
            ax = axs[row_idx]
            x_labels = ["x", "y", "z"] + ([] if metric == "HR" else ["norm"])
            x = np.arange(len(x_labels))
            width = 0.15

            for j, ds_name in enumerate(USE_DATASETS):
                vals = [df.loc[ds_name, c] if isinstance(df.loc[ds_name, c], (list, np.ndarray)) else [] for c in cols]
                for g_idx, group_vals in enumerate(vals):
                    if metric == "Corr" and cond == "Before" and g_idx == 1:
                        continue
                    if isinstance(group_vals, (list, np.ndarray)) and len(group_vals) > 0:

                        display_j = j
                        color_j = j

                        colname = cols[g_idx] if g_idx < len(cols) else ""
                        if (metric == "RMS" and cond == "Before"
                            and (colname.endswith("acc_z") or colname.endswith("_z"))):
                            if "pt5" in USE_DATASETS and "pt6" in USE_DATASETS:
                                if ds_name == "pt5":
                                    display_j = USE_DATASETS.index("pt6")
                                    color_j   = USE_DATASETS.index("pt6")
                                elif ds_name == "pt6":
                                    display_j = USE_DATASETS.index("pt5")
                                    color_j   = USE_DATASETS.index("pt5")

                        pos = x[g_idx] + display_j * width
                        ax.boxplot(group_vals,
                                   positions=[pos],
                                   widths=width,
                                   patch_artist=True,
                                   boxprops=dict(facecolor=f"C{color_j}"),
                                   medianprops=dict(color="black"))

                        mu = float(np.mean(group_vals))
                        ax.scatter([pos], [mu], marker="^", s=70, color=f"C{color_j}", edgecolors="k", zorder=3)

            centers = x + width * (n_datasets - 1) / 2
            for b in np.arange(len(x_labels) - 1) + 0.5 + width * (n_datasets - 1) / 2:
                ax.axvline(b, linestyle="--", color="gray", alpha=0.6)
            ax.set_title(f"{metric} - {cond}")
            ax.grid(axis="y")
            ax.set_xticks(centers)
            ax.set_xticklabels(x_labels)

            if metric == "RMS":
                ax.set_ylim(0.5, 5.5)
            elif metric == "Corr":
                ax.set_ylim(0.4, 1.0)
            elif metric == "HR":
                ax.set_ylim(0.7, 2.5)

        handles = [mpatches.Patch(facecolor=f"C{j}", label=USE_DATASETS[j]) for j in range(n_datasets)]
        axs[0].legend(handles=handles, loc="upper right")
        plt.tight_layout()
        plt.show()

# ====== 周期波形の重ね合わせ（各軸） ======
if SHOW_OVERLAYS:
    colors = {"Before": "black", "LRL": "royalblue", "RLR": "firebrick"}
    cond_order = ["Before", "LRL", "RLR"]
    axes_list = ["acc_x","acc_y","acc_z","acc_norm"]

    for axis in axes_list:
        fig, axs = plt.subplots(len(USE_DATASETS), len(cond_order),
                                figsize=(15, 9), sharex=True)
        if len(USE_DATASETS) == 1:
            axs = np.array([axs])

        for r, ds_name in enumerate(USE_DATASETS):
            for c, cond in enumerate(cond_order):
                ax = axs[r, c]
                cyc = df_dict[cond].loc[ds_name, f"CYC_DATA_{axis}"] \
                      if f"CYC_DATA_{axis}" in df_dict[cond].columns else np.empty((0, 100))
                cyc = np.asarray(cyc, dtype=float)
                for k in range(cyc.shape[0]):
                    x = np.linspace(0, 100, cyc.shape[1])
                    ax.plot(x, cyc[k], color=colors[cond], alpha=0.35, lw=1.2)
                if r == 0:
                    ax.set_title(cond, fontsize=12)
                if c == 0:
                    ax.set_ylabel(ds_name, fontsize=12)
                ax.grid(True, alpha=0.3)

        fig.suptitle(f"Cycle overlays: {axis}", fontsize=14)
        plt.tight_layout()
        plt.show()

# ====== 確認出力：新仕様 norm の周期平均が ~0 か ======
if USE_CYCLE_RECALC_NORM and SUBTRACT_NORM_MEAN:
    for cond in ["LRL","RLR"]:
        if cond not in df_dict: continue
        print(f"\n[Check] {cond} での『周期再計算 norm（norm も周期平均を引く）』の周期平均：")
        for ds_name in USE_DATASETS:
            means = df_dict[cond].loc[ds_name].get("MEAN_acc_norm_per_cycle", None)
            if means is None or (isinstance(means, float) and np.isnan(means)):
                print(f"  {ds_name}: (no data)")
                continue
            arr = np.asarray(means, dtype=float)
            print(f"  {ds_name}: mean(abs(mean_per_cycle)) = {np.mean(np.abs(arr)):.3e}, "
                  f"max(abs(mean_per_cycle)) = {np.max(np.abs(arr)):.3e}, n={len(arr)}")
