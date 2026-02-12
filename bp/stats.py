# %%
import pandas as pd 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import glob
import os
from scipy.signal import find_peaks
# %%
# res = pd.read_csv("../experiments/full-v3/checkpoints/inference/10020306_23052851_out.csv")
# res["t"] = pd.to_datetime(res["t"])

def get_SBP_DBP(abp, t, fs):
    """
    Inputs:
    - abp (pd Series)
    - t (pd Series)
    - fs (float)
    Returns:
    - SBPx, SBPy, DBPx, DBPy (pd Series)
    """

    prominence = abp.std()
    peaks, _ = find_peaks(abp, distance=fs*0.4, prominence=prominence)
    systolic_values = abp[peaks]
    systolic_times = t[peaks]
    
    diastolic_values = []
    diastolic_indices = []

    for i in range(len(peaks)-1):
      valley = abp[peaks[i]:peaks[i+1]]
      
      min_idx = valley.idxmin()
      diastolic_indices.append(min_idx)
      diastolic_values.append(valley[min_idx])

    diastolic_values = pd.Series(diastolic_values, index=diastolic_indices)
    diastolic_times  = t.loc[diastolic_indices]

    return systolic_times, systolic_values, diastolic_times, diastolic_values

def SBP_DBP_averages(output_file, fs=62.4725, usecols = ["t", "P", "ABP"], window_size=4):
    """
    Reads output csv file in chunks with window
    Input
    - output_file: .csv ground truth and predictions
    - usecols: inference output columns to use for metrics/plotting
    - window_size (int): size of window 
    """
    true_SBPs, true_DBPs, pred_SBPs, pred_DBPs = [], [], [], []
    
    dt = 1.0 / fs
    
    window_size = int(window_size / dt) #number of samples to grab
    chunksize = 2 * window_size #some output files is only ~1000 rows, need smaller chunksize (8s)

    current_window = pd.DataFrame()

    # Read output in 2*(window size) chunks
    for chunk in pd.read_csv(output_file, usecols=usecols, chunksize=chunksize):
    # Append the chunk to current window
        current_window = pd.concat([current_window, chunk], ignore_index=True)
        
        # As long as there's a full window of data
        while len(current_window) >= window_size:
            data = current_window.iloc[:window_size]
            # Get all SBP/DBP values in window
            _, SBP_true, _, DBP_true = get_SBP_DBP(data["ABP"], data["t"], fs)
            _, SBP_pred, _, DBP_pred = get_SBP_DBP(data["P"], data["t"], fs)

            # Append window average to list of sliding window averages
            true_SBPs.append(SBP_true.mean())
            pred_SBPs.append(SBP_pred.mean())
            true_DBPs.append(DBP_true.mean())
            pred_DBPs.append(DBP_pred.mean())

            # Slide the window down, drop rows that have been processed
            current_window = current_window.iloc[window_size:].reset_index(drop=True)
        
        # Keep tail of window for next chunk
        current_window = current_window.iloc[-window_size:].reset_index(drop=True)
    
    return true_SBPs, pred_SBPs, true_DBPs, pred_DBPs


def calculate_BA_stats(true_vals, pred_vals, per_subject=False):
    """
    Calculate Bland-Altman stats for a list of true and predicted BP values.
    
    Inputs:
    - true_vals, pred_vals: list of measurements (average systolic or diastolic BP), 
    one measurement per window(default_size=4)
    - per_subject: plots each subject as point when True, plots windowed values for single subject when False
    
    Returns:
    - means: list of mean values ( (true + pred)/2 ) when per_subject=False
    or single value when per_subject=True
    - diffs: list of differences (true - pred) when per_subject=False
    or single value when per_subject=True
    - means_std: None when per_subject=False
    (float) std of means of windows when per_subject=True
    - diffs_std: None when per_subject=False
    (float) std of diffs of windows when per_subject=True
    """

    # No window level diff std or mean std for single subject plotting
    std_diff  = None    
    std_means = None
    
    # Calculate means and diffs of (true, prediction) measurements per window
    means = np.array([(t + p) / 2 for t, p in zip(true_vals, pred_vals)] , dtype=float)
    diffs = np.array([t - p for t, p in zip(true_vals, pred_vals)], dtype=float)
    mask = ~np.isnan(means) & ~np.isnan(diffs)
    means, diffs = means[mask], diffs[mask]

    
    if per_subject:
        # Calculate window level std for error bars in plot
        std_diff  = np.std(diffs)    
        std_means = np.std(means)
        
        # Collapse window means and difs
        means = np.mean(means)
        diffs = np.mean(diffs)
    
    return means, diffs, std_means, std_diff

def calculate_mae(inference_dir):
    # Collect output CSVs
    file_paths = glob.glob(os.path.join(inference_dir, "*.csv"))
    mae = 0.0
    for file_path in file_paths:
        df = pd.read_csv(file_path, usecols= ["t", "ABP", "P"])
        abs_error = (df["ABP"]-df["P"]).abs().mean()
        mae += abs_error

    return mae


def plot_bland_altman(inference_dir, per_subject=True, fs=62.4725, window_size=4):
    """
    Plot Bland-Altman for SBP/DBP from inference CSVs.
    
    Args:
    - inference_dir: path to CSVs
    - per_subject: if True, plot subject-level across all files; if False, plot window-level for first subject
    - fs: sampling frequency
    - window_size: size of window in seconds
    """
    
    def plot_subplt(ax, means, diffs, means_std=None, diffs_std=None, title=""):
        means, diffs = np.array(means), np.array(diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)

        # Scatter with error bars if available
        if means_std is not None or diffs_std is not None:
            ax.errorbar(means, diffs, 
                        xerr=means_std, yerr=diffs_std, 
                        fmt='o', alpha=0.6, ecolor='green', capsize=3)
        else:
            ax.scatter(means, diffs, alpha=0.6)

        # Horizontal lines
        ax.axhline(mean_diff, color="black", linestyle="-", label=f"Mean diff = {mean_diff:.2f}")
        ax.axhline(mean_diff + std_diff, color="black", linestyle=(5, (10, 3)), label= f"+/- 1 SD = {std_diff:.2f}")
        ax.axhline(mean_diff - std_diff, color="black", linestyle=(5, (10, 3)))
        ax.axhline(mean_diff + 2*std_diff, color="black", linestyle="--", label= f"+/- 2 SD = {2*std_diff:.2f}")
        ax.axhline(mean_diff - 2*std_diff, color="black", linestyle="--")

        ax.set_title(title)
        ax.set_xlabel("(Invasive + Predicted Arterial Pressure)/2")
        ax.set_ylabel("Invasive - Predicted Arterial Pressure")
        ax.legend()

    
    SBP_means, SBP_difs, SBP_means_std, SBP_difs_std = [], [], [], []
    DBP_means, DBP_difs, DBP_means_std, DBP_difs_std = [], [], [], []

    # Collect CSVs
    file_paths = sorted(glob.glob(os.path.join(inference_dir, "*.csv")))
    if not file_paths:
        raise ValueError(f"No CSV files found in {inference_dir}")
    
    if per_subject:
        # Loop through all subjects
        for file_path in file_paths:
            print(f"Processing {file_path}")
            true_SBPs, pred_SBPs, true_DBPs, pred_DBPs = SBP_DBP_averages(file_path, fs, window_size=window_size)

            # Pool window-level B-A means and diffs (global average), retain std of window means and diffs
            sbp_mean, sbp_diff, sbp_mean_std, sbp_diff_std = calculate_BA_stats(true_SBPs, pred_SBPs, per_subject=True)
            dbp_mean, dbp_diff, dbp_mean_std, dbp_diff_std = calculate_BA_stats(true_DBPs, pred_DBPs, per_subject=True)

            SBP_means.append(sbp_mean)
            SBP_difs.append(sbp_diff)
            SBP_means_std.append(sbp_mean_std)
            SBP_difs_std.append(sbp_diff_std)

            DBP_means.append(dbp_mean)
            DBP_difs.append(dbp_diff)
            DBP_means_std.append(dbp_mean_std)
            DBP_difs_std.append(dbp_diff_std)

    else:
        # Only first subject, plot window-level stats
        file_path = file_paths[0]
        print(f"Processing first subject {file_path}")
        true_SBPs, pred_SBPs, true_DBPs, pred_DBPs = SBP_DBP_averages(file_path, fs, window_size=window_size)

        # SBP_means_std, SBP_difs_std = None -> (per_subject=False)
        SBP_means, SBP_difs, SBP_means_std, SBP_difs_std = calculate_BA_stats(true_SBPs, pred_SBPs, per_subject=False)
        DBP_means, DBP_difs, DBP_means_std, DBP_difs_std = calculate_BA_stats(true_DBPs, pred_DBPs, per_subject=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    plot_subplt(axes[0], SBP_means, SBP_difs, means_std=SBP_means_std, diffs_std=SBP_difs_std, title="Bland-Altman Systolic ABP")
    plot_subplt(axes[1], DBP_means, DBP_difs, means_std=DBP_means_std, diffs_std=DBP_difs_std, title="Bland-Altman Diastolic ABP")
    plt.tight_layout()
    plt.show()


# %%

def plot_vis(res, vis_start=0, vis_len=1000):
    tdf = res.iloc[vis_start:vis_start+vis_len]


    fig, axs = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
        
    t = tdf["t"]
    axs[0].plot(t, tdf["II"], label='ECG', alpha=0.6)
    axs[0].set_ylabel("ECG (mV)")
    ax1b = axs[0].twinx()
    ax1b.plot(t, tdf["PLETH"], label='PPG', alpha=0.6, color='tab:green')
    ax1b.set_ylabel("PPG (a.u.)")
    ax1b.legend(loc='upper right')
    
    
    # if "phi" in tdf:
    #     ax1b.plot(t, np.sin(tdf["phi"]), label="$sin(\\phi (t))$", linestyle='--', alpha=0.6)
    #     ax1b.plot(t, np.sin(tdf["phi_ref"]), label="$True sin(\\phi (t))$", linestyle='--', alpha=0.6)
    #     ax1b.legend(loc='upper right')


    axs[1].plot(t, tdf["imputed_Q"], label='Q(t)', alpha=0.6)
    axs[1].set_ylabel("Inferred Flow (mL/s)")
    ax2b = axs[1].twinx()
    ax2b.plot(t, tdf["imputed_dQdt"], label="$ \\partial Q/ \\partial t$", alpha=0.6, linestyle='--', color='tab:green')
    ax2b.set_ylabel("mL/$s^2$")
    ax2b.legend(loc='upper right')
        

    axs[2].plot(t, tdf["imputed_P"], label='P(t)', alpha=0.6)
    axs[2].plot(t, tdf["ABP"], label='ABP', alpha=0.6)
    axs[2].set_ylabel("Pressure (mmHg)")

    ax3b = axs[2].twinx()
    ax3b.plot(t, tdf["imputed_dPdt"], label="$ \\partial P/ \\partial t$", alpha=0.6, linestyle='--', color='tab:green')
    ax3b.set_ylabel("mmHg/s")
    ax3b.legend(loc='upper right')

    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    axs[2].legend(loc='upper left')
    # axs[3].legend(loc='upper left')
    axs[-1].set_xlabel("Time")
        
    plt.tight_layout()
    return axs