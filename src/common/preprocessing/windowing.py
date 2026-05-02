import pandas as pd
import numpy as np

def window_signal(x, fs, window_ms=200, hop_ms=100, labels=None, timestamps_ms=None) -> tuple[np.ndarray, list, np.ndarray]:
    #Copied 1D array not to change value
    x = x.copy()
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)

    win_len = int(fs * window_ms / 1000)
    hop_len = int(fs * hop_ms / 1000)
    n_samples = len(x)

    windows, window_labels, times_ms = [], [], []

    for start in range(0, n_samples - win_len + 1, hop_len):
        #Defining basic variables
        end = start + win_len
        t_start_ms = (start/fs) * 1000
        t_end_ms = (end/fs) * 1000
        times_ms.append([t_start_ms, t_end_ms])
        windows.append(x[start:end])
        overlap_durations = {}
        majority_vote = None

        #Searching and applying majority vote for that specific window
        if labels is not None and timestamps_ms is not None:
            #Zips Time values and Label values, add the given quantity to the overlap_durations dict,
            for (start_i, end_i), label_i in zip(timestamps_ms, labels):
                overlap = max(0 , min(t_end_ms, end_i) - max(t_start_ms, start_i))
                if overlap > 0:
                    overlap_durations[label_i] = overlap_durations.get(label_i, 0) + overlap
            if overlap_durations:
                #Searching through dictionary which key value has the largest value
                majority_vote = max(overlap_durations, key=overlap_durations.get)
            else:
                majority_vote = None
        window_labels.append(majority_vote)


    windows = np.stack(windows).astype(np.float32)
    times_ms = np.array(times_ms).astype(np.float32)

    return windows, window_labels, times_ms

def window_signal_np(
    x,
    fs,
    window_ms=200,
    hop_ms=100,
    labels=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x).copy()
    x = np.nan_to_num(x, nan=0.0).astype(np.float32)

    if x.ndim not in (1, 2):
        raise ValueError("x must be 1D or 2D")

    n_samples = x.shape[0]

    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("labels must be 1D")
        if len(labels) != n_samples:
            raise ValueError("labels must have the same number of samples as x")

    win_len = int(fs * window_ms / 1000)
    hop_len = int(fs * hop_ms / 1000)

    if win_len <= 0 or hop_len <= 0:
        raise ValueError("window_ms and hop_ms must produce positive lengths")
    if n_samples < win_len:
        raise ValueError("signal is shorter than one window")

    windows = []
    window_labels = []
    times_ms = []

    for start in range(0, n_samples - win_len + 1, hop_len):
        end = start + win_len

        windows.append(x[start:end])

        t_start_ms = (start / fs) * 1000.0
        t_end_ms = (end / fs) * 1000.0
        times_ms.append([t_start_ms, t_end_ms])

        if labels is not None:
            y_win = labels[start:end]
            vals, counts = np.unique(y_win, return_counts=True)
            majority_vote = vals[np.argmax(counts)]
            window_labels.append(majority_vote)
        else:
            window_labels.append(None)

    if labels is not None:
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        print("Window labels available:", unique_labels)
        print("Window label counts:", dict(zip(unique_labels.tolist(), counts.tolist())))

    windows = np.stack(windows).astype(np.float32)
    window_labels = np.asarray(window_labels)
    times_ms = np.asarray(times_ms, dtype=np.float32)

    return windows, window_labels, times_ms

