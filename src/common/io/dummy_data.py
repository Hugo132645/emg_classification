import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt

def generate_dummy_emg(seconds, fs, classes, block_s=5, mode="raw", seed=None, channel = "multi") -> tuple[pd.DataFrame, list[str], list[tuple[int, int]]]:
    #Applying replicable seed
    if seed is not None:
        np.random.seed(seed)
    gesture_rest = 0

    num_samples = int(seconds * fs)
    time = np.linspace(0, seconds, num_samples)
    block_sampling = int(block_s * fs)

    #Default gesture values
    gesture_values = {
        'fist': {'amp': 1.0, 'freq': 70},
        'open': {'amp': 0.7, 'freq': 55},
        'pinch': {'amp': 0.5, 'freq': 45},
        'rest': {'amp': 0.0, 'freq': 10}
    }

    #Creating skeletons for return values -> signal, label
    signal = np.zeros(num_samples)
    label = np.empty(num_samples, dtype='U10')
    iterator = 0
    prev_class = 'rest'

    interval_labels = []
    timestamps_ms = []

    #Iterating through each block
    while iterator < num_samples:
        #Random gesture, with rests in between
        if gesture_rest % 2 == 0:
            choices = [c for c in classes if c != prev_class]
            class_name = np.random.choice(choices)
            prev_class = class_name
        else:
            class_name = 'rest'

        #Defined end_block for single gesture
        end_block = min(iterator + block_sampling, num_samples)
        amp = gesture_values[class_name]['amp']
        freq = gesture_values[class_name]['freq']

        if class_name == "none":
            burst = np.zeros(end_block - iterator)
        else:

            #Generating band-limited EMG_like bursts for a gesture block
            high = min(freq + 20, 0.45 * fs)
            low = min(max(1.0, freq - 20), high * 0.5)
            sos = butter(4, [low / (fs / 2), high / (fs / 2)], btype='bandpass', output='sos')
            noise = np.random.randn(end_block - iterator)
            burst = sosfiltfilt(sos, noise)
            burst = amp * burst / np.max(np.abs(burst))
            #Make sure max is not zero

        signal[iterator:end_block] = burst
        label[iterator:end_block] = class_name

        #Assigning label and timestamp_ms block for return
        start_ms = (iterator / fs) * 1000
        end_ms = (end_block / fs) * 1000
        interval_labels.append(class_name)
        timestamps_ms.append((start_ms, end_ms))

        iterator += block_sampling
        gesture_rest += 1

    #Envelope mode
    if mode == "envelope":
        rectified = np.abs(signal)
        sos_env = butter(2, 5 / (fs / 2), btype='lowpass', output='sos')
        signal = sosfiltfilt(sos_env, rectified)



    #Slightly different version for multi_channel vs single_channel input
    if channel == "single":
        noise = np.random.laplace(0, 0.03 / np.sqrt(2), num_samples)
    else:
        noise = np.random.normal(0, 0.02, num_samples)
    signal += noise
    df = pd.DataFrame({'time': time, 'signal': signal, 'label': label})
    return df, interval_labels, timestamps_ms

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    signl, lbls, _ = generate_dummy_emg(seconds=1000, fs=1000, classes=["rest", "open", "pinch", "fist"])
    x_time = np.linspace(0, 1000, 1000*1000, dtype=np.float64)
    plt.plot(x_time, signl['signal'])
    plt.show()
