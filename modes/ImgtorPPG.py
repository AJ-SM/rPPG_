import numpy as np
import cv2 as cv 
from scipy.signal import butter , filtfilt
from scipy.fft import fft, fftfreq

def extract_rgb_signal(frames, masks):

    rgb_signal = []

    for frame, mask in zip(frames, masks):

        mean_bgr = cv.mean(frame, mask=mask)[:3]

        rgb_signal.append(mean_bgr[::-1])  # convert to RGB

    return np.array(rgb_signal)


def green_method(rgb):

    return rgb[:,1]   # green channel



def chrom_method(rgb):

    R = rgb[:,0]
    G = rgb[:,1]
    B = rgb[:,2]

    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B

    alpha = np.std(X) / np.std(Y)

    S = X - alpha * Y

    return S



def pos_method(rgb):

    rgb = rgb / np.mean(rgb, axis=0)

    projection = np.array([[0,1,-1],[-2,1,1]])

    S = projection @ rgb.T

    alpha = np.std(S[0]) / np.std(S[1])

    signal = S[0] + alpha * S[1]

    return signal


from sklearn.decomposition import FastICA

def ica_method(rgb):

    ica = FastICA(n_components=3)

    components = ica.fit_transform(rgb)

    return components[:,0]


def run_all_rppg(rgb_signal):

    results = {}

    results["GREEN"] = green_method(rgb_signal)
    results["CHROM"] = chrom_method(rgb_signal)
    results["POS"] = pos_method(rgb_signal)
    results["ICA"] = ica_method(rgb_signal)

    return results


def bandpass_filter(signal, fs, low=0.7, high=4):

    nyquist = 0.5 * fs

    low = low / nyquist
    high = high / nyquist

    b, a = butter(3, [low, high], btype='band')

    filtered = filtfilt(b, a, signal)

    return filtered


def compute_fft(signal, fs):

    n = len(signal)

    fft_values = fft(signal)

    freqs = fftfreq(n, d=1/fs)

    return freqs, np.abs(fft_values)



def estimate_heart_rate(signal, fs):

    freqs, fft_mag = compute_fft(signal, fs)

    # keep only positive frequencies
    pos_mask = freqs > 0

    freqs = freqs[pos_mask]
    fft_mag = fft_mag[pos_mask]

    # find peak frequency
    peak_freq = freqs[np.argmax(fft_mag)]

    # convert to BPM
    heart_rate = peak_freq * 60

    return heart_rate



def process_rppg_pipeline(frames, masks, embeddings, fps):

    # Step 1: Extract RGB signal from ROI
    rgb_signal = extract_rgb_signal(frames, masks)

    # Step 2: Run all rPPG algorithms
    rppg_signals = run_all_rppg(rgb_signal)

    results = {}

    for algo_name, signal in rppg_signals.items():

        # Step 3: Bandpass filter
        filtered = bandpass_filter(signal, fps)

        # Step 4: Estimate heart rate
        hr = estimate_heart_rate(filtered, fps)

        results[algo_name] = {
            "rppg_signal": signal,
            "filtered_signal": filtered,
            "heart_rate": hr
        }

    return {
        "embeddings": embeddings,
        "rgb_signal": rgb_signal,
        "results": results
    }


# output = process_rppg_pipeline(frames, masks, embeddings, fps)


# print(output["results"]["GREEN"]["heart_rate"])
# print(output["results"]["CHROM"]["heart_rate"])
# print(output["results"]["POS"]["heart_rate"])
# print(output["results"]["ICA"]["heart_rate"])