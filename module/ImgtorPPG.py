import numpy as np
import cv2 as cv 
from scipy.signal import butter , filtfilt
from scipy.fft import rfft, rfftfreq
from sklearn.decomposition import FastICA

def extract_rgb_signal(frames, masks):
    rgb_signal = []
    for frame, mask in zip(frames, masks):
        # Calculate mean BGR in mask area
        mean_bgr = cv.mean(frame, mask=mask)[:3]
        rgb_signal.append(mean_bgr[::-1])  # Convert BGR to RGB
    return np.array(rgb_signal)

# --- Standard rPPG Methods ---
def green_method(rgb):
    return rgb[:, 1]

def chrom_method(rgb):
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    X = 3*R - 2*G
    Y = 1.5*R + G - 1.5*B
    alpha = np.std(X) / np.std(Y)
    return X - alpha * Y

def pos_method(rgb):
    # Plane-Orthogonal-to-Skin (POS) implementation [cite: 168, 170]
    rgb_norm = rgb / np.mean(rgb, axis=0)
    projection = np.array([[0, 1, -1], [-2, 1, 1]])
    S = projection @ rgb_norm.T
    alpha = np.std(S[0]) / np.std(S[1])
    return S[0] + alpha * S[1]

def ica_method(rgb):
    ica = FastICA(n_components=3)
    components = ica.fit_transform(rgb)
    return components[:, 0]

# --- Signal Processing for Paper Requirements ---
def bandpass_filter(signal, fs, low=0.7, high=4):
    nyquist = 0.5 * fs
    b, a = butter(3, [low / nyquist, high / nyquist], btype='band')
    return filtfilt(b, a, signal)

def compute_rppg_vector(signal, n_out=50):
    """
    Converts time-domain signal to a normalized frequency vector f in R^50.
    """
    # rfft returns only positive frequencies
    # Using n=99 ensures the output length is exactly 50
    fft_mag = np.abs(rfft(signal, n=99))
    
    # Normalize such that ||f||2 = 1 as required by paper 
    norm = np.linalg.norm(fft_mag, ord=2)
    if norm > 0:
        fft_mag = fft_mag / norm
        
    return fft_mag

def process_rppg_pipeline(frames, masks, embeddings, fps):
    # Step 1: Extract RGB signal
    rgb_signal = extract_rgb_signal(frames, masks)

    # Step 2: Run algorithms
    results = {}
    rppg_signals = {
        "GREEN": green_method(rgb_signal),
        "CHROM": chrom_method(rgb_signal),
        "POS": pos_method(rgb_signal),
        "ICA": ica_method(rgb_signal)
    }

    for algo_name, signal in rppg_signals.items():
        # Step 3: Filter
        filtered = bandpass_filter(signal, fps)

        # Step 4: Convert to Fourier Domain vector f 
        # This 50-length array is the actual "ground truth" for the RNN [cite: 190]
        rppg_vector = compute_rppg_vector(filtered, n_out=50)

        # Peak frequency for heart rate (BPM)
        freqs = rfftfreq(99, d=1/fps)
        peak_freq = freqs[np.argmax(rppg_vector)]
        hr = peak_freq * 60

        results[algo_name] = {
            "rppg_signal": signal,
            "filtered_signal": filtered,
            "rppg_vector": rppg_vector, # Use this for RNN loss
            "heart_rate": hr
        }

    return {
        "embeddings": embeddings,
        "rgb_signal": rgb_signal,
        "results": results
    }