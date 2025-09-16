import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import max_len_seq
import os
import random
import soundfile as sf
import glob
from tqdm import tqdm

def generate_sine_wave(f: float = 1000, t: float = 1, fs: float = 48000):
    """
    Generate a sine wave signal for a given frequency, time duration, and sampling frequency.

    Parameters:
    f (float): Sine wave frequency in Hz. Default is 1000 Hz.
    t (float): Time duration in seconds. Default is 1 second.
    fs (float): Sampling frequency in Hz. Default is 48000 Hz.

    Returns:
    np.ndarray: Array of sine wave signal.
    """
    t = np.arange(0, t, 1/fs)
    signal = np.sin(2 * np.pi * f * t)
    return signal

def generate_mls(t: float = 1, fs: float = 48000, order: int = None):
    """
    Generate a maximum length sequence (MLS) signal for a given time duration and sampling frequency, or order.
    If order is not provided, it will be calculated based on the time duration and sampling frequency.
    
    Parameters:
    t (float): Time duration in seconds. Default is 1 second.
    fs (float): Sampling frequency in Hz. Default is 48000 Hz.
    order (int): Order of the MLS. Default is None.
    
    Returns:
    np.ndarray: MLS array.
    """
    if order is not None:
        mls = max_len_seq(nbits = order)[0]
    else:
        n_samples = int(t * fs)
        order = int(np.ceil(np.log2(n_samples)))
        mls = max_len_seq(nbits = order, length = n_samples)[0]
    return mls.astype(float)

def generate_random_signal(t: float = 1, fs: float = 48000, n_samples: int = None):
    """
    Generate random signals for a given time duration and sampling frequency from a standard normal distribution.

    Parameters:
    t (float): Time duration in seconds. Default is 1 second.
    fs (float): Sampling frequency in Hz. Default is 48000 Hz.
    n_samples (int): Number of samples to generate. If None, it will be calculated based on t and fs.

    Returns:
    np.ndarray: Array of random signals.
    """
    if n_samples is None:
        n_samples = int(t * fs)
    signals = np.random.randn(n_samples)
    return signals

def generate_source_angle(az_limit: float = 90, el_limit: float = 90):
    """
    Generate the incoming source azimuth and elevation angles in degrees from a uniform distribution.    

    Parameters:
    az_limit (float): The azimuth absolute angle limit in degrees. Default is 90.
    el_limit (float): The elevation absolute angle limit in degrees. Default is 180.

    Returns:
    tuple: A tuple containing the azimuth and elevation angles in degrees.
    """
    az = np.random.uniform(-az_limit, az_limit)
    el = np.random.uniform(-el_limit, el_limit)
    return az, el

def fetch_timit_signal(signal_duration: float = 0.1, fs: float = 16000, num_samples: int = None, idx: int = None, sample_idx: int = None):
    """
    Fetch a signal from the TIMIT speech corpus dataset.

    Parameters:
    signal_duration (float): Duration of the signal in seconds. Default is 0.1 seconds.
    fs (float): Sampling frequency in Hz. Default is 16000 Hz.
    num_samples (int): Number of samples to fetch. If None, it will be calculated based on signal_duration and fs.
    idx (int): Index of the WAV file to fetch. If None, a random file will be selected.
    sample_idx (int): Starting sample index for fetching the signal. If None, a random start index will be used.

    Returns:
    np.ndarray: A single channel array of audio samples from the TIMIT dataset.
    """
    
    if num_samples is None:
        num_samples = int(signal_duration * fs)

    timit_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'signals', 'timit')
    search_path = os.path.join(timit_dir, 'TEST', '*', '*', '*.WAV')
    wav_files = [os.path.relpath(f, timit_dir) for f in glob.glob(search_path)]

    if not wav_files:
        raise FileNotFoundError("No wav files found in {}".format(timit_dir))

    if idx is not None and 0 <= idx < len(wav_files):
        wav_file = wav_files[idx]
    else:
        wav_file = random.choice(wav_files)

    wav_path = os.path.join(timit_dir, wav_file)
    data, sr = sf.read(wav_path)

    # If data is stereo, use first channel
    if data.ndim > 1:
        print("Stereo signal detected, using first channel only.")
        data = data[:, 0]  # Take first channel if stereo

    if len(data) < num_samples:
        raise ValueError("Requested more samples than available in the signal.")

    start = sample_idx if sample_idx is not None else random.randint(0, len(data) - num_samples)
    end = start + num_samples

    tqdm.write(f"Read file: {wav_path}")
    tqdm.write(f"Read samples: {start}:{end}")

    return data[start:end].astype(float)

# Example usage
if __name__ == "__main__":
    t = 1.0  # 1 second
    fs = 1000  # 1000 Hz

    # signal = generate_random_signal(t, fs)
    signal = fetch_timit_signal(signal_duration=t, fs=fs, num_samples=1024, idx = 1, sample_idx = 10)

    print(len(signal))
    print(max(signal))

    # Plot the generated signal
    plt.plot(signal)
    plt.title("Random Signal")
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.show()