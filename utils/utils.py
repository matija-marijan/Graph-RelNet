import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import os

def spherical_to_cartesian(azimuth: np.ndarray, elevation: np.ndarray, denormalize_to_degrees: bool = False, degrees_to_radians: bool = False, linear_task: bool = False):
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters:
    azimuth (numpy.ndarray): The azimuth angles in degrees.
    elevation (numpy.ndarray): The elevation angles in degrees.
    denormalize_to_degrees (bool): Whether to denormalize the angles from [-1, 1] to [-90, 90] degrees. Default is False.
    degrees_to_radians (bool): Whether to convert the angles from degrees to radians. Default is False.

    Returns:
    tuple[numpy.ndarray, numpy.ndarray]: The x and y coordinates in Cartesian coordinates.
    """
    if denormalize_to_degrees:
        azimuth = azimuth * 180 - 90
        elevation = elevation * 180 - 90

    if degrees_to_radians:
        azimuth = np.deg2rad(azimuth)
        elevation = np.deg2rad(elevation)

    if linear_task:
        x = np.cos(azimuth)
        y = np.zeros_like(elevation)
    else:
        x = np.cos(elevation) * np.cos(azimuth)
        y = np.cos(elevation) * np.sin(azimuth)

    return x, y

# Refactor, add docstrings, and type hints
def plot_source(ground_truth, prediction, convert_to_cartesian: bool = False, description: str = None, plot: bool = True, save: bool = False, maae: float = None, az_maae: float = None, el_maae: float = None):

    # Undersample the data for plotting if there are too many points
    if len(ground_truth) > 500:
        indices = np.random.choice(len(ground_truth), 500, replace=False)
        ground_truth = ground_truth[indices]
        prediction = prediction[indices]
        
    denormalize_to_degrees = True
    degrees_to_radians = False
    linear_task = False

    # TODO: Check if plotting is ok for [-90, 90] and [0, 180] ranges!

    # Extract azimuth and elevation from ground_truth and prediction
    azimuth_true, elevation_true = ground_truth[:, 0], ground_truth[:, 1]
    azimuth_prediction, elevation_prediction = prediction[:, 0], prediction[:, 1]

    # Convert spherical coordinates to Cartesian coordinates, and denormalize the angles
    if convert_to_cartesian:
        x_true, y_true = spherical_to_cartesian(azimuth = azimuth_true, elevation = elevation_true,
                                                denormalize_to_degrees = denormalize_to_degrees, degrees_to_radians = degrees_to_radians, linear_task = linear_task)
        x_prediction, y_prediction = spherical_to_cartesian(azimuth = azimuth_prediction, elevation = elevation_prediction,
                                                            denormalize_to_degrees = denormalize_to_degrees, degrees_to_radians = degrees_to_radians, linear_task = linear_task)
    else:
        x_true, y_true = np.rad2deg(azimuth_true), np.rad2deg(elevation_true)
        x_prediction, y_prediction = np.rad2deg(azimuth_prediction), np.rad2deg(elevation_prediction)

    # Plot the Cartesian coordinates
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.scatter(x_true, y_true, c='blue', marker='o', alpha=0.5, label='True', zorder=2)
    plt.scatter(x_prediction, y_prediction, c='red', marker='x', alpha=0.75, label='Prediction', zorder=3)
    
    # Plot dotted lines between corresponding true and prediction points
    for i in range(len(x_true)):
        plt.plot([x_true[i], x_prediction[i]], [y_true[i], y_prediction[i]], 'k--', alpha=0.25, zorder=1)
    
    plt.xlabel('Azimuth [deg]')
    plt.ylabel('Elevation [deg]')
    title_str = f'Azimuth and Elevation Plot'
    if maae is not None:
        title_str += f'\n(MAAE: {maae:.2f}°)'
    plt.title(title_str)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()

    # Scatter plot of azimuth
    plt.subplot(1, 3, 2)
    plt.scatter(x_true, x_prediction, c='blue', marker='o', alpha=0.5)
    plt.xlabel('True Azimuth [deg]')
    plt.ylabel('Predicted Azimuth [deg]')
    title_str = f'True vs. Predicted Azimuth'
    if az_maae is not None:
        title_str += f'\n(MAAE: {az_maae:.2f}°)'
    plt.title(title_str)
    plt.grid(True)
    plt.plot([min(x_true), max(x_true)], [min(x_true), max(x_true)], 'k--')  # Add y=x line

    # Scatter plot of elevation
    plt.subplot(1, 3, 3)
    plt.scatter(y_true, y_prediction, c='red', marker='x', alpha=0.5)
    plt.xlabel('True Elevation [deg]')
    plt.ylabel('Predicted Elevation [deg]')
    title_str = f'True vs. Predicted Elevation'
    if el_maae is not None:
        title_str += f'\n(MAAE: {el_maae:.2f}°)'
    plt.title(title_str)
    plt.grid(True)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--')  # Add y=x line

    plt.suptitle(f'{description}')

    plt.tight_layout()
    if plot:
        plt.show()
    if save:
        plt.savefig(f'results/plots/{description}.png')
    plt.close()

    # TODO: plot correlation between true and predicted azimuth and elevation

def calculate_distances(mic_array: np.ndarray, triu: bool = False):
    """
    Calculate the pairwise Euclidean distances between microphones in an array.

    Parameters:
    mic_array (numpy.ndarray): A 2D array where each row represents the 3D coordinates of a microphone array. 
    triu (bool): Whether to return only the upper triangular part of the distance matrix. Default is False.

    Returns:
    numpy.ndarray: A 2D array where the element at [i, j] represents the Euclidean distance between microphone i and microphone j.
    """
    distances = np.linalg.norm(mic_array[:, np.newaxis, :] - mic_array[np.newaxis, :, :], axis=2)
    if triu:
        return distances[np.triu_indices_from(distances, k=1)]
    return distances

def gcc_phat(sig, refsig, fs=None, max_tau=None, max_shift=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    Source: https://github.com/xiongyihui/tdoa/blob/master/gcc_phat.py
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Cross-Power Spectrum
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    # Phase Transform
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    if max_shift is None:
        max_shift = int(interp * n / 2)
    if max_tau and fs:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(cc) - max_shift

    # Sometimes, there is a 180-degree phase difference between the two microphones.
    # shift = np.argmax(np.abs(cc)) - max_shift

    if fs:
        tau = shift / float(interp * fs)
    else:
        tau = shift / float(interp)
    
    return tau, cc

def mean_absolute_angle_error(prediction: np.ndarray, ground: np.ndarray):
    """
    Calculate the mean absolute angle error between predicted and ground truth angles.

    Parameters:
    prediction (numpy.ndarray): Angles in radians; shape (N, 2) or (2, N). Columns/rows are [azimuth, elevation].
    ground (numpy.ndarray): Same shape as prediction.

    Returns:
    tuple[float, float, float]: (overall MAAE deg, azimuth MAAE deg, elevation MAAE deg)
    """
    assert prediction.shape == ground.shape, "Prediction and ground truth shapes do not match."

    pred = np.asarray(prediction)
    gt = np.asarray(ground)

    # Normalize to (N, 2)
    if pred.shape[0] == 2 and pred.ndim == 2:
        pred = pred.T
        gt = gt.T
    elif pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(f"Expected angles with shape (N, 2) or (2, N), got {prediction.shape}")

    error_deg = np.rad2deg(np.abs(pred - gt))  # shape (N, 2)

    overall = float(np.mean(error_deg))
    az = float(np.mean(error_deg[:, 0]))
    el = float(np.mean(error_deg[:, 1]))
    return overall, az, el

def maae_by_mic_count(predictions: np.ndarray, labels: np.ndarray, mic_counts: np.ndarray):
    """
    Compute MAAE grouped by number of microphones.

    Returns a dict: {mic_count: maae_value}
    """
    results = {}
    az_results = {}
    el_results = {}
    unique_counts = np.unique(mic_counts)
    
    for k in unique_counts:
        mask = mic_counts == k
        if mask.sum() == 0:
            continue
        maae, az_maae, el_maae = mean_absolute_angle_error(predictions[mask], labels[mask])
        
        # mean_absolute_angle_error returns a tensor; convert to float
        try:
            results[int(k)] = float(maae.item())
            az_results[int(k)] = float(az_maae.item())
            el_results[int(k)] = float(el_maae.item())
        
        except Exception:
            results[int(k)] = float(maae)
            az_results[int(k)] = float(az_maae)
            el_results[int(k)] = float(el_maae)

    return results, az_results, el_results

def plot_maae_bars(maae_dict: dict, title: str = "MAAE by mic count", save_path: str | None = None):
    if not maae_dict:
        return
    counts = sorted(maae_dict.keys())
    values = [maae_dict[c] for c in counts]
    plt.figure(figsize=(6, 4))
    plt.bar([str(c) for c in counts], values, color="#4C78A8")
    plt.xlabel("Number of microphones")
    plt.ylabel("MAAE (deg)")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

# fs : int or float?
def wavelength_normalization(mic_array: np.ndarray, fs: float, air_temp: float = 20, air_speed: float = 331):
    """
    Normalize the 3D coordinates of a microphone array by the wavelength of sound in the air at a given temperature.

    Parameters:
    mic_array (numpy.ndarray): A 2D array where each row represents the 3D coordinates of a microphone in an array.
    fs (int): The sampling frequency of the audio signal.
    air_temp (float): The air temperature in degrees Celsius. Default is 20.
    air_speed (float): The speed of sound in the air at 0 degrees Celcius in m/s. Default is 331.

    Returns:
    numpy.ndarray: A 2D array where each row represents the normalized 3D coordinates of a microphone array.
    float: The normalization parameter q1.
    """

    # Normalization distance
    distances = calculate_distances(mic_array, triu = True)
    # distances = pdist(mic_array)
    mic_distance = np.max(distances)
    # Speed of sound
    sound_speed = air_speed * np.sqrt(1 + air_temp / 273.15)
    # Normalization frequency
    f_norm = sound_speed / (2 * mic_distance)  
    # Normalization wavelength
    lambda_norm = sound_speed / f_norm 
    # Normalized microphone array
    mic_array = mic_array / lambda_norm
    # q1 normalization parameter
    q1 = fs / f_norm

    return mic_array, q1

def frequency_output_calculation(signals, angles, mic_array: np.ndarray, q1: float, distance: float = 1, SNR: float = 100, save: bool = False, filename: str = 'data/output.csv'):
    """
    Calculate the output of a microphone array given the audio sources and microphone array configuration.

    Parameters:
    signals (list): A list of audio signals from the source(s).
    angles (list): A list of tuples where each tuple contains the azimuth and elevation angles of the source(s).
    mic_array (numpy.ndarray): A 2D array where each row represents the 3D coordinates of a microphone array, normalized by wavelength.
    q1 (float): The normalization parameter.
    distance (float): The distance from the source to the center of the microphone array. Default is 1 m.
    SNR (float): The signal-to-noise ratio. Default is 100.
    save (bool): Whether to save the output to a CSV file. Default is False.
    filename (str): The name of the CSV file to save the output. Default is 'data/output.csv'.

    Returns:
    numpy.ndarray: A 2D array where each row represents the output of a microphone in the array.
    """

    # Far-field approximation (Fraunhofer distance)
    # distance_to_source > 2 * array_aperture ^ 2 / normalized_wavelength
    # -> Omit distance from steering vector (distance = 1.0 -> unit steering vector)
    # -> Approximately planar wavefront

    # Near-field approximation
    # -> Include distance in steering vector
    # -> Amplitude decay with 1/r^2
    # -> Spherical wavefront

    # TODO: Add noise
    #   - Add noise (done)
    #   - Add attenuation 1/r^2 (?)
    #   - Add reverberation (?)
    
    # Maybe not the best solution!
    if not isinstance(signals, list):
        signals = [signals]
    if not isinstance(angles, list):
        angles = [angles]

    assert len(signals) == len(angles), "The number of signals and sources (angles) should be equal."

    num_samples = len(signals[0])
    num_mics = len(mic_array)

    # Temperature coefficient
    k = np.arange(num_samples)
    temp = k / num_samples - 0.5

    # Total output
    output = np.zeros((num_mics, num_samples), dtype=complex)

    for signal, (az, el) in zip(signals, angles):

        az_rad = np.deg2rad(az)
        el_rad = np.deg2rad(el)
        
        # Steering vector
        a = distance * np.array([-np.cos(el_rad) * np.cos(az_rad),
                      -np.cos(el_rad) * np.sin(az_rad), 
                      -np.sin(el_rad)])
        
        # Attenuation (only for near-field)
        # if attenuation == True:
        #     r = np.linalg.norm(a)
        #     attenuation = 1 / (r ** 2)
        #     signal = signal * attenuation

        # Noise
        noise = np.random.normal(0, 1, num_samples)
        signal += noise / SNR
        
        # Frequency domain processing
        X = np.fft.fft(signal, num_samples)
        X_shift = np.fft.fftshift(X)
        
        # Compute steering vectors for all mics
        v = np.exp(-1j * 2 * np.pi * q1 * temp[:, None] * (mic_array @ a))
        
        # Time domain output for this source
        for mic in range(num_mics):
            S = np.fft.fftshift(v[:, mic] * X_shift)
            output[mic] += np.fft.ifft(S)
    
    if save:
        output_df = pd.DataFrame(output.real)
        output_df.to_csv(filename, index=False, header=False)

    return output.real

def fractional_delay(signal, delay_samples):
    """
    Fractional delay via frequency-domain phase shift with zero padding.
    """
    n = len(signal)
    nfft = 1 << (2*n - 1).bit_length()
    S = np.fft.rfft(signal, nfft)
    k = np.arange(len(S))
    phase = np.exp(-1j * 2*np.pi * k * delay_samples / nfft)
    y = np.fft.irfft(S * phase, nfft)[:n]
    return y

def time_output_calculation(signals, angles, mic_array: np.ndarray, SNRdB: float = 100, fs: float = 48000, air_temp: float = 20, air_speed: float = 331):
    """
    Far-field simulation of sound propagation in the time domain for given source positions and microphone array.

    Parameters:
    signals (list): List of input signals (1D numpy arrays).
    angles (list): List of tuples containing (azimuth, elevation) angles for each source.
    mic_array (np.ndarray): 2D array representing the microphone array positions.
    SNRdB (float): Signal-to-noise ratio in dB.
    fs (float): Sampling frequency in Hz.
    air_temp (float): Air temperature in degrees Celsius.
    air_speed (float): Speed of sound in air in m/s.

    Returns:
    np.ndarray: 2D array where each row represents the output of a microphone in the array.
    """

    # TODO: 
    # Improvements:
    #       - SNR sample from Uniform[Range]
    #       - Sensor imperfections (Random mic gain and phase offsets)
    sound_speed = air_speed * np.sqrt(1 + air_temp / 273.15)
    if not isinstance(signals, list):
        signals = [signals]
    if not isinstance(angles, list):
        angles = [angles]

    assert len(signals) == len(angles), "The number of signals and sources (angles) should be equal."
    
    num_samples = len(signals[0])
    num_mics = len(mic_array)
    output = np.zeros((num_mics, num_samples))

    for signal, (az, el) in zip(signals, angles):

        az_rad = np.deg2rad(az)
        el_rad = np.deg2rad(el)
        
        # Unit direction vector
        u = np.array([np.cos(el_rad) * np.cos(az_rad),
                      np.cos(el_rad) * np.sin(az_rad), 
                      np.sin(el_rad)])
        
        # Delays in seconds for each microphone
        delays = (mic_array @ u) / sound_speed

        for m in range(num_mics):
            delay_samples = delays[m] * fs
            # shifted = np.roll(signal, int(round(delay_samples)))
            shifted = fractional_delay(signal, delay_samples)
            output[m] += shifted

    # Noise
    for m in range(num_mics):
        sig_power = np.mean(output[m] ** 2)
        noise_power = sig_power / (10 ** (SNRdB / 10))
        noise = np.sqrt(noise_power) * np.random.randn(num_samples)
        output[m] += noise

    return output

if __name__ == "__main__":

    x = np.random.randn(1000)

    delay = 0.1
    fs = 1000  # Sampling frequency in Hz   
    roll = int(delay * fs)  # Convert delay to samples
    y = np.roll(x, roll)  # Shift x by roll samples
    t = np.arange(len(x)) / fs  # Time axis in seconds

    plt.figure(figsize=(10, 4))
    plt.plot(t, x, label='x (original)')
    plt.plot(t, y, label=f'y (shifted by {delay}s)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Sine Waves: Original and Shifted')
    plt.legend()
    plt.tight_layout()
    plt.show()

    delay = gcc_phat(x, y, interp=32)
    print(f"Estimated delay using gcc_phat: {delay[0]} samples, roll = {roll} samples")

    # GCC-PHAT is not permutation invariant - the delay can be negative or positive