import numpy as np
import pandas as pd
import os

from utils.utils import *
from utils.geometry_utils import generate_random_points, plot_geometry
from utils.signal_utils import generate_random_signal, generate_source_angle, generate_sine_wave, generate_mls, fetch_timit_signal

from tqdm import tqdm
import argparse
import random

def set_seed(seed: int = 0):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    random.seed(seed)
    np.random.seed(seed)

def create_dataset(num_arrays: int = 20, num_signals: int = 25, num_angles: int = 100, num_sources: int = 1, 
                   signal_type: str = 'random', sampling_frequency: int = 48000, signal_duration: float = 0.1, SNR: float = 100,
                   sine_frequency: int = 1000, mls_order: int = None, num_samples: int = None,
                   array_type: str = 'random', min_distance: float = 0.025, max_diameter: float = 0.4, num_mics: int = None,
                   angle_type: str = 'random', az_limit: float = 75, el_limit: float = 75, save_cartesian: bool = False,
                   dataset_path: str = 'data', runtime_dataset: bool = False):

    if angle_type == 'sweep':
        azimuths = np.linspace(-az_limit, az_limit, num_angles)
        elevations = np.linspace(-el_limit, el_limit, num_angles)
        azimuth_grid, elevation_grid = np.meshgrid(azimuths, elevations)
        azimuths, elevations = azimuth_grid.ravel(), elevation_grid.ravel()
        num_angles = len(azimuths)

    for i in tqdm(range(num_arrays), desc='Arrays', position=0, leave=False):
        num_points = np.random.randint(4, 8) if num_mics is None else num_mics

        if array_type == 'random':
            mic_array = generate_random_points(num_points=num_points, min_distance=min_distance, max_diameter=max_diameter)

        elif array_type in ['linear', 'suvorov', 'BK', 'P']:
            mic_array = pd.read_csv(f'{args.dataset_path}/arrays/fixed/{array_type}_array.csv', header=None).values
            mic_array = mic_array * max_diameter / np.max(calculate_distances(mic_array, triu=True))

        elif array_type == '1D':
            mic_array = np.zeros((num_points, 3))
            mic_array[:, 0] = 0
            mic_array[:, 1] = np.linspace(-max_diameter / 2, max_diameter / 2, num_points)
            mic_array[:, 2] = 0

        elif array_type == '1D_random':
            mic_array = np.zeros((num_points, 3))
            mic_array[:, 0] = 0
            mic_array[:, 1] = np.random.uniform(-max_diameter / 2, max_diameter / 2, num_points)
            mic_array[:, 2] = 0

        mic_array_df = pd.DataFrame(mic_array)
        # mic_array_df = pd.DataFrame(mic_array / max_diameter * 2)   
        # norm_array, q1 = wavelength_normalization(mic_array, sampling_frequency)        # Fs!
        # plot_geometry(mic_array)

        for j in tqdm(range(num_signals), desc='Signals', position=1, leave=False):

            if signal_type == 'random':
                signals = [generate_random_signal(signal_duration, sampling_frequency, num_samples) for _ in range(num_sources)]
            elif signal_type == 'sine':
                signals = [generate_sine_wave(f = sine_frequency, t = signal_duration, fs = sampling_frequency) for _ in range(num_sources)]
            elif signal_type == 'sine_random':
                signals = [generate_sine_wave(f = np.random.uniform(20, 20000), t = signal_duration, fs = sampling_frequency) for _ in range(num_sources)]
            elif signal_type == 'mls':
                signals = [generate_mls(order = mls_order) for _ in range(num_sources)]
            elif signal_type == 'timit':
                signals = [fetch_timit_signal(signal_duration = signal_duration, fs = sampling_frequency, num_samples = num_samples) for _ in range(num_sources)]

            for k in tqdm(range(num_angles), desc='Angles', position=2, leave=False):

                if angle_type == 'sweep':
                    azimuth = azimuths[k]
                    elevation = elevations[k]
                    angles = [(azimuth, elevation) for _ in range(num_sources)]

                elif angle_type == 'random':
                    angles = [generate_source_angle(az_limit = az_limit, el_limit = el_limit) for _ in range(num_sources)]

                # angles = [(angle[0], 0) for angle in angles]  # Only azimuth angles (linear case)
                # norm_angles = [((angle[0] + 90) / 180, angle[1] / 180) for angle in angles]
                # norm_angles = [(np.log(angle[0] + 90 + 1), np.log(angle[1] + 1)) for angle in angles]

                if save_cartesian:
                    norm_angles = []
                    for (az, el) in angles:
                        az, el = np.deg2rad(az), np.deg2rad(el)
                        norm_angles.append([np.cos(el) * np.cos(az),
                                            np.cos(el) * np.sin(az),
                                            np.sin(el)])
                else:
                    norm_angles = np.deg2rad(angles)

                angles_df = pd.DataFrame(norm_angles)

                angles_df.to_csv(f'{dataset_path}/sources/angles_{i}_{j}_{k}.csv', index=False, header=False)
                
                if runtime_dataset:
                    output = signals
                else:
                    # output = frequency_output_calculation(signals, angles, norm_array, q1, save=True, filename=f'{dataset_path}/signals/signals_{i}_{j}_{k}.csv', SNR = SNR)
                    output = time_output_calculation(signals, angles, mic_array, SNRdB = SNR, fs = sampling_frequency)
                output_df = pd.DataFrame(output)
                output_df.to_csv(f'{dataset_path}/signals/signals_{i}_{j}_{k}.csv', index=False, header=False)

                mic_array_df.to_csv(f'{dataset_path}/arrays/array_{i}_{j}_{k}.csv', index=False, header=False)

                # plt.figure(figsize=(12, 6))
                # for plt_i in range(output.shape[0]):
                #     plt.plot(output[plt_i], label=f'Mic {plt_i+1}')
                # plt.plot(signals[0], label='Source Signal', color='black', linewidth=2)
                # plt.title('Microphone Recordings')
                # plt.xlabel('Sample')
                # plt.ylabel('Amplitude')
                # plt.legend()
                # plt.tight_layout()
                # plt.show()

    tqdm.write(f"Dataset generated with {num_arrays} arrays and {num_arrays * num_signals * num_angles} total samples.")

def parsing():
    
    parser = argparse.ArgumentParser(description='Create dataset for sound source localization')

    parser.add_argument('--split_dataset', action='store_true', default=False,
                        help="Split the dataset into training and test sets by microphone arrays (default: False).")
    parser.add_argument('--split_ratio', type=float, default=0.75,
                        help="Ratio of training to test dataset (default: 0.75).")
    parser.add_argument('--clear', action='store_true', default=False,
                        help="Only clear the dataset and exit (default: False).")
    parser.add_argument('--validation', action='store_true', default=False,
                        help="Create ONLY an independent validation set (default: False).")
    
    parser.add_argument('--num_arrays', type=int, default=20,
                        help="Number of arrays to generate (default: 20).")
    parser.add_argument('--num_signals', type=int, default=50,
                        help="Number of signals per array (default: 50).")
    parser.add_argument('--num_angles', type=int, default=18,
                        help="Number of angles per signal (default: 18).")
    parser.add_argument('--num_sources', type=int, default=1,
                        help="Number of audio sources (default: 1).")
    
    parser.add_argument('--signal_type', type=str, choices=['random', 'sine', 'mls', 'sine_random', 'timit'], default='random',
                        help="Type of signal generation technique: [random, sine, mls, sine_random, timit] (default: random).")
    parser.add_argument('--sampling_frequency', type=int, default=48000,
                        help="Sampling frequency (default: 48000).")
    parser.add_argument('--signal_duration', type=float, default=0.1,
                        help="Maximum time for signal generation (default: 0.1).")
    parser.add_argument('--SNR', type=float, default=100,
                        help="Signal-to-noise ratio (default: 100).")
    
    parser.add_argument('--sine_frequency', type=float, default=1000,
                        help="Frequency of the sine wave signal (default: 1000).")
    parser.add_argument('--mls_order', type=int, default=None,
                        help="Order of the maximum length sequence (MLS) signal (default: None).")
    parser.add_argument('--num_samples', type=int, default=None,
                        help="Number of samples for generating signals (default: None).")
    
    parser.add_argument('--array_type', type=str, choices=['random', 'linear', 'suvorov', 'BK', 'P', '1D', '1D_random'], default='random',
                        help="Type of array generation technique: [random, linear, suvorov, BK, P, 1D, 1D_random] (default: random).")
    parser.add_argument('--min_distance', type=float, default=0.025,
                        help="Minimum distance between microphones (default: 0.025).")
    parser.add_argument('--max_diameter', type=float, default=0.4,
                        help="Maximum diameter of the array (default: 0.4).")
    parser.add_argument('--num_mics', type=int, default=None,
                        help="Number of microphones in the array (default: None, randomly chosen between 5 and 25).")

    parser.add_argument('--angle_type', type=str, choices=['random', 'sweep'], default='random',
                        help="Type of angles to generate: 'random' or 'sweep' (default: random).")
    parser.add_argument('--az_limit', type=float, default=75,
                        help="Azimuth absolute angle limit in degrees (default: 75).")
    parser.add_argument('--el_limit', type=float, default=75,
                        help="Elevation absolute angle limit in degrees (default: 75).")
    parser.add_argument('--save_cartesian', action='store_true', default=False,
                        help="Whether to save the Cartesian coordinates of the angles.")

    parser.add_argument('--dataset_path', type=str, default='data',
                        help="Path to the root dataset folder (default: data).")
    parser.add_argument('--runtime_dataset', action='store_true',
                        help="Create a runtime dataset and don't run output simulation (default: False).")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility (default: None).")

    args = parser.parse_args()
    
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")

    return args

def create_clear_folders(root_folder: str = 'data'):
    """
    Clear the specified root folder and create necessary subfolders for dataset storage.
    
    Parameters:
    root_folder (str): The root folder where the dataset will be stored. Default is 'data'.
    """
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    for subfolder in ['signals', 'arrays', 'sources', 'processed']:
        subfolder_path = os.path.join(root_folder, subfolder)

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            if os.path.isfile(file_path) and file_path.endswith('.csv'):
                os.remove(file_path)

    processed_folder = os.path.join(root_folder, 'processed')
    if os.path.exists(processed_folder):
        for filename in os.listdir(processed_folder):
            if filename.endswith('.pt'):
                os.remove(os.path.join(processed_folder, filename))

def main(args: argparse.Namespace = None):

    if args.validation:
        create_clear_folders(f'{args.dataset_path}/validation')
    else:    
        if args.split_dataset:
            create_clear_folders(f'{args.dataset_path}/train')
            create_clear_folders(f'{args.dataset_path}/test')

        else:
            create_clear_folders(args.dataset_path)

    if args.clear:
        exit()

    if args.seed is not None:
        set_seed(args.seed)

    if args.validation:
        create_dataset(num_arrays=args.num_arrays, num_signals=args.num_signals, num_angles = args.num_angles, num_sources=args.num_sources, 
                       sampling_frequency=args.sampling_frequency, signal_duration=args.signal_duration, signal_type=args.signal_type,
                       sine_frequency=args.sine_frequency, mls_order=args.mls_order, num_samples=args.num_samples, SNR=args.SNR,
                       array_type=args.array_type, min_distance=args.min_distance, max_diameter=args.max_diameter, num_mics=args.num_mics,
                       angle_type=args.angle_type, az_limit=args.az_limit, el_limit=args.el_limit, save_cartesian=args.save_cartesian,
                       dataset_path=f'{args.dataset_path}/validation', runtime_dataset=args.runtime_dataset)
    else:
        if args.split_dataset:
            train_dataset_path = f'{args.dataset_path}/train'
            test_dataset_path = f'{args.dataset_path}/test'
            train_num_arrays = int(args.split_ratio * args.num_arrays)
            test_num_arrays = args.num_arrays - train_num_arrays

            create_dataset(num_arrays=train_num_arrays, num_signals=args.num_signals, num_angles = args.num_angles, num_sources=args.num_sources, 
                        sampling_frequency=args.sampling_frequency, signal_duration=args.signal_duration, signal_type=args.signal_type,
                        sine_frequency=args.sine_frequency, mls_order=args.mls_order, num_samples=args.num_samples, SNR=args.SNR,
                        array_type=args.array_type, min_distance=args.min_distance, max_diameter=args.max_diameter, num_mics=args.num_mics,
                        angle_type=args.angle_type, az_limit=args.az_limit, el_limit=args.el_limit, save_cartesian=args.save_cartesian,
                        dataset_path=train_dataset_path, runtime_dataset=args.runtime_dataset)

            create_dataset(num_arrays=test_num_arrays, num_signals=args.num_signals, num_angles = args.num_angles, num_sources=args.num_sources, 
                        sampling_frequency=args.sampling_frequency, signal_duration=args.signal_duration, signal_type=args.signal_type,
                        sine_frequency=args.sine_frequency, mls_order=args.mls_order, num_samples=args.num_samples, SNR=args.SNR,
                        array_type=args.array_type, min_distance=args.min_distance, max_diameter=args.max_diameter, num_mics=args.num_mics,
                        angle_type=args.angle_type, az_limit=args.az_limit, el_limit=args.el_limit, save_cartesian=args.save_cartesian,
                        dataset_path=test_dataset_path, runtime_dataset=args.runtime_dataset)
        else:
            create_dataset(num_arrays=args.num_arrays, num_signals=args.num_signals, num_angles = args.num_angles, num_sources=args.num_sources, 
                            sampling_frequency=args.sampling_frequency, signal_duration=args.signal_duration, signal_type=args.signal_type,
                            sine_frequency=args.sine_frequency, mls_order=args.mls_order, num_samples=args.num_samples, SNR=args.SNR,
                            array_type=args.array_type, min_distance=args.min_distance, max_diameter=args.max_diameter, num_mics=args.num_mics,
                            angle_type=args.angle_type, az_limit=args.az_limit, el_limit=args.el_limit, save_cartesian=args.save_cartesian,
                            dataset_path=args.dataset_path, runtime_dataset=args.runtime_dataset)

if __name__ == '__main__':
    args = parsing()
    main(args)