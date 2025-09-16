import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("..")

from utils.utils import *
from utils.signal_utils import generate_random_signal, generate_mls
from utils.geometry_utils import generate_random_points, plot_geometry

def delay_and_sum(recordings: np.ndarray, array: np.ndarray, distance: float, fs: int, azimuth: np.ndarray, elevation: np.ndarray):

    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)

    energy_matrix = np.zeros((len(el_rad), len(az_rad)))

    # for elevation_index in tqdm(range(len(el_rad)), desc='Elevation', leave=False):
    for elevation_index in range(len(el_rad)):
        # for azimuth_index in tqdm(range(len(az_rad)), desc='Azimuth', leave=False):
        for azimuth_index in range(len(az_rad)):
            
            # x = distance * np.cos(el_rad[elevation_index]) * np.cos(az_rad[azimuth_index])
            # y = distance * np.cos(el_rad[elevation_index]) * np.sin(az_rad[azimuth_index])
            # z = distance * np.sin(el_rad[elevation_index])

            # dis = np.sqrt((array[:, 0] - x) ** 2 + (array[:, 1] - y) ** 2 + (array[:, 2] - z) ** 2)

            # delay = (dis - np.min(dis)) / 343

            # Unit direction vector
            u = [np.cos(el_rad[elevation_index]) * np.cos(az_rad[azimuth_index]),
                 np.cos(el_rad[elevation_index]) * np.sin(az_rad[azimuth_index]),
                 np.sin(el_rad[elevation_index])]
            
            delays = (array @ u) / 343
            delays -= np.min(delays)

            delayed_sample = np.round(delays * fs).astype(int)
            
            # print(delayed_sample)

            sum = np.zeros(recordings.shape[1])
            # for mic in tqdm(range(len(array)), desc='Microphone', leave=False):
            for mic in range(len(array)):
                if delayed_sample[mic] > 0:
                    # zeros = np.zeros(delayed_sample[mic])
                    # sum += np.concatenate((zeros, recordings[mic, :-delayed_sample[m]]))
                    sum[:len(recordings[mic]) - delayed_sample[mic]] += recordings[mic, delayed_sample[mic]:]
                else:
                    sum += recordings[mic, :]
                    # sum += recordings[mic, :-delayed_sample[mic]]
            E = np.sum(sum ** 2) / len(array)
            energy_matrix[elevation_index, azimuth_index] = E

    energy_matrix = energy_matrix / np.max(energy_matrix)
    energy_matrix = 10 * np.log10(energy_matrix + 1e-10)

    return energy_matrix

def estimate_angles(energy_matrix: np.ndarray, grid_azimuth: np.ndarray, grid_elevation: np.ndarray):
    """
    Estimate the azimuth and elevation angles from the delay and sum energy matrix.
    Args:
        energy_matrix (np.ndarray): The energy matrix obtained from the delay and sum method.
        grid_azimuth (np.ndarray): The azimuth angles used in the energy matrix.
        grid_elevation (np.ndarray): The elevation angles used in the energy matrix.
    Returns:
        estimated_angles (np.ndarray): The estimated angles from the energy matrix.
    """    

    max_index = np.unravel_index(np.argmax(energy_matrix), energy_matrix.shape)
    
    # Get the corresponding azimuth and elevation angles
    estimated_azimuth = grid_azimuth[max_index[1]]
    estimated_elevation = grid_elevation[max_index[0]]
    # estimated_angles = np.array([estimated_azimuth, estimated_elevation])

    # predictions = np.array([np.deg2rad(estimated_azimuth), np.deg2rad(estimated_elevation)])
    # ground = np.array([np.deg2rad(true_angles[0][0]), np.deg2rad(true_angles[0][1])])
    # # Calculate the loss as the difference between the true and estimated angles
    # maae, az_maae, el_maae = mean_absolute_angle_error(predictions, ground)

    return estimated_azimuth, estimated_elevation

def plot_energy_matrix(energy_matrix: np.ndarray, true_angles: list, grid_azimuth: np.ndarray, grid_elevation: np.ndarray, estimated_angles: np.ndarray, loss: float):
    """
    Plot the energy matrix as an image and mark the true and estimated source positions.
    Args:
        energy_matrix (np.ndarray): The energy matrix obtained from the delay and sum method.
        true_angles (list): The true angles of the source in degrees, e.g., [(azimuth, elevation)].
        grid_azimuth (np.ndarray): The azimuth angles used in the energy matrix.
        grid_elevation (np.ndarray): The elevation angles used in the energy matrix.
        estimated_angles (np.ndarray): The estimated angles from the energy matrix.
        loss (float): The Mean Absolute Angle Error (MAAE) between the estimated and true angles.
    """
    
    estimated_azimuth = estimated_angles[0]
    estimated_elevation = estimated_angles[1]

    plt.figure()
    # Plot the energy map
    plt.imshow(energy_matrix, 
            extent=[grid_azimuth[0], grid_azimuth[-1], grid_elevation[-1], grid_elevation[0]], 
            aspect='auto')
    # Mark true source position
    plt.scatter(true_angles[0][0], true_angles[0][1], c='red', marker='x', s=100, 
            label=f'True Source ({true_angles[0][0]:.2f}°, {true_angles[0][1]:.2f}°)')
    # Mark estimated source position
    plt.scatter(estimated_azimuth, estimated_elevation, c='black', marker='x', s=100, 
            label=f'Estimated ({estimated_azimuth}°, {estimated_elevation}°)')
    plt.title('Energy Map')
    plt.xlabel('Azimuth (degrees)')
    plt.ylabel('Elevation (degrees)')
    plt.colorbar()
    plt.legend(loc='lower right', title=f"MAE = {loss:.2f}°")
    plt.show()

    # plt.figure()
    # X, Y = np.meshgrid(azimuth, elevation)
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, energy_matrix, cmap='viridis')
    # ax.set_zlim([-3, 0])
    # ax.set_xlabel('Azimuth (degrees)')
    # ax.set_ylabel('Elevation (degrees)')
    # ax.set_zlabel('Energy (dB)')
    # ax.set_title('Energy Map')
    # plt.colorbar(ax.plot_surface(X, Y, energy_matrix, cmap='viridis'))
    # plt.show()

def main():

    fs = 48000
    t_max = 0.1
    distance = 1.0          # Far-field distance = 1.0 -> unit steering vector
    min_distance = 0.05
    max_diameter = 0.4

    # angles = [(30, 60)]
    angles = [(np.random.uniform(-75, 75), np.random.uniform(-75, 75))]
    signals = [generate_random_signal(t_max, fs, 256) for _ in range(len(angles))]
    # signals = [generate_mls(order = 8) for _ in range(len(angles))]

    # array_name = 'linear'
    # array_name = 'suvorov'
    # array_name = 'BK'
    array_name = 'P'
    array = pd.read_csv(f'data/arrays/fixed/{array_name}_array.csv', delimiter=',', header=None).values

    # Normalize the array
    distances = calculate_distances(array, triu = True)
    array = array * max_diameter / np.max(distances)

    array = generate_random_points(num_points = 24, min_distance = min_distance, max_diameter = max_diameter)
    # plot_geometry(array)

    # norm_array, q1 = wavelength_normalization(array, fs)
    # recordings = frequency_output_calculation(signals = signals, angles = angles, mic_array = norm_array, q1 = q1, distance = 1, save = False, SNR = 1)
    recordings = time_output_calculation(signals, angles, array, SNRdB = 30, fs = fs)

    # plt.figure(figsize=(12, 6))
    # for i in range(recordings.shape[0]):
    #     plt.plot(recordings[i], label=f'Mic {i+1}')
    # plt.plot(signals[0], label='Source Signal', color='black', linewidth=2)
    # plt.title('Microphone Recordings')
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    step = 2
    grid_azimuth = np.arange(-90, 91, step)
    grid_elevation = np.arange(90, -91, -step)

    energy_matrix = delay_and_sum(recordings, array, distance, fs, grid_azimuth, grid_elevation)
    est_az, est_el = estimate_angles(energy_matrix, grid_azimuth, grid_elevation)
    est_angles_arr = np.array([[est_az, est_el]])  # shape (1, 2)
    true_angles_arr = np.array([[angles[0][0], angles[0][1]]])  # shape (1, 2)
    loss = mean_absolute_angle_error(est_angles_arr, true_angles_arr)[0]
    plot_energy_matrix(energy_matrix, angles, grid_azimuth, grid_elevation, [est_az, est_el], loss)
    # print(f"MAE: {loss:.2f}, Estimated Angles: {estimated_angles}, True Angles: {[f'{a:.2f}' for a in angles[0]]}")

if __name__ == "__main__":
    main()