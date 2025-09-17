import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("..")

from utils.utils import *
from utils.dataset_utils import InMemoryGraphDataset

from evaluation.delay_and_sum import delay_and_sum, estimate_angles, plot_energy_matrix
from utils.geometry_utils import plot_geometry

import argparse
import wandb
import os
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json

# Helper for parallel processing. Must be at module level to be picklable by multiprocessing.
def _process_delay_and_sum(recordings, array, distance, fs, grid_azimuth, grid_elevation):
    energy_matrix = delay_and_sum(recordings, array, distance, fs, grid_azimuth, grid_elevation)
    est_az, est_el = estimate_angles(energy_matrix, grid_azimuth, grid_elevation)
    num_mics = array.shape[0]
    return est_az, est_el, num_mics

def _process_delay_and_sum_pack(args):
    recordings, array, distance, fs, grid_azimuth, grid_elevation = args
    return _process_delay_and_sum(recordings, array, distance, fs, grid_azimuth, grid_elevation)

def parsing():

    parser = argparse.ArgumentParser(description='GraphSSL - Graph-based Sound Source Localization')

    parser.add_argument('--resolution', type=int, default=1,
                        help="Resolution of the azimuth and elevation grid in degrees (default: 1).")
    parser.add_argument('--distance', type=float, default=1.0,
                        help="Distance between microphones in meters (default: 1.0).")
    parser.add_argument('--fs', type=int, default=48000,
                        help="Sampling frequency in Hz (default: 48000).")

    parser.add_argument('--num_workers', type=int, default=4, 
                        help="Number of workers for dataloader (default: 4).")
    parser.add_argument('--batch_size', type=int, default=1, 
                        help="Batch size for training (default: 1).")
    parser.add_argument('--n_jobs', type=int, default=max(1, (os.cpu_count() or 1) // 2),
                        help="Parallel workers for delay-and-sum evaluation (default: half of CPU cores). Use 1 to disable.")
    parser.add_argument('--parallel_backend', type=str, choices=['processes', 'threads', 'none'], default='processes',
                        help="Parallel backend for evaluation. 'processes' is recommended for CPU-bound NumPy/Python code.")
    parser.add_argument('--print_estimates', action='store_true', default=False,
                        help="Print per-sample estimated vs true angles (default: False).")
    
    parser.add_argument('--read_split_dataset', action='store_true', default=False,
                        help="Read already split dataset from files, or split it manually (default: False).")
    parser.add_argument('--validate', action='store_true', default=False,
                        help="Validate the model on a separate validation set (default: False).")
    
    parser.add_argument('--path', type=str, default='data',
                        help="Path to the dataset directory (default: data).")
    parser.add_argument('--wandb', action='store_true', default=False,
                        help="Use Weights & Biases for logging (default: False).")
    parser.add_argument('--save', action='store_true', default=False,
                        help="Save results to a .csv file (default: False)")
    parser.add_argument('--description', type=str, default = None,
                        help="Add context to results being saved (default: None)")
    
    parser.add_argument('--signal_processing', type=str, choices=['none', 'raw', 'fft', 'fft_magnitude', 'fft_phase', 'stft', 'stft_magnitude', 'stft_phase'], default='raw',
                        help="Signal processing method: [none, raw, fft, fft_magnitude, fft_phase, stft, stft_magnitude, stft_phase] (default: raw).")
    parser.add_argument('--edge_processing', type=str, choices=['none', 'distance', 'gcc_phat', 'gcc_phat_delay', 'cross_correlation', 'cross_correlation_delay', 'cross_power_spectrum', 'cross_spectral_matrix'], default='distance',
                        help="Edge processing method: [none, distance, gcc_phat, gcc_phat_delay, cross_correlation, cross_correlation_delay, cross_power_spectrum, cross_spectral_matrix] (default: distance).")
                        
    # TODO: Add dataset creation or reading option arguments
    args = parser.parse_args()

    for key, value in args.__dict__.items():
        print(f"{key}: {value}")

    return args

def main(args: argparse.Namespace = None):

    if args.wandb:
        wandb.init(project="GraphSSL", config=args)

    if args.validate:
        root = f'{args.path}/validation'
        signals_dir = f'{root}/signals'
        arrays_dir = f'{root}/arrays'
        angles_dir = f'{root}/sources'

        validation_dataset = InMemoryGraphDataset(root = root, signals_dir = signals_dir, arrays_dir = arrays_dir, angles_dir = angles_dir, signal_method= args.signal_processing, edge_method=args.edge_processing)
        # Avoid nested multiprocessing contention: if using a process pool below, keep DataLoader single-threaded here.
        loader_workers = 0 if (args.parallel_backend == 'processes' and args.n_jobs and args.n_jobs > 1) else args.num_workers
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=loader_workers)
        print(f"Validation size: {len(validation_dataset)}")
    
    else:
        if args.read_split_dataset:
            root = f'{args.path}/train'
            signals_dir = f'{root}/signals'
            arrays_dir = f'{root}/arrays'
            angles_dir = f'{root}/sources'

            train_dataset = InMemoryGraphDataset(root = root, signals_dir = signals_dir, arrays_dir = arrays_dir, angles_dir = angles_dir, signal_method= args.signal_processing, edge_method=args.edge_processing)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            root = f'{args.path}/test'
            signals_dir = f'{root}/signals'
            arrays_dir = f'{root}/arrays'
            angles_dir = f'{root}/sources'

            test_dataset = InMemoryGraphDataset(root = root, signals_dir = signals_dir, arrays_dir = arrays_dir, angles_dir = angles_dir, signal_method= args.signal_processing, edge_method=args.edge_processing)
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            num_samples = np.shape(train_dataset[0].x)[1]
            print(f"Number of samples: {num_samples}")

            num_mics = np.shape(train_dataset[0].x)[0]
            print(f"Number of microphones: {num_mics}")

            edge_dim = np.shape(train_dataset[0].edge_attr)[1]
            print(f"Edge dimension: {edge_dim}")

            print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

        else:
            root = f'{args.path}'
            signals_dir = f'{root}/signals'
            arrays_dir = f'{root}/arrays'
            angles_dir = f'{root}/sources'

            dataset = InMemoryGraphDataset(root = root, signals_dir = signals_dir, arrays_dir = arrays_dir, angles_dir = angles_dir, signal_method= args.signal_processing, edge_method=args.edge_processing)

            train_size = int(0.75 * len(dataset))
            test_size = len(dataset) - train_size

            print(f"Train size: {train_size}, Test size: {test_size}")
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

            train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
            test_dataloader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)

    estimated_azimuths = []
    estimated_elevations = []
    true_azimuths = []
    true_elevations = []
    total_mic_count = []

    # Precompute grids once (they were inside the loop before)
    grid_azimuth = np.arange(-90, 91, args.resolution)
    grid_elevation = np.arange(90, -91, -args.resolution)

    if args.validate:
        # Build a task generator so we don't have to materialize all samples in memory.
        def _task_iter():
            for data in validation_dataloader:
                recordings, array, angles = data.x.numpy(), data.array.numpy(), data.y.numpy()
                # Maintain existing true-angle bookkeeping for metric compatibility
                angles_deg = [tuple(np.rad2deg(angles[0]))]
                true_azimuths.append(np.deg2rad(angles_deg[0][0]))
                true_elevations.append(np.deg2rad(angles_deg[0][1]))
                yield (recordings, array, args.distance, args.fs, grid_azimuth, grid_elevation)

        if args.parallel_backend != 'none' and args.n_jobs and args.n_jobs > 1:
            Executor = ThreadPoolExecutor if args.parallel_backend == 'threads' else ProcessPoolExecutor
            with Executor(max_workers=args.n_jobs) as ex:
                results_iter = ex.map(_process_delay_and_sum_pack, _task_iter())
                for est_az, est_el, num_mics in tqdm(results_iter, total=len(validation_dataset), desc="Delay and Sum on Validation Data", leave=False):
                    estimated_azimuths.append(est_az)
                    estimated_elevations.append(est_el)
                    total_mic_count.append(num_mics)
        else:
            for t in tqdm(_task_iter(), total=len(validation_dataset), desc="Delay and Sum on Validation Data", leave=False):
                est_az, est_el, num_mics = _process_delay_and_sum(*t)
                estimated_azimuths.append(est_az)
                estimated_elevations.append(est_el)
                total_mic_count.append(num_mics)

        if args.print_estimates:
            for i in range(len(estimated_azimuths)):
                est_az, est_el = estimated_azimuths[i], estimated_elevations[i]
                true_az, true_el = true_azimuths[i], true_elevations[i]
                # tqdm.write(f"Estimated Angles: ({est_az}, {est_el}), True Angles (rad): ({true_az:.2f}, {true_el:.2f})")

    estimated_azimuths = np.array(estimated_azimuths)
    estimated_elevations = np.array(estimated_elevations)
    true_azimuths = np.array(true_azimuths)
    true_elevations = np.array(true_elevations)
    total_mic_count = np.array(total_mic_count)

    # Shape angles as (N, 2): columns = [azimuth, elevation]
    estimated_angles = np.column_stack((np.deg2rad(estimated_azimuths), np.deg2rad(estimated_elevations)))
    true_angles = np.column_stack((true_azimuths, true_elevations))
    maae, az_maae, el_maae = mean_absolute_angle_error(estimated_angles, true_angles)
    maae_per_array, az_maae_per_array, el_maae_per_array = maae_by_mic_count(estimated_angles, true_angles, total_mic_count)
    num_mics_per_array = sorted(list({int(c) for c in total_mic_count.tolist()}))

    print(f"Validation MAAE: {np.mean(maae):.5f}\nValidation Azimuth MAAE: {np.mean(az_maae):.5f} \nValidation Elevation MAAE: {np.mean(el_maae):.5f}")
    
    if args.wandb:
        wandb.log({"val_maae": np.mean(maae), "val_az_maae": np.mean(az_maae), "val_el_maae": np.mean(el_maae)})
        wandb.finish()

    if args.save:
        csv_dir = f"results/metrics/delay_and_sum_{args.description}_results.csv"
        try:
            os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
            row = {
                'maae': json.dumps(maae) if 'maae' in locals() else '',
                'az_maae': json.dumps(az_maae) if 'az_maae' in locals() else '',
                'el_maae': json.dumps(el_maae) if 'el_maae' in locals() else '',
                'maae_per_array': json.dumps(maae_per_array) if 'maae_per_array' in locals() else '',
                'az_maae_per_array': json.dumps(az_maae_per_array) if 'az_maae_per_array' in locals() else '',
                'el_maae_per_array': json.dumps(el_maae_per_array) if 'el_maae_per_array' in locals() else '',
                'num_mics_per_array': json.dumps(num_mics_per_array) if 'num_mics_per_array' in locals() else ''
            }
            with open(csv_dir, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            tqdm.write(f"Results CSV written to {csv_dir}")
        except Exception as e:
            tqdm.write(f"Failed to write results CSV: {e}")
        
if __name__ == '__main__':
    args = parsing()
    main(args)