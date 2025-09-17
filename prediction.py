import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
from utils.dataset_utils import InMemoryGraphDataset, RuntimeGraphDataset, OutputSimulationTransform
from utils.utils import plot_source, mean_absolute_angle_error, maae_by_mic_count

from models.RelNet import RelNet
from models.Graph_RelNet import Graph_RelNet

import argparse
import os
import csv
import json
import random
import numpy as np

import wandb
from tqdm import tqdm

models_dir = os.path.join(os.path.dirname(__file__), 'models')
all_models = [
    os.path.splitext(f)[0]
    for f in os.listdir(models_dir)
    if f.endswith('.py') and not f.startswith('__')
]

def predict(model: torch.nn.Module, device: str, dataloader: DataLoader):

    model.eval()
    predictions = []
    labels = []
    mic_counts = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Prediction", position=1, leave=False):
            data = data.to(device)
            output = model(data)
            y = data.y.float().to(device)

            predictions.append(output.detach().cpu())
            labels.append(y.detach().cpu())

            # Determine number of microphones per graph in batch
            if hasattr(data, 'batch') and data.batch is not None:
                # batch is [num_nodes] with graph ids
                counts = torch.bincount(data.batch, minlength=int(data.num_graphs)).cpu()
            else:
                # fallback
                if hasattr(data, 'array') and data.array is not None:
                    counts = torch.tensor([data.array.size(0)])
                else:
                    counts = torch.tensor([data.x.size(0)])

            mic_counts.append(counts)

    predictions = torch.cat(predictions, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    mic_counts = torch.cat(mic_counts, dim=0).numpy()

    return labels, predictions, mic_counts

def parsing():

    parser = argparse.ArgumentParser(description='GraphSSL - Graph-based Sound Source Localization')

    parser.add_argument('--seed', type=int, default = None,
                        help="Random seed for reproducibility (default: None).")
    parser.add_argument('--batch_size', type=int, default=128, 
                        help="Batch size for training (default: 128).")
    parser.add_argument('--num_workers', type=int, default=4, 
                        help="Number of workers for dataloader (default: 4).")
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help="Number of epochs for training (default: 100).")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for optimizer (default: 0.001).")
    parser.add_argument('--scheduler', action='store_true', default=False,
                        help="Use a learning rate scheduler (default: False).")
    
    parser.add_argument('--wandb', action='store_true', default=False,
                        help="Use Weights & Biases for logging (default: False).")
    parser.add_argument('--plot', type=int, default=0,
                        help="Interval for plotting results during training (default: 0 - no plotting).")
    parser.add_argument('--save', action='store_true', default=False,
                        help="Save the best model and results based on test loss (default: False).")
    parser.add_argument('--description', type=str, default = None,
                        help="Add context to results being saved or plotted (default: None)")
    
    parser.add_argument('--model', type=str, choices=all_models, default='Graph_RelNet',
                        help="Model to use for training. Default is 'Graph_RelNet'. Choose from: " + ", ".join(all_models) + ".")
    parser.add_argument('--dataset_path', type=str, default='data/validation',
                        help="Path to the dataset directory (default: data).")
    parser.add_argument('--runtime_dataset', action='store_true', default=False,
                        help="Use a runtime dataset instead of a static one (default: False).")

    parser.add_argument('--signal_processing', type=str, choices=['none', 'raw', 'fft', 'fft_magnitude', 'fft_phase', 'stft', 'stft_magnitude', 'stft_phase'], default='raw',
                        help="Signal processing method: [none, raw, fft, fft_magnitude, fft_phase, stft, stft_magnitude, stft_phase] (default: raw).")
    parser.add_argument('--edge_processing', type=str, choices=['none', 'distance', 'gcc_phat', 'gcc_phat_delay', 'cross_correlation', 'cross_correlation_delay', 'cross_power_spectrum', 'cross_spectral_matrix'], default='distance',
                        help="Edge processing method: [none, distance, gcc_phat, gcc_phat_delay, cross_correlation, cross_correlation_delay, cross_power_spectrum, cross_spectral_matrix] (default: distance).")

    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    return args

def main(args: argparse.Namespace):

    # Set seed:
    if args.seed is not None:
        seed = args.seed
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    desc = f"{args.model}_seed_{args.seed}"
    if args.description is not None:
        desc += f"_{args.description}"

    if args.wandb:
        wandb.init(project="GraphSSL", config=args, group = f"{args.model}_{args.description}" if args.description is not None else f"{args.model}", name = desc)

    signals_dir = f"{args.dataset_path}/signals"
    arrays_dir = f"{args.dataset_path}/arrays"
    angles_dir = f"{args.dataset_path}/sources"

    dataset = InMemoryGraphDataset(root = args.dataset_path, signals_dir = signals_dir, arrays_dir = arrays_dir, angles_dir = angles_dir,
                                    signal_method = args.signal_processing, edge_method = args.edge_processing)
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_samples = np.shape(dataset[0].x)[1]
    print(f"Number of samples: {num_samples}")

    num_mics = np.shape(dataset[0].x)[0]
    print(f"Number of microphones: {num_mics}")

    edge_dim = np.shape(dataset[0].edge_attr)[1]
    print(f"Edge dimension: {edge_dim}")

    model_name = args.model
    
    if model_name == 'Graph_RelNet':
        model = Graph_RelNet(edge_dim = edge_dim)
    
    elif model_name == 'RelNet':
        model = RelNet(edge_dim = edge_dim)
        
    else:
        raise ValueError(f"Model {model_name} is not supported. Please choose from: {', '.join(all_models)}.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    checkpoint_path = f"results/models/{desc}.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)

    g_val, p_val, mic_counts_val = predict(model = model, device = device, dataloader = dataloader)
    val_maae, val_az_maae, val_el_maae = mean_absolute_angle_error(g_val, p_val)
    tqdm.write(f"Validation MAAE: {float(val_maae):.5f}")
    tqdm.write(f"Validation azimuth MAAE: {float(val_az_maae):.5f}")
    tqdm.write(f"Validation elevation MAAE: {float(val_el_maae):.5f}")

    # Per-array metrics (group by number of microphones)
    val_maae_per_array, val_az_maae_per_array, val_el_maae_per_array = maae_by_mic_count(p_val, g_val, mic_counts_val)
    num_mics_per_array = sorted(list({int(c) for c in mic_counts_val.tolist()}))
    
    if args.wandb:
        wandb.log({"val_maae": float(val_maae), "val_az_maae": float(val_az_maae), "val_el_maae": float(val_el_maae)})

    if args.plot:
        plot_source(g_val, p_val, desc = desc, save = args.save, description = f"{args.description}_val" if args.description else "val")

    if args.save:
        plot_source(g_val, p_val, plot = False, save = True, description = f"{desc}_val", maae = val_maae, az_maae = val_az_maae, el_maae = val_el_maae)

    if args.save:
        csv_dir = f"results/metrics/{desc}_results_val.csv"
        # Write results row to CSV
        try:
            os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
            row = {
                'val_maae': float(val_maae.item()) if 'val_maae' in locals() and hasattr(val_maae, 'item') else (float(val_maae) if 'val_maae' in locals() else ''),
                'val_az_maae': float(val_az_maae.item()) if 'val_az_maae' in locals() and hasattr(val_az_maae, 'item') else (float(val_az_maae) if 'val_az_maae' in locals() else ''),
                'val_el_maae': float(val_el_maae.item()) if 'val_el_maae' in locals() and hasattr(val_el_maae, 'item') else (float(val_el_maae) if 'val_el_maae' in locals() else ''),
                'num_mics_per_array': json.dumps(num_mics_per_array) if 'num_mics_per_array' in locals() else '',
                'val_maae_per_array': json.dumps(val_maae_per_array) if 'val_maae_per_array' in locals() else '',
                'val_az_maae_per_array': json.dumps(val_az_maae_per_array) if 'val_az_maae_per_array' in locals() else '',
                'val_el_maae_per_array': json.dumps(val_el_maae_per_array) if 'val_el_maae_per_array' in locals() else ''
            }
            with open(csv_dir, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            tqdm.write(f"Results CSV written to {csv_dir}")
        except Exception as e:
            tqdm.write(f"Failed to write results CSV: {e}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    args = parsing()
    main(args)