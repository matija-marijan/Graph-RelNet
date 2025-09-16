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

criterion = torch.nn.MSELoss()

models_dir = os.path.join(os.path.dirname(__file__), 'models')
all_models = [
    os.path.splitext(f)[0]
    for f in os.listdir(models_dir)
    if f.endswith('.py') and not f.startswith('__')
]

def train(model: torch.nn.Module, device: str, dataloader: DataLoader, optimizer: torch.optim, scheduler: torch.optim = None, epoch: int = None, log: bool = 0):

    model.train()
    total_loss = 0

    for batch_idx, data in enumerate(tqdm(dataloader, desc=f"Training epoch {epoch}", position=1, leave=False)):
        
        data = data.to(device)
        output = model(data)
        y = data.y.float().to(device)
        # print(f"Output shape: {output.shape}, Target shape: {y.shape}")   # [batch_size, 2]
        # for o, t in zip(output, y):
        #     print(f"Output: {o.item():.3f}, Target: {t.item():.3f}")

        # Linear array (no elevation)
        # output = output[:, 0]
        # y = y[:, 0]

        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()

        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad norm: {param.grad.norm().item()}")
        #     else:
        #         print(f"{name} has no gradient")
        
        optimizer.step()

        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)

    if scheduler is not None:
        scheduler.step()

    if log:
        wandb.log({"loss": avg_loss}, commit=False)
    tqdm.write(f"\nEpoch {epoch} loss: {avg_loss:.5f}")    

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
    parser.add_argument('--read_split_dataset', action='store_true', default=False,
                        help="Read already split dataset from files, or split it manually (default: False).")
    parser.add_argument('--validate', action='store_true', default=False,
                        help="Validate the model on a separate validation set (default: False).")
    parser.add_argument('--path', type=str, default='data',
                        help="Path to the dataset directory (default: data).")
    parser.add_argument('--runtime_dataset', action='store_true', default=False,
                        help="Use a runtime dataset instead of a static one (default: False).")

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

        if args.runtime_dataset:
            pre_transform = OutputSimulationTransform()
            transform = OutputSimulationTransform(signal_method='raw', edge_method = 'gcc_phat_delay')
            dataset = RuntimeGraphDataset(root = root, signals_dir = signals_dir, arrays_dir = arrays_dir, angles_dir = angles_dir, transform = transform, pre_transform = None)
        else:
            dataset = InMemoryGraphDataset(root = root, signals_dir = signals_dir, arrays_dir = arrays_dir, angles_dir = angles_dir, signal_method= args.signal_processing, edge_method=args.edge_processing)
        
        num_samples = np.shape(dataset[0].x)[1]
        print(f"Number of samples: {num_samples}")

        num_mics = np.shape(dataset[0].x)[0]
        print(f"Number of microphones: {num_mics}")

        edge_dim = np.shape(dataset[0].edge_attr)[1]
        print(f"Edge dimension: {edge_dim}")

        train_size = int(0.75 * len(dataset))
        test_size = len(dataset) - train_size

        print(f"Train size: {train_size}, Test size: {test_size}")
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        test_dataloader = DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    
    if args.validate:
        root = f'{args.path}/validation'
        signals_dir = f'{root}/signals'
        arrays_dir = f'{root}/arrays'
        angles_dir = f'{root}/sources'

        validation_dataset = InMemoryGraphDataset(root = root, signals_dir = signals_dir, arrays_dir = arrays_dir, angles_dir = angles_dir, signal_method= args.signal_processing, edge_method=args.edge_processing)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        print(f"Validation size: {len(validation_dataset)}")

    # Model initialization

    model_name = args.model
    
    if model_name == 'Graph_RelNet':
        model = Graph_RelNet(edge_dim = edge_dim)
    
    elif model_name == 'RelNet':
        model = RelNet(edge_dim = edge_dim)
                
    else:
        raise ValueError(f"Model {model_name} is not supported. Please choose from: {', '.join(all_models)}.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = None
    if args.scheduler:
        lambda_lr = lambda epoch: 0.75 ** (epoch // 25)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda_lr)

    # print("Training...")

    best_test_maae = np.inf
    for epoch in tqdm(range(args.num_epochs), desc="Epoch", position = 0, leave = False):

        # print(f"Epoch: {epoch}")
        # tqdm.write(f"Epoch: {epoch}")
        train(model = model, device = device, dataloader = train_dataloader, optimizer = optimizer, scheduler = scheduler, epoch = epoch, log = args.wandb)
        # current_lr = optimizer.param_groups[0]['lr']
        # tqdm.write(f"Current LR: {current_lr:.6f}")

        g_train, p_train, _ = predict(model = model, device = device, dataloader = train_dataloader)
        # print(f"Ground train shape: {g_train.shape}, Predictions train shape: {p_train.shape}")

        # train_mse = criterion(torch.tensor(p_train), torch.tensor(g_train))
        train_maae, train_az_maae, train_el_maae = mean_absolute_angle_error(p_train, g_train)
        tqdm.write(f"Epoch {epoch} Train MAAE: {float(train_maae):.5f}")

        g_test, p_test, _ = predict(model = model, device = device, dataloader = test_dataloader)
        # print(f"Ground truth shape: {G.shape}, Predictions shape: {P.shape}")

        # mse = criterion(torch.tensor(p_test), torch.tensor(g_test))
        test_maae, test_az_maae, test_el_maae = mean_absolute_angle_error(p_test, g_test)
        tqdm.write(f"Epoch {epoch} Test MAAE: {float(test_maae):.5f}")

        if args.wandb:
            wandb.log({"test_maae": float(test_maae), "train_maae": float(train_maae)})

        if args.plot != 0 and epoch % args.plot == 0 and epoch != 0:
            plot_source(g_train, p_train, description = f"{desc}_train", maae = train_maae, az_maae = train_az_maae, el_maae = train_el_maae)
            plot_source(g_test, p_test, description = f"{desc}_test", maae = test_maae, az_maae = test_az_maae, el_maae = test_el_maae)

        if test_maae < best_test_maae:
            best_test_maae = test_maae
            best_test_az_maae = test_az_maae
            best_test_el_maae = test_el_maae
            best_model = model
            if args.save:
                torch.save(best_model.state_dict(), f"results/models/{desc}.pth")
                plot_source(g_train, p_train, plot = False, save = True, description = f"{desc}_train", maae = train_maae, az_maae = train_az_maae, el_maae = train_el_maae)
                plot_source(g_test, p_test, plot = False, save = True, description = f"{desc}_test", maae = test_maae, az_maae = test_az_maae, el_maae = test_el_maae)

    tqdm.write(f"Best test MAAE: {best_test_maae:.5f}")
    tqdm.write(f"Best test azimuth MAAE: {best_test_az_maae:.5f}")
    tqdm.write(f"Best test elevation MAAE: {best_test_el_maae:.5f}")

    if args.plot != 0:
        plot_source(g_train, p_train, description = f"{desc}_train", maae = train_maae, az_maae = train_az_maae, el_maae = train_el_maae)
        plot_source(g_test, p_test, description = f"{desc}_test", maae = test_maae, az_maae = test_az_maae, el_maae = test_el_maae)

    if args.validate:
        # best_model = model.load_state_dict(best_model)
        g_val, p_val, mic_counts_val = predict(model = best_model, device = device, dataloader = validation_dataloader)
        val_maae, val_az_maae, val_el_maae = mean_absolute_angle_error(p_val, g_val)
        tqdm.write(f"Validation MAAE: {float(val_maae):.5f}")
        tqdm.write(f"Validation azimuth MAAE: {float(val_az_maae):.5f}")
        tqdm.write(f"Validation elevation MAAE: {float(val_el_maae):.5f}")

        # Per-array metrics (group by number of microphones)
        val_maae_per_array, val_az_maae_per_array, val_el_maae_per_array = maae_by_mic_count(p_val, g_val, mic_counts_val)
        num_mics_per_array = sorted(list({int(c) for c in mic_counts_val.tolist()}))

        if args.wandb:
            wandb.log({"val_maae": float(val_maae), "val_az_maae": float(val_az_maae), "val_el_maae": float(val_el_maae)})

        if args.plot != 0:
            plot_source(g_val, p_val, description = f"{desc}_val", maae = val_maae, az_maae = val_az_maae, el_maae = val_el_maae)

        if args.save:
            plot_source(g_val, p_val, plot = False, save = True, description = f"{desc}_val", maae = val_maae, az_maae = val_az_maae, el_maae = val_el_maae)

    if args.save:
        csv_dir = f"results/metrics/{desc}_results.csv"
        # Write results row to CSV
        try:
            os.makedirs(os.path.dirname(csv_dir), exist_ok=True)
            row = {
                'best_test_maae': float(best_test_maae.item()) if hasattr(best_test_maae, 'item') else float(best_test_maae),
                'best_test_az_maae': (float(best_test_az_maae.item()) if 'best_test_az_maae' in locals() and hasattr(best_test_az_maae, 'item') else (float(best_test_az_maae) if 'best_test_az_maae' in locals() else '')),
                'best_test_el_maae': (float(best_test_el_maae.item()) if 'best_test_el_maae' in locals() and hasattr(best_test_el_maae, 'item') else (float(best_test_el_maae) if 'best_test_el_maae' in locals() else '')),
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

if __name__ == '__main__':
    args = parsing()
    main(args)