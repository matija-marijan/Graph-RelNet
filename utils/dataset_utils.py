import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset as TorchDataset
import torch_geometric.transforms as T
import numpy as np
from utils.utils import calculate_distances, gcc_phat, time_output_calculation, wavelength_normalization, frequency_output_calculation
import os
import pandas as pd
from tqdm import tqdm

from scipy.signal import stft
import matplotlib.pyplot as plt

def signal_processing(signals: np.ndarray, method: str = 'raw'):
    """
    Process audio signals based on the specified method.

    Parameters:
    signals (numpy.ndarray): A 2D array where each row represents an audio signal from a microphone.
    method (str): The method to use for processing. Options are:
        - 'none': Return zeros of the same shape.
        - 'raw': Return raw waveforms.
        - 'fft': Return normalized FFT.
        - 'fft_magnitude': Return normalized FFT magnitudes.
        - 'fft_phase': Return normalized phase from FFT.
        - 'stft': Return normalized STFT spectrograms.
        - 'stft_magnitude': Return normalized STFT magnitude spectrograms.
        - 'stft_phase': Return normalized phase from STFT.

    Returns:
    numpy.ndarray: Processed signals.
    """

    if method == 'none' or method is None:
        return np.zeros_like(signals)
    
    elif method == 'raw':
        return signals
    
    elif method.startswith('fft'):
        len_signal = signals.shape[1]
        n_fft = 2 ** (int(np.floor(np.log2(len_signal))))
        fft = np.fft.rfft(signals, axis=1, n=n_fft)

        if method == 'fft_magnitude':
            fft_signals = np.abs(fft)
        elif method == 'fft_phase':
            fft_signals = np.angle(fft)
        else:  # 'fft' or any other method
            fft_signals = fft

        # Normalize the FFT signals
        fft_signals = (fft_signals - np.mean(fft_signals, axis=1, keepdims=True)) / np.std(fft_signals, axis=1, keepdims=True)

        return fft_signals

    elif method.startswith('stft'):
        len_signal = signals.shape[1]
        n_fft = 2 ** (int(np.floor(np.log2(len_signal))))

        stft_spectrogram_stack = []
        for signal in signals:
            # nperseg = 64,  noverlap = 32  -> [ 33,  33] STFT
            # nperseg = 128, noverlap = 112 -> [ 65,  65] STFT
            # nperseg = 256, noverlap = 248 -> [129, 129] STFT
            # nperseg = 512, noverlap = 508 -> [257, 257] STFT

            f, t, Zxx = stft(x=signal, nperseg=256, noverlap=248)

            if method == 'stft_magnitude':
                stft_spectrogram = np.abs(Zxx)
            elif method == 'stft_phase':
                stft_spectrogram = np.angle(Zxx)
            else:  # 'stft' or any other method
                stft_spectrogram = Zxx

            # Normalize each spectrogram
            stft_spectrogram = (stft_spectrogram - np.mean(stft_spectrogram)) / (np.std(stft_spectrogram) + 1e-8)

            stft_spectrogram_stack.append(stft_spectrogram)
        stft_spectrogram_stack = np.stack(stft_spectrogram_stack)

        return stft_spectrogram_stack
    
    else:
        raise ValueError("Invalid method. Use 'raw', 'none', 'fft', 'fft_magnitude', 'fft_phase', 'stft', 'stft_magnitude', or 'stft_phase'.")
    
def edge_processing(mic_array: np.ndarray, signals: np.ndarray = None, method: str = 'distance'):
    """
    Process edges based on the specified method.

    Parameters:
    mic_array (numpy.ndarray): 2D array where each row is the 3D coordinates of a microphone.
    signals (numpy.ndarray, optional): 2D array where each row is the audio signal from a microphone. Required for some edge methods.
    method (str): Method for edge attributes. Options:
        - 'none': All edge weights set to 1.
        - 'distance': Pairwise Euclidean distances between microphones.
        - 'gcc_phat': Generalized cross-correlation with phase transform (requires signals).
        - 'gcc_phat_delay': Argmax delay from GCC-PHAT method (requires signals).
        - 'cross_correlation': Full cross-correlation vector as edge attribute (requires signals).
        - 'cross_correlation_delay': Maximum cross-correlation lag (requires signals).        
        - 'cross_power_spectrum': Normalized cross power spectrum as edge attribute (requires signals).
        - 'cross_spectral_matrix': Cross spectral matrix (requires signals).

    Returns:
    tuple: (edge_weights, directed, ndim)
        - edge_weights: Array or matrix of edge attributes.
        - directed (bool): Whether the resulting graph is directed.
        - ndim (bool): Whether edge attributes are multidimensional (True for vector-valued attributes).
    """
    num_mics = mic_array.shape[0]

    if method == 'none' or method is None:
        directed = False
        ndim = False

        edge_weights = np.ones((num_mics, num_mics))

    elif method == 'distance':
        directed = False
        ndim = False

        edge_weights = calculate_distances(mic_array)

    elif method == 'gcc_phat' and signals is not None:
        directed = True
        ndim = True

        edge_weights = np.empty((num_mics, num_mics), dtype=object)

        for i in range(num_mics):
            for j in range(num_mics):
                if i != j:
                    _, gcc = gcc_phat(signals[i], signals[j], max_shift=200, interp=1)
                    edge_weights[i, j] = gcc
                    
    elif method == 'gcc_phat_delay' and signals is not None:
        directed = True
        ndim = False

        delays = np.zeros((num_mics, num_mics))
        for i in range(num_mics):
            for j in range(i + 1, num_mics):
                delay, _ = gcc_phat(signals[i], signals[j], interp=16)
                delays[i, j] = delay
                delays[j, i] = -delay

        # Normalize delays
        num_samples = signals.shape[1]
        # edge_weights = delays / num_samples * 2
        # max_delay = max_diameter / c * sampling_frequency 
        edge_weights = delays / np.max(np.abs(delays))
    
    elif method == 'cross_correlation' and signals is not None:
        directed = True
        ndim = True

        edge_weights = np.empty((num_mics, num_mics), dtype=object)

        for i in range(num_mics):
            for j in range(num_mics):
                if i != j:
                    corr = np.correlate(signals[i], signals[j], mode='full')
                    norm = np.linalg.norm(signals[i] - np.mean(signals[i])) * np.linalg.norm(signals[j] - np.mean(signals[j])) 
                    corr = corr / norm if norm != 0 else corr  # Avoid division by zero

                    edge_weights[i, j] = corr

    elif method == 'cross_correlation_delay' and signals is not None:
        directed = True
        ndim = False

        delays = np.zeros((num_mics, num_mics))
        for i in range(num_mics):
            for j in range(i + 1, num_mics):
                corr = np.correlate(signals[i], signals[j], mode='full')
                delays[i, j] = np.argmax(corr) - (len(signals[i]) - 1)
                delays[j, i] = -delays[i, j]

        edge_weights = delays / np.max(np.abs(delays))

    elif method == 'cross_power_spectrum' and signals is not None:
        directed = True
        ndim = True

        edge_weights = np.empty((num_mics, num_mics), dtype=object)
        
        for i in range(num_mics):
            for j in range(num_mics):
                if i != j:
                    n = signals[i].shape[0] + signals[j].shape[0]
                    sig = np.fft.rfft(signals[i], n=n)
                    refsig = np.fft.rfft(signals[j], n=n)
                    cps = sig * np.conj(refsig)

                    cps = cps / np.max(np.abs(cps))  # Normalize CPS

                    edge_weights[i, j] = cps

    elif method == 'cross_spectral_matrix' and signals is not None:
        directed = False
        ndim = False

        signals_fft = np.fft.rfft(signals, axis=1)
        csm = np.matmul(signals_fft, np.conjugate(signals_fft.T))
        edge_weights = csm

    else:
        raise ValueError("Invalid method or missing signals for the specified method.")

    return edge_weights, directed, ndim

def array_to_graph(mic_array: np.ndarray, signals: np.ndarray = None, method: str = 'distance'):
    """
    Convert a microphone array to a graph representation.

    Parameters:
    mic_array (numpy.ndarray): 2D array where each row is the 3D coordinates of a microphone.
    signals (numpy.ndarray, optional): 2D array where each row is the audio signal from a microphone. Required for some edge methods.
    method (str): Method for edge attributes. Options:
        - 'none': All edge weights set to 1.
        - 'distance': Pairwise Euclidean distances between microphones.
        - 'gcc_phat': Generalized cross-correlation with phase transform (requires signals).
        - 'gcc_phat_delay': Argmax delay from GCC-PHAT method (requires signals).
        - 'cross_correlation': Full cross-correlation vector as edge attribute (requires signals).
        - 'cross_correlation_delay': Maximum cross-correlation lag (requires signals).        
        - 'cross_power_spectrum': Normalized cross power spectrum as edge attribute (requires signals).
        - 'cross_spectral_matrix': Cross spectral matrix (requires signals).

    Returns:
    tuple: (edge_index, edge_attr)
        - edge_index: 2D array of shape (2, num_edges) representing the edges in the graph.
        - edge_attr: 1D (num_edges, ) or 2D (num_edges, num_features) array of edge attributes corresponding to the edges in edge_index.
    """

    num_mics = mic_array.shape[0]

    edge_weights, directed, ndims = edge_processing(mic_array, signals, method)

    if directed:
        edge_index = np.array([[i, j] for i in range(num_mics) for j in range(num_mics) if i != j]).T
    else:
        edge_index = np.array([[i, j] for i in range(num_mics) for j in range(i + 1, num_mics)]).T

    if ndims:
        edge_attr = []
        for i in range(edge_weights.shape[0]):
            for j in range(edge_weights.shape[1]):
                if i != j:
                    edge_attr.append(edge_weights[i, j])
        edge_attr = np.array(edge_attr)
    else:
        edge_attr = edge_weights[edge_index[0], edge_index[1]].reshape(-1, 1)

    return edge_index, edge_attr

class InMemoryGraphDataset(InMemoryDataset):
    """
    A dataset class for loading and processing microphone array data into a graph format.
    This class inherits from InMemoryDataset and processes the data into a format suitable for graph neural
    networks, with options for different signal processing methods and edge attribute calculations.
    """
    
    def __init__(self, root: str, signals_dir: str, arrays_dir: str, angles_dir: str, signal_method: str = 'raw', edge_method: str = 'distance'):

        self.signal_method = signal_method
        self.edge_method = edge_method
        
        self.signals_dir = signals_dir
        self.arrays_dir = arrays_dir
        self.angles_dir = angles_dir
        self._length = None
        
        super().__init__(root, transform = None, pre_transform = None)

        # self.data, self.slices = torch.load(self.processed_paths[0])

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'signals_{i}.csv' for i in range(self.len())]
    
    @property
    def processed_file_names(self):
        return [f"InMemoryGraphDataset_{self.signal_method}_{self.edge_method}.pt"]
    
    def download(self):
        pass

    def _download(self):
        pass

    def len(self):
        if self._length is None:
            self._length = len([f for f in os.listdir(self.signals_dir) if f.endswith('.csv')])
        return self._length

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        # self.process()

    def process(self):
        data_list = []
        # print(len(self))

        signal_files = sorted([f for f in os.listdir(self.signals_dir) if f.endswith('.csv')])
        for signal_file in tqdm(signal_files, desc="Processing dataset"):

            # Extract the base name (without extension) to match array and angle files
            base_name = os.path.splitext(signal_file)[0]

            signals_path = os.path.join(self.signals_dir, signal_file)
            raw_signals = pd.read_csv(signals_path, header=None).values
            # print(type(signals))

            signals = signal_processing(raw_signals, method=self.signal_method)
            # print(f"Processed signals shape: {signals.shape}")

            array_path = os.path.join(self.arrays_dir, f'array_{base_name.split("_", 1)[-1]}.csv')
            mic_array = pd.read_csv(array_path, header=None).values

            target_path = os.path.join(self.angles_dir, f'angles_{base_name.split("_", 1)[-1]}.csv')
            target = pd.read_csv(target_path, header=None).values

            edge_index, edge_attr = array_to_graph(mic_array, raw_signals, method=self.edge_method)

            data = Data(x = torch.tensor(signals, dtype=torch.float), 
                        edge_index = torch.tensor(edge_index, dtype=torch.long), 
                        edge_attr = torch.tensor(edge_attr, dtype=torch.float), 
                        y = torch.tensor(target, dtype=torch.float),
                        array = torch.tensor(mic_array, dtype=torch.float))
            
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), os.path.join(self.processed_dir, f"InMemoryGraphDataset_{self.signal_method}_{self.edge_method}.pt"))

class RuntimeGraphDataset(InMemoryDataset):
    """
    A dataset class for creating and storing microphone array geometries, source positions and source signals.
    The class supports runtime processing of microphone signals, and simulating the recordings.
    This class inherits from InMemoryDataset and processes the data into a format suitable for graph neural
    networks, with options for different signal processing methods and edge attribute calculations.
    """
    
    def __init__(self, root: str, signals_dir: str, arrays_dir: str, angles_dir: str, transform = None, pre_transform = None):
        
        self.signals_dir = signals_dir
        self.arrays_dir = arrays_dir
        self.angles_dir = angles_dir
        self._length = None

        self.signals = None
        self.angles = None
        self.arrays = None
        
        super().__init__(root, transform = transform, pre_transform = pre_transform)

        # self.data, self.slices = torch.load(self.processed_paths[0])

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'signals_{i}.csv' for i in range(self.len())]
    
    @property
    def processed_file_names(self):
        return [f"RuntimeGraphDataset_{self.pre_transform.get_name()}.pt" if self.pre_transform else "RuntimeGraphDataset.pt"]

    def download(self):
        pass

    def _download(self):
        pass

    def len(self):
        if self._length is None:
            self._length = len([f for f in os.listdir(self.signals_dir) if f.endswith('.csv')])
        return self._length

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        # self.process()

    def process(self):
        data_list = []

        signal_files = sorted([f for f in os.listdir(self.signals_dir) if f.endswith('.csv')])
        for signal_file in tqdm(signal_files, desc="Processing dataset"):

            # Extract the base name (without extension) to match array and angle files
            base_name = os.path.splitext(signal_file)[0]

            signals_path = os.path.join(self.signals_dir, signal_file)
            signals = pd.read_csv(signals_path, header=None).values

            array_path = os.path.join(self.arrays_dir, f'array_{base_name.split("_", 1)[-1]}.csv')
            mic_array = pd.read_csv(array_path, header=None).values

            target_path = os.path.join(self.angles_dir, f'angles_{base_name.split("_", 1)[-1]}.csv')
            target = pd.read_csv(target_path, header=None).values

            edge_index, edge_attr = array_to_graph(mic_array, method=None)

            data = Data(x = torch.tensor(signals, dtype=torch.float), 
                        edge_index = torch.tensor(edge_index, dtype=torch.long), 
                        edge_attr = torch.tensor(edge_attr, dtype=torch.float), 
                        y = torch.tensor(target, dtype=torch.float),
                        array = torch.tensor(mic_array, dtype=torch.float))
            
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        torch.save((data, slices), os.path.join(self.processed_dir, f"RuntimeGraphDataset_{self.pre_transform.get_name()}.pt"))     # Save pre_transform_name!

class OutputSimulationTransform(T.BaseTransform):
    def __init__(self, signal_method: str = 'raw', edge_method: str = 'distance', fs: float = 48000, SNR: float = 100):
        super().__init__()
        self.signal_method = signal_method
        self.edge_method = edge_method
        self.fs = fs
        self.SNR = SNR

    def __call__(self, data: Data):
        signals, edge_index, edge_attr, array, angles, batch = data.x, data.edge_index, data.edge_attr, data.array, data.y, data.batch
        
        angles = np.rad2deg(angles)        
        # norm_array, q1 = wavelength_normalization(array, self.fs)
        # output = frequency_output_calculation(signals[0].numpy(), angles[0].numpy(), norm_array.numpy(), q1, save=False, SNR=self.SNR)
        output = time_output_calculation(signals[0].numpy(), angles[0].numpy(), array.numpy(), self.SNR, self.fs)
        
        data.x = torch.tensor(output, dtype=torch.float)

        return data
    
    def get_name(self):
        return f"OutputSimulationTransform_{self.signal_method}_{self.edge_method}_{self.fs}_{self.SNR}"

class AddNoiseTransform(T.BaseTransform):
    pass

class DropoutMicsTransform(T.BaseTransform):
    pass

class AugmentArrayTransform(T.BaseTransform):
    pass

if __name__ == "__main__":

    root = 'data/runtime'
    signals_dir = 'data/runtime/signals'
    arrays_dir = 'data/runtime/arrays'
    angles_dir = 'data/runtime/sources'

    pre_transform = OutputSimulationTransform()

    train_dataset = RuntimeGraphDataset(root = root, signals_dir = signals_dir, arrays_dir = arrays_dir, angles_dir = angles_dir, pre_transform = pre_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)