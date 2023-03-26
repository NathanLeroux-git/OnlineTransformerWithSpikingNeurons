### data_loader #### 
from scipy.io import loadmat
import torch
import torch.utils.data as data_utils
import numpy as np
from snntorch import spikegen
from scipy.signal import spectrogram
from utils import download_dataset
import os

def _load_ninapro8_emg_windows(args, times, dataset_location='../datasets/ninapro8_dataset/'):
    """ Loading the raw sEMG dataset and apllying a few transformations:
    There are the options of normalize the data, apply a short time fourrier transform, or convert the signal into events (spikes)

    RETURNS
    -------
    [signal_list, DOA_list], start_idx_list
        - signal_list: list[subject][time] (input data to classify)
            Elements of the list : torch.tensor of shape (n_channels=16, n_time_steps)
        - DOA_list: list[subject][time] (targets)
            Elements of the list : torch.tensor of shape (n_doa=5, n_time_steps)
        - start_idx_list: list[subject][time] (indexes of the data time windows)
            Elements of the list : torch.long of shape (n_windows)
    """
    subjects = [args.subjects] # The backbone of this code is designed to have the option of training on multiple subjects, even though we train only for one subject at the time here.

    signal_list = []
    DOA_list = []
    start_idx_list = []
    # Collecting data from different subjects, and different "times". "times" means the different recording sessions, 0 and 1 are used for training and 2 for testing.
    for sub_count, subject in enumerate(subjects):
        signal_time = []
        DOA_time = []
        start_idx_time = []
        for time_count, time in enumerate(times):       
            print(f'loading subject={subject}\ttime={time}...')         
            name = f'{dataset_location}/S{subject+1:.0f}_E1_A{time+1:.0f}.mat'   
            if not(os.path.exists(name)):        
                download_dataset(f'S{subject+1:.0f}_E1_A{time+1:.0f}.mat')
            signal = torch.HalfTensor(loadmat(name)['emg']) # input
            DOF = torch.Tensor(loadmat(name)['glove']) # target
            # Converting the 18 channels degrees of freedom into 5 degrees of actuation through a matrix multiplicaton
            DOA = DOF_to_DOA(DOF)
            # signal pre-processing (optional)
            signal = normalize_data(signal) if args.normalize_data else signal                
            if args.spectrogram:
                signal, DOA = short_term_fourrier_transform(signal, DOA, args.spectrogram_config)   
                signal = normalize_data(signal) if args.normalize_data else signal
            signal = to_spike_events(signal, threshold=args.delta_spike_threshold, off_spike=args.off_spike) if args.convert_raw_data_to_spikes else signal
            n_channels = signal.shape[1]
            signal = torch.permute(signal, (1,0)) # convert (time, channel) to (channel, time)     
            DOA = torch.permute(DOA, (1,0)) # convert (time, DOA) to (DOA, time)   
            # Spliting very long signals of several minutes into smaller time windows (typically 1 second). Here we just need the index of start of eac time window. 
            start_idx = transformation_to_small_windows(signal, DOA, args.window_size, args.sliding_size)
            signal_time.append(signal)
            DOA_time.append(DOA)
            start_idx_time.append(start_idx)
    signal_list.append(signal_time)
    DOA_list.append(DOA_time) 
    start_idx_list.append(start_idx_time)
    return [signal_list, DOA_list], start_idx_list

class dataset_generator_continuous(torch.utils.data.Dataset):
    def __init__(self, args, data, metadata, train):
        super().__init__()
        self.signal, self.DOA = data
        _, self.s, self.t, self.start_idx = metadata
        self.window_size = args.window_size        
        self.device = args.device
    
        if train:
            # rand_init is used for data augmentation: instead of using always the same time windows, they are randomly shifted
            self.rand_init = (torch.rand(args.data_augmentation_factor)*args.window_size).int()
            self.data_augmentation_factor = args.data_augmentation_factor
        else:
            self.rand_init = torch.zeros(1).int()
            self.data_augmentation_factor = 1
    
    def __getitem__(self, item):
        # The index of the different sample of the batch (same time trace with different random initialization to variate data)
        effective_bs_idx = item % self.data_augmentation_factor
        rand_init = self.rand_init[effective_bs_idx]
        # The index of the different parts of the total time trace
        idx = item // self.data_augmentation_factor
        signal = self.signal[self.s[idx]][self.t[idx]][:, self.start_idx[idx]+rand_init:self.start_idx[idx]+self.window_size+rand_init].type(torch.float).to(self.device)
        DOA = self.DOA[self.s[idx]][self.t[idx]][:, self.start_idx[idx]+rand_init:self.start_idx[idx]+self.window_size+rand_init].type(torch.float).to(self.device)
        return signal, DOA
    
    def __len__(self):
        dataset_length = len(self.start_idx)*self.data_augmentation_factor
        return dataset_length 

def make_loader(args, data, start_idx, train=True, specific_subject=None):    
    metadata = prep_metadata_and_choosing_sessions(args, start_idx, data[1], train=train, specific_subject=specific_subject)
    # Training batch size is definined by argument. 
    # Testing batch size is defined by argument if the argument is not greater than the size of the test dataset. Otherwise, the test batch size is equal to the length of the test dataset.
    batch_size = args.batch_size if train else args.test_batch_size if args.test_batch_size<len(metadata[0]) else len(metadata[0])
    # Create data-loader generator object
    Dataset = dataset_generator_continuous(args, data, metadata, train)
    shuffle = args.shuffle_dataset if train else False
    return data_utils.DataLoader(Dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                drop_last=True,
                                )

def normalize_data(signal):
    return (signal-torch.mean(signal))/torch.std(signal)     

def transformation_to_small_windows(signal, DOA, window_size = 300, sliding_size = 30):    
    start_idx = torch.arange(0, signal.shape[-1]-2*window_size, sliding_size).type(torch.long)
    return start_idx

def to_spike_events(signal, threshold=0.2, off_spike=False):
    signal = spikegen.delta(signal, threshold=threshold, off_spike=off_spike) 
    if off_spike:
        signal_with_off_spikes = torch.zeros((2, signal.shape[0], signal.shape[1]))
        signal_with_off_spikes[0][signal==1.] = 1
        signal_with_off_spikes[1][signal==-1.] = 1
        signal_with_off_spikes = signal_with_off_spikes.transpose(0,1)
        signal_with_off_spikes = signal_with_off_spikes.reshape(signal.shape[0], signal.shape[1]*2)
        signal = signal_with_off_spikes
    return signal.type(torch.bool)

def prep_metadata_and_choosing_sessions(args, start_idx, DOA, train=True, specific_subject=None):
    # write the different combinations of subject and time in a list [subject][time]
    subject, time = make_sub_times_lists([args.subjects], len(DOA[0]))
    metadata = [subject, time, start_idx]
    # flatten the lists into 1d lists
    metadata = flatten_data(args, metadata)    
    metadata = separate_subjects(metadata, specific_subject) if specific_subject is not None else metadata
    return metadata

def make_sub_times_lists(subjects, num_times):
    subjects_list, times_list = [], []
    times = list(range(num_times))
    times_list = [[time for time in times] for sub in subjects]
    subjects_list = [[sub for time in times] for sub in range(len(subjects))]
    return subjects_list, times_list 
    
def flatten_data(args, metadata):
    subjects, times, start_idx = metadata
    flat_sub, flat_times, flat_start_idx = [], [], []
    counter = 0
    for sub, _ in enumerate([args.subjects]):
        for time, _ in enumerate(times[0]):
            for i, _ in enumerate(start_idx[sub][time]):                    
                flat_sub.append(subjects[sub][time])
                flat_times.append(times[sub][time])
                flat_start_idx.append(start_idx[sub][time][i])
                counter += 1
    example_num = list(range(counter))
    return [np.array(example_num), np.array(flat_sub), np.array(flat_times), np.array(flat_start_idx)]

def add_random_init_time_step(args, metadata, train):    
    rand_init = (torch.rand(1)*args.window_size).int().item() if train else 0
    metadata[-1] = metadata[-1] + rand_init
    for i in range(len(metadata)):
        metadata[i] = metadata[i][0:-1]
    return metadata

def separate_subjects(metadata, specific_subject):
    # train subjects one at the time
    subjects = metadata[1]
    indices_list = list(subjects==specific_subject)
    metadata = [metadata[i][indices_list] for i in range(len(metadata))]
    return metadata

def DOF_to_DOA(x):
    linear_transformation_matrix = torch.Tensor([[0.639, 0.000, 0.000, 0.000, 0.000],
                                                     [0.383, 0.000, 0.000, 0.000, 0.000],
                                                     [0.000, 1.000, 0.000, 0.000, 0.000],
                                                     [-0.639,0.000, 0.000, 0.000, 0.000],
                                                     [0.000, 0.000, 0.400, 0.000, 0.000],
                                                     [0.000, 0.000, 0.600, 0.000, 0.000],
                                                     [0.000, 0.000, 0.000, 0.400, 0.000],
                                                     [0.000, 0.000, 0.000, 0.600, 0.000],
                                                     [0.000, 0.000, 0.000, 0.000, 0.000],
                                                     [0.000, 0.000, 0.000, 0.000,0.1667],
                                                     [0.000, 0.000, 0.000, 0.000,0.3333],
                                                     [0.000, 0.000, 0.000, 0.000, 0.000],
                                                     [0.000, 0.000, 0.000, 0.000,0.1667],
                                                     [0.000, 0.000, 0.000, 0.000,0.3333],
                                                     [0.000, 0.000, 0.000, 0.000, 0.000],
                                                     [0.000, 0.000, 0.000, 0.000, 0.000],
                                                     [-0.19, 0.000, 0.000, 0.000, 0.000],
                                                     [0.000, 0.000, 0.000, 0.000, 0.000],
                                                     [0.000, 0.000, 0.000, 0.000, 0.000]
                                                     ])
    """
    Matrix found in
    "Agamemnon Krasoulis, Sethu Vijayakumar, and Kianoush Nazarpour. Effect of user practice
    on prosthetic finger control with an intuitive myoelectric decoder. 13. ISSN 1662-453X. URL
    https://www.frontiersin.org/articles/10.3389/fnins.2019.00891."
    """
    linear_transformation_matrix = linear_transformation_matrix[0:-1].transpose(0,1)
    linear_transformation_matrix = linear_transformation_matrix.unsqueeze(0)
    x = x.unsqueeze(2).float()
    y = linear_transformation_matrix @ x
    return y[:,:,0].half()

def short_term_fourrier_transform(x, DOA, window_parameters):
    sampling_rate = 2e+3 # samples per seconds
    window_size, stride = window_parameters
    f, t, x = spectrogram(x,   # need to replace t and f with _ to save memory
                fs=sampling_rate,
                nperseg= window_size, # size of the windows to do the fft on -> it also defines the number of frequencies that are sampled (along with the sampling rate that defines the maximum freq fnyquist=fsampling/2)
                noverlap= window_size-stride, # how many points overlap between windows -> making this number close to nperseg increase the number of time steps we get at the end. It could be improved but would require a lot of memory
                return_onesided=True, # important to get only positive frequencies
                axis=0,
                )  # returns fft (freq, channels, time)
    # x = np.abs(x)
    x = x[:-1] # It is more convenient to drop the last frequency rather than having an odd number
    f = f[:-1]

    # The upper half frequencies are useless (do not contain any information)
    DOA = DOA[::stride]
    DOA = DOA[:x.shape[2]]

    x = torch.Tensor(x.reshape((x.shape[0]*x.shape[1], x.shape[2]))).transpose(0,1)
    return x, DOA # (channels*freq, time): we consider that frequencies are other sort of channels

