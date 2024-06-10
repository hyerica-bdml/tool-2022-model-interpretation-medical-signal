import copy
import numpy as np

from scipy import signal
from .utils_common import classifier_fn

def make_on_off_sample_matrix(
    n_features, 
    ratio=0.5, 
    num_samples=100, 
    ecg_mode=True
):   
    '''
    1. to generate masking to obtain perturbed samples according to on-off ratio and the number of samples
    args
        n_features: the number of segments. if ecg_mode, it is a list of the number of segments for each channel.
        ratio: perturbation ratio to be classified as original label up to 50 percent.
        num_samples: the number of randomdly generated perturbed samples.
        ecg_mode
    returns
        on_off_matrix: masking for generating perturbed samples
    '''
    assert n_features is not None
    
    all_n_features = np.sum(n_features) if ecg_mode else n_features
    off_num = int(all_n_features * ratio)
    
    on_off_matrix = []
    off_vector = np.zeros(num_samples*off_num, dtype=np.int_)
    on_vector = np.ones(num_samples*(all_n_features - off_num), dtype=np.int_)
    on_off_vector = np.concatenate((on_vector, off_vector), axis=-1)
    np.random.shuffle(on_off_vector)
    
    on_off_matrix = np.reshape(on_off_vector, (num_samples,-1))

    if ecg_mode:
        on_off_matrix[0,:]=1
    else:
        on_off_matrix[0]=1
    return on_off_matrix

def get_a_logit_of_sample(
    net, 
    Zxx, 
    n_features, 
    on_off_matrix=None,
    ratio=0.5, 
    num_samples=100, 
    nperseg=40, 
    data_len=10000, 
    segments=None, 
    DEVICE='cpu', 
    ecg_mode=True
):
    '''
    2. to generate logit given perturbed STFT sample
    args
        net: target model to be explained
        Zxx: STFT of x from scipy.signal.stft
        n_features: the number of segments. if ecg_mode, it is a list of the number of segments for each channel.
        on_off_matrix: masking to generate perturbed sample
        ratio: masking ratio (to be no used in this function. will be removed after code refactoring)
        num_samples: the number of randomly generated perturbed samples.
        nperseg: the length of each segment for scipy.signal.stft. This value is found by utils_eeg.find_an_available_nperseg
        data_len: to unify the length of the signal
        segments: superpixel image founded using Felzenszwalb algorithm. segments, _ from utils_eeg.make_segments
        DEVICE: cpu or cuda
        ecg_mode
    returns
        sample_labels: labels associated with given perturbed samples        
    '''
    assert on_off_matrix is not None or segments is not None
    
    # Used to turn off the masked regions in the sample data
    off_image = np.zeros_like(Zxx, dtype=np.float_)
    # List to store samples
    sample_imgs = []
    sample_labels = []
    fs = 1000 if ecg_mode else 200 # sampling rate for scipy.signal.stft
    
    # for sample_idx in tqdm(range(len(on_off_matrix)))
    for sample_idx in range(len(on_off_matrix)):
        sample_image = copy.deepcopy(Zxx)
        # mask: Parts of the image to be removed
        mask = np.zeros(segments.shape).astype(bool)
        if ecg_mode:
            mask[segments == -1] = True
            # Each sample combines the number of segments for all leads
            # Therefore, turn on and off segments based on the number of segments per lead
            used_n_features = 0
            for j in range(len(n_features)):
                # Find the parts to turn off for each lead
                off_indexes = np.where(on_off_matrix[sample_idx][used_n_features:used_n_features+n_features[j]]==0)[0]
                for off in off_indexes:
                    mask[j][segments[j] == off] = True
                used_n_features += n_features[j]
        else:
            zeros = np.where(on_off_matrix[sample_idx]==0)[0]
            for z in zeros:
                mask[segments==z]=True
                
        sample_image[mask] = off_image[mask]
        # Originally, Zxx has band 0 at the top, but it is flipped for easier viewing
        # Therefore, flip it back to restore correctly using istft
        sample_image = np.array(sample_image)
        sample_imgs.append(sample_image)
    if len(sample_imgs) > 0:
        sample_imgs = np.array(sample_imgs)
        _, xrec = signal.istft(sample_imgs, fs, nperseg = nperseg)
        logits = classifier_fn(net, xrec.reshape(-1,12,data_len), DEVICE) if ecg_mode else classifier_fn(net, xrec.reshape(-1,1,3000))        
        sample_labels.extend(logits) 
    return sample_labels

def find_a_on_off_ratio(
    net, 
    Zxx, 
    segments, 
    num_samples, 
    nperseg, 
    DATA_LEN, 
    n_features, 
    rand_label, 
    DEVICE='cpu', 
    ecg_mode=True
):
    '''
    to find perturbation ratio
    args
        net: target model to be explained
        Zxx: STFT of x from scipy.signal.stft
        segments: superpixel image founded using Felzenszwalb algorithm. segments, _ from utils_eeg.make_segments
        num_samples: (int) the number of randomdly generated perturbed samples.
        nperseg: the length of each segment for scipy.signal.stft. This value is found by utils_eeg.find_an_available_nperseg
        DATA_LEN: to unify the length of the signal
        n_features: the number of segments. if ecg_mode, it is a list of the number of segments for each channel.
        rand_label: target class
        DEVICE: cpu or cuda
        ecg_mode
    returns
        on_off_ratio: (float) found ratio
    '''
    on_off_ratio, pred_label_max_num = 0., num_samples
    for off_ratio in np.arange(0.1, 1, 0.1):
        # print(f'Start!! - off ratio is {off_ratio} =============================')
        on_off_matrix = make_on_off_sample_matrix(n_features, ratio=off_ratio, num_samples=num_samples,ecg_mode=ecg_mode)
        sample_labels = get_a_logit_of_sample(net, Zxx, n_features, on_off_matrix, ratio = off_ratio, num_samples = num_samples, nperseg = nperseg, data_len=DATA_LEN, segments=segments, DEVICE=DEVICE,ecg_mode=ecg_mode)

        # Obtain unique label values and their counts by taking argmax for each sample
        sample_label_count = np.unique(np.argmax(sample_labels, axis=-1), return_counts=True)

        # print('======= Check a pred =======')
        # print(f'real label : {rand_label} & pred for data[0] : {np.argmax(sample_labels[0])}')

        # Find the ratio with the largest number of predicted labels
        # The smaller the number of most predicted classes, the more the model is confused
        # Use this ratio to find the corresponding part
        if ecg_mode and (np.max(sample_label_count[1]) < pred_label_max_num):
            pred_label_max_num = np.max(sample_label_count[1])
            on_off_ratio = off_ratio
            
        # Add a condition to use the ratio when the number of samples classified as the original label is more than 50
        if not ecg_mode and np.max(sample_label_count[1]) < pred_label_max_num and sample_label_count[1][np.where(sample_label_count[0]==rand_label)[0].item()]>(num_samples//2):
            pred_label_max_num = np.max(sample_label_count[1])
            on_off_ratio = off_ratio
        # print(sample_label_count)
        # print('=======================================================\n')
    # print(f'The ratio to be used is {on_off_ratio}')
    return on_off_ratio

def __main__(
    net, 
    Zxx, 
    segments, 
    num_samples, 
    nperseg, 
    DATA_LEN, 
    rand_label, 
    DEVICE, 
    ecg_mode=True
):
    '''
    args
        net: torch model to be explained
        Zxx: STFT of x from scipy.signal.stft
        segments: superpixel image founded using Felzenszwalb algorithm. segments, _ from utils_eeg.make_segments
        num_samples: (int) the number of randomly generated perturbed samples. default=100
        nperseg: (int) the length of each segment for scipy.signal.stft. This value is found by utils_eeg.find_an_available_nperseg
        DATA_LEN: (int) to unify the length of the signal. default=10000
        rand_label: (int) target class
        DEVICE: (str) cuda or cpu
        ecg_mode: (bool) to affect sampling rate or feature dimension. Note: This is not only for ECG signal, but for specific shape of data.
    returns
        on_off_matrix: perturbed samples
        sample_labels: labels associated with given perturbed samples
        n_features: the number of channels
    '''
    # The feature (segment) in the image must be turned on and off
    n_features = [] if ecg_mode else np.unique(segments).shape[0]
    if ecg_mode: # if ecg_mode, Zxx has channels, so n_features will have list of the number of features.
        for i in range(Zxx.shape[0]):
            n_features.append(np.unique(segments[i]).shape[0])
    # print(f'all pixel : {np.sum(n_features)}')
    
    on_off_ratio = find_a_on_off_ratio(net, Zxx, segments, num_samples, nperseg, DATA_LEN, n_features, rand_label, DEVICE, ecg_mode)
    
    # print('Start creating samples(z).')
    on_off_matrix = make_on_off_sample_matrix(n_features, on_off_ratio, num_samples, ecg_mode)
    sample_labels = get_a_logit_of_sample(net, Zxx, n_features, on_off_matrix, ratio = on_off_ratio, num_samples = num_samples, nperseg = nperseg, data_len=DATA_LEN, segments=segments, DEVICE=DEVICE, ecg_mode=ecg_mode)
    return on_off_matrix, sample_labels, n_features