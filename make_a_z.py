import copy
# from tqdm.auto import tqdm
import numpy as np

from scipy import signal
from .utils_common import classifier_fn

''' 1. num_samples만큼 on-off ratio 만들기 '''
def make_on_off_sample_matrix(n_features, ratio = 0.5, num_samples = 100, ecg_mode=True):   
    assert n_features is not None
    
    all_n_features = np.sum(n_features) if ecg_mode else n_features
    off_num = int(all_n_features * ratio)
    
    on_off_matrix = []
    off_vector = np.zeros(num_samples*off_num, dtype=np.int_)
    on_vector = np.ones(num_samples*(all_n_features - off_num), dtype=np.int_)
    on_off_vector = np.concatenate((on_vector, off_vector), axis=-1)
    np.random.shuffle(on_off_vector)
    
    on_off_matrix = np.reshape(on_off_vector, (num_samples,-1))
#     print(on_off_matrix)
    if ecg_mode:
        on_off_matrix[0,:]=1
    else:
        on_off_matrix[0]=1
    return on_off_matrix

''' 2. on_off_matrix 토대로 sample image 만들기 '''
def get_a_logit_of_sample(net, Zxx, n_features, on_off_matrix=None,ratio=0.5, num_samples=100, nperseg=40, data_len = 10000, segments=None, DEVICE='cuda', ecg_mode=True):
    assert on_off_matrix is not None or segments is not None
    
    # 하나의 sample 데이터에 곱하여 mask 영역에 따라 이미지 off하게 하는 역할
    off_image = np.zeros_like(Zxx, dtype=np.float_)
    # 샘플 저장할 리스트
    sample_imgs = []
    sample_labels = []
    fs = 1000 if ecg_mode else 200
    
    # for sample_idx in tqdm(range(len(on_off_matrix)))
    for sample_idx in range(len(on_off_matrix)):
        sample_image = copy.deepcopy(Zxx)
        # mask : 이미지에서 지울 부분
        mask = np.zeros(segments.shape).astype(bool)
        if ecg_mode:
            mask[segments == -1] = True
            # sample 하나당 모든 lead의 segment 수가 합쳐져 있음
            # 따라서 lead별 segment 수만큼 넘겨서 끄고 킬 것임
            used_n_features = 0
            for j in range(len(n_features)):
                # lead 별 0인 부분을 찾아서 off
                off_indexes = np.where(on_off_matrix[sample_idx][used_n_features:used_n_features+n_features[j]]==0)[0]
                for off in off_indexes:
                    mask[j][segments[j] == off] = True
                used_n_features += n_features[j]
        else:
            zeros = np.where(on_off_matrix[sample_idx]==0)[0]
            for z in zeros:
                mask[segments==z]=True
                
        sample_image[mask] = off_image[mask]
        # 원래 Zxx는 band 0가 위에 위치하는데, 보기 편하게 하려고 filp 한 상태임
        # 따라서 제대로 복원(istft)하기위해 원래대로 돌림
        sample_image = np.array(sample_image)
        sample_imgs.append(sample_image)
    if len(sample_imgs) > 0:
        sample_imgs = np.array(sample_imgs)
        _, xrec = signal.istft(sample_imgs, fs, nperseg = nperseg)
        logits = classifier_fn(net, xrec.reshape(-1,12,data_len), DEVICE) if ecg_mode else classifier_fn(net, xrec.reshape(-1,1,3000))        
        sample_labels.extend(logits) 
    return sample_labels



def find_a_on_off_ratio(net, Zxx, segments, num_samples, nperseg, DATA_LEN, n_features, rand_label, DEVICE='cuda', ecg_mode=True):
    on_off_ratio, pred_label_max_num = 0., num_samples
    for off_ratio in np.arange(0.1, 1, 0.1):
        # print(f'Start!! - off ratio is {off_ratio} =============================')
        on_off_matrix = make_on_off_sample_matrix(n_features, ratio=off_ratio, num_samples=num_samples,ecg_mode=ecg_mode)
        sample_labels = get_a_logit_of_sample(net, Zxx, n_features, on_off_matrix, ratio = off_ratio, num_samples = num_samples, nperseg = nperseg, data_len=DATA_LEN, segments=segments, DEVICE=DEVICE,ecg_mode=ecg_mode)

        # sample 마다 argmax를 취하여 유니크한 label 값 & 개수를 얻음
        sample_label_count = np.unique(np.argmax(sample_labels, axis=-1), return_counts=True)

        # print('======= Check a pred =======')
        # print(f'real label : {rand_label} & pred for data[0] : {np.argmax(sample_labels[0])}')

        # ratio별로 예측 label 개수 중 가장 큰 값
        # 즉, 가장 많이 예측된 class의 개수가 가장 작을수록 model이 헷갈린다는 것임
        # 이 ratio를 사용하도록 해당 부분을 찾음
        if ecg_mode and (np.max(sample_label_count[1]) < pred_label_max_num):
            pred_label_max_num = np.max(sample_label_count[1])
            on_off_ratio = off_ratio
            
        # 원래 label로 분류되는 sample이 50 이상일 때 조건도 달아준 것
        if len(np.where(sample_label_count[0]==rand_label)[0])!=0:
            # print((np.where(sample_label_count[0]==rand_label)), len(np.where(sample_label_count[0]==rand_label)))
            if not ecg_mode and np.max(sample_label_count[1]) < pred_label_max_num and sample_label_count[1][np.where(sample_label_count[0]==rand_label)[0].item()]>(num_samples//2):
                pred_label_max_num = np.max(sample_label_count[1])
                on_off_ratio = off_ratio
        # print(sample_label_count)
        # print('=======================================================\n')
    # print(f'The ratio to be used is {on_off_ratio}')
    return on_off_ratio

def __main__(net, Zxx, segments, num_samples, nperseg, DATA_LEN, rand_label, DEVICE, ecg_mode=True):
    # image에서 feature(segment)를 껐다 켰다 해야 함
    n_features = [] if ecg_mode else np.unique(segments).shape[0]
    if ecg_mode:
        for i in range(Zxx.shape[0]):
            n_features.append(np.unique(segments[i]).shape[0])
    # print(f'all pixel : {np.sum(n_features)}')
    
    on_off_ratio = find_a_on_off_ratio(net, Zxx, segments, num_samples, nperseg, DATA_LEN, n_features, rand_label, DEVICE, ecg_mode)
    
    # print('Start creating samples(z).')
    on_off_matrix = make_on_off_sample_matrix(n_features, on_off_ratio, num_samples, ecg_mode)
    sample_labels = get_a_logit_of_sample(net, Zxx, n_features, on_off_matrix, ratio = on_off_ratio, num_samples = num_samples, nperseg = nperseg, data_len=DATA_LEN, segments=segments, DEVICE=DEVICE, ecg_mode=ecg_mode)
    return on_off_matrix, sample_labels, n_features



