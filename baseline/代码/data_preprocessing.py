import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
import utils
from pathlib import Path
import pickle
from scipy.stats import mode
import scipy.signal as signal

def import_filenames(directory_path):
    """
    import all file names of a directory """
    filename_list = []
    dir_list = []
    for root, dirs, files in os.walk(directory_path, topdown=False):
        filename_list = files
        dir_list = dirs
    return filename_list, dir_list


def normalize(x, x_mean, x_std):
    """
    perform z-score normalization of a signal """
    x_scaled = (x - x_mean) / x_std
    return x_scaled


def make_window(signal, freq, overlap_pct, window_size_sec):
    """
    perform cropped signals of window_size seconds for the whole signal
    overlap input is in percentage of window_size
    window_size is in seconds """

    window_size = freq * window_size_sec
    overlap = int(window_size * (overlap_pct / 100))
    start = 0
    segmented = np.zeros((1, window_size), dtype=float)
    while (start + window_size <= len(signal)):
        segment = signal[start:start + window_size]
        segment = segment.reshape(1, len(segment))
        segmented = np.append(segmented, segment, axis=0)
        start = start + window_size - overlap
    return segmented[1:]

"""ef make_window_labels(labels, freq, overlap_pct, window_size_sec):
    Generate a single label (mode) for each window.
    window_size = int(window_size_sec * freq)
    step_size = int(window_size * (1 - overlap_pct))
    windows = []
    for i in range(0, len(labels) - window_size + 1, step_size):
        window_labels = labels[i:i + window_size]
        ###windows.append(mode(window_labels)[0][0])  # 取窗口内众数标签
        if len(window_labels) == 0:  # 处理空窗口
            windows.append(0)  # 默认标签（可根据需要调整）
            continue
        mode_result = mode(window_labels, axis=None, keepdims=True)  # 使用 keepdims=True 确保输出为数组
        mode_value = mode_result[0][0]  # 获取众数
        windows.append(mode_value)
    return np.array(windows, dtype=float)"""


def extract_wesad_dataset(overlap_pct, window_size_sec, data_save_path, save=True):
    print('WESAD')
    wesad_path = r"D:/WESAD/"
    ###wesad_labels_path = r"D:/WESAD"
    freq = 256
    utils.makedirs(data_save_path)
    window_size = window_size_sec * freq
    # 定义受试者 ID
    subject_id = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    # 生成文件路径列表
    wesad_file_path = [(os.path.join(wesad_path, f"S{i}", f"S{i}.pkl")) for i in subject_id]
    # 检查文件是否存在
    wesad_file_names = [f for f in wesad_file_path if os.path.exists(f)]
    if wesad_file_names:
        print("找到以下文件:")
        for file_path in wesad_file_names:
            print(file_path)
    else:
        raise FileNotFoundError("未找到任何有效的 .pkl 文件！")
###初始化两个字典
    wesad_dict = {}
    wesad_labels = {}

    for file_path in tqdm(wesad_file_names):
        subject_id = int(os.path.basename(os.path.dirname(file_path))[1:])###从文件路径中提取被试的数字ID
        ###如果data是个字典，就打印它的键，如果不是就打印"Not a dictionary"
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
                print(type(data), flush=True)
                print(data.keys() if isinstance(data, dict) else "Not a dictionary", flush=True)
        except Exception as e:
            print(f"处理 {file_path} 时出错：{str(e)}")
            continue

        # 这部分必须在 try 里面或之后，但要保持结构清晰
        ecg_raw = data['signal']['chest']['ECG'].flatten()
        labels = data['label']  # 通常是个 1D array
        print(type(labels), np.shape(labels))  # 先确认
        labels = np.array(labels).flatten()
        min_length = min(len(ecg_raw), len(labels))
        ecg_raw = ecg_raw[:min_length]
        labels = labels[:min_length]
        print(f"Subject {subject_id}: values length = {len(ecg_raw)}, labels length = {len(labels)}")

        # 高通滤波器设计（用于抗混叠）
        hp_filter = signal.cheby2(N=4, rs=60,
                                  Wn=0.8 / (700 / 2),  # 0.8Hz / Nyquist
                                  btype='highpass', analog=False, output='sos')
        ecg_filtered = signal.sosfiltfilt(hp_filter, ecg_raw)

        # 使用 resample_poly 下采样（700Hz -> 256Hz）
        ecg_ds = signal.resample_poly(ecg_filtered, up=256, down=700)

        # 同步下采样标签（直接下采样到对应数量）
        ratio = len(ecg_ds) / len(ecg_raw)
        indices = np.round(np.arange(len(ecg_ds)) / ratio).astype(int)
        indices = np.clip(indices, 0, len(labels) - 1)
        labels_ds = labels[indices]
        labels=labels_ds

        print(f"下采样后 ECG 长度: {len(ecg_ds)}, Labels 长度: {len(labels_ds)}")
        # === 每个用户的 z-score 标准化 ===
        sort_data = np.sort(ecg_ds)
        x_std = np.std(sort_data[int(0.025 * sort_data.shape[0]): int(0.975 * sort_data.shape[0])])
        x_mean = np.mean(sort_data)
        data = normalize(ecg_ds, x_mean, x_std)

        # === 加窗处理 ===
        window_size = window_size_sec* freq  # 10秒窗口，对应2560个采样点
        data_windowed = make_window(data, freq, overlap_pct, window_size_sec)
        labels_windowed = make_window(labels, freq, overlap_pct, window_size_sec)
        labels_mode=mode(labels_windowed, axis=1, keepdims=True)[0]  # 使用 keepdims=True 确保输出为数组
        labels_windowed = labels_mode.reshape(-1, 1)
        print(f"Subject {subject_id}: data_windowed shape = {data_windowed.shape}, labels_windowed shape = {labels_windowed.shape}")
        if len(data_windowed) != len(labels_windowed):
            raise ValueError(f"Shape mismatch for subject {subject_id}: data_windowed has {len(data_windowed)} rows, labels_windowed has {len(labels_windowed)} rows")

        # 加窗后保存进字典（放在 for 循环内部）
        wesad_dict[subject_id] = data_windowed
        wesad_labels[subject_id] = labels_windowed

    print('dict unpacking...')
    final_set = np.zeros((1, window_size + 2), dtype=float)
    for subject_id in tqdm(wesad_dict.keys()):
        values = wesad_dict[subject_id]
        labels = wesad_labels[subject_id]
        print(f"Subject {subject_id}: values shape = {values.shape}, labels shape = {labels.shape}")
        if len(values) != len(labels):
            raise ValueError(f"Shape mismatch for subject {subject_id}: values has {len(values)} rows, labels has {len(labels)} rows")
        key = np.repeat(subject_id, len(values)).astype(float).reshape(-1, 1)
        ###labels_max = np.amax(labels, axis=1)
        labels_final= labels.reshape(-1, 1)
        signal_set = np.hstack((key, labels_final, values))
        final_set = np.vstack((final_set, signal_set))
    ## first column stands for labels: XX.CC == XX person id, and CC clips
    final_set = final_set[1:]
    if save:
        np.save(os.path.join(data_save_path, 'wesad_dict.npy'), final_set)
    print('wesad files importing finished')
    return final_set

###测试代码
if __name__ == "__main__":
    overlap_pct = 0.5
    window_size_sec = 10
    data_save_path = r"./wesad_processed"

    final_data = extract_wesad_dataset(overlap_pct, window_size_sec, data_save_path, save=True)

    print("Final data shape:", final_data.shape)
    print("示例前5行：")
    print(final_data[:5])


def load_data(path):
    dataset = np.load(path, allow_pickle=True)
    return dataset


def wesad_prepare_for_10fold(wesad_data, numb_class=4):
    person = wesad_data[:, 0]
    y_stress = wesad_data[:, 1]
    ecg = wesad_data[:, 2:]
    mask = (y_stress != 0) & (y_stress != 5) & (y_stress != 6) & (y_stress != 7)
    ecg = ecg[mask]  ###只保留mask为True的ecg数据

    ###ecg = ecg[mask]
    person = person[mask].reshape(-1, 1)
    y_stress = y_stress[mask] - 1
    y_stress = y_stress.reshape(-1, 1)

    wesad_data = np.hstack((person, y_stress, ecg))

    return wesad_data  # 4 class
if __name__ == '__main__':
    overlap_pct = 0
    window_size_sec = 10
    data_save_path = r"D:\WESAD"
    save = True
    # 生成 wesad_data
    wesad_data = extract_wesad_dataset(overlap_pct, window_size_sec, data_save_path, save)
    # 预处理为 10 折交叉验证
    wesad_data = wesad_prepare_for_10fold(wesad_data, numb_class=4)


def save_list(mylist, filename):
    for i in range(len(mylist)):
        temp = mylist[i]
        with open(filename, 'a', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(temp)
    return
