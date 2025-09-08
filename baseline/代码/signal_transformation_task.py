# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 06:48:19 2019

@author: Pritam
"""

import numpy as np
import math
import cv2


# 你原来就有的函数保留：
# add_noise_with_SNR, scaled, permute, time_warp, negate, hor_filp, ...

# ---------- 新增 1：时间屏蔽（Time Mask） ----------
def time_mask(x, mask_frac=0.1, n_masks=1, fill_mode='zero'):
    """
    在时间维随机遮挡一段连续片段。
    - mask_frac: 每个mask长度占比（0~1），默认 10%
    - n_masks:   遮挡段数量
    - fill_mode: 'zero' 用 0；'mean' 用均值
    """
    x = np.asarray(x, dtype=float).copy()
    L = len(x)
    if L == 0:
        return x

    mask_len = max(1, int(L * mask_frac))
    for _ in range(max(1, n_masks)):
        start = np.random.randint(0, max(1, L - mask_len + 1))
        end = start + mask_len
        if fill_mode == 'mean':
            x[start:end] = float(np.mean(x))
        else:
            x[start:end] = 0.0
    return x

# ---------- 新增 2：DGW-like 时间形变（平滑的非线性时间扭曲） ----------
def dgw_like_warp(x, max_scale=0.3, n_pieces=5):
    """
    近似“DGW”风格：分成若干片段，每段用随机伸缩比例做时间重采样，然后整体插值回原长度。
    - max_scale: 每段伸缩系数在 [1-max_scale, 1+max_scale] 之间
    - n_pieces:  分段数量（越大越复杂）
    """
    x = np.asarray(x, dtype=float)
    L = len(x)
    if L <= 2:
        return x.copy()

    # 随机段边界
    cuts = sorted(np.random.choice(np.arange(1, L-1), size=max(1, n_pieces-1), replace=False))
    bounds = [0] + cuts + [L]

    # 每段随机伸缩
    scales = np.random.uniform(1 - max_scale, 1 + max_scale, size=len(bounds)-1)

    # 构造“变形后的时间轴”累计长度
    warped_cum = [0.0]
    for (a, b), s in zip(zip(bounds[:-1], bounds[1:]), scales):
        seg_len = b - a
        warped_cum.append(warped_cum[-1] + seg_len * s)
    warped_cum = np.asarray(warped_cum)

    # 归一化到 [0,1]
    warped_t = warped_cum / warped_cum[-1]
    orig_t = np.linspace(0.0, 1.0, L)

    # 段边界的原始时间坐标
    orig_knots = np.array(bounds) / float(L-1)

    # 反求：给定等距 orig_t，找到对应的“原索引”位置，再用 x 插值
    map_t = np.interp(orig_t, warped_t, orig_knots)  # 对应到原始的时间坐标
    map_idx = map_t * (L - 1)
    y = np.interp(map_idx, np.arange(L), x)
    return y

# ---------- 新增 3：裁剪 + 填充（Crop & Pad） ----------
def crop_and_pad(x, crop_frac_range=(0.6, 0.95), pad_mode='edge'):
    """
    随机裁剪中间一段，再在两侧用边缘值填充回原长度。
    - crop_frac_range: 裁剪后长度占比范围（比如 0.6~0.95）
    - pad_mode: np.pad 的 mode，'edge' 用边缘值填充；也可用 'constant'
    """
    x = np.asarray(x, dtype=float)
    L = len(x)
    if L == 0:
        return x.copy()

    low, high = crop_frac_range
    low = max(1e-3, min(low, 1.0))
    high = max(low, min(high, 1.0))

    new_L = max(1, int(L * np.random.uniform(low, high)))
    start = np.random.randint(0, max(1, L - new_L + 1))
    segment = x[start:start + new_L]

    # 随机左右 padding 总量 = L - new_L
    pad_total = L - new_L
    pad_left = np.random.randint(0, pad_total + 1)
    pad_right = pad_total - pad_left

    y = np.pad(segment, (pad_left, pad_right), mode=pad_mode)
    return y

def add_noise(signal, noise_amount):
    """ 
    adding noise
    """
    noise = np.random.normal(1, noise_amount, np.shape(signal)[0])
    noised_signal = signal+noise
    return noised_signal
    
def add_noise_with_SNR(signal, noise_amount):###根据指定的信噪比向信号中添加高斯噪声
    """ 
    adding noise
    created using: https://stackoverflow.com/a/53688043/10700812 
    """
    
    target_snr_db = noise_amount #20
    # 计算信号的功率并将其转换为分贝dB
    x_watts = signal ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # 计算噪声，然后将其转换为瓦特（Watts）
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))     # Generate an sample of white noise
    noised_signal = signal + noise_volts        # noise added signal

    return noised_signal

def scaled(signal, factor):
    """"
    scale the signal
    """
    scaled_signal = signal * factor
    return scaled_signal


def negate(signal):
    """ 
    negate the signal取反
    """
    negated_signal = signal * (-1)
    return negated_signal

    
def hor_filp(signal):
    """ 
    flipped horizontally 水平翻转
    """
    hor_flipped = np.flip(signal)
    return hor_flipped


def permute(signal, pieces):
    """ 
    signal: numpy array (batch x window)
    pieces: number of segments along time    
    """
    pieces       = int(np.ceil(np.shape(signal)[0]/(np.shape(signal)[0]//pieces)).tolist())
    piece_length = int(np.shape(signal)[0]//pieces)
    #total_length = piece_length * pieces
    #padding_length = total_length - len(signal)
    #padded_signal = np.pad(signal, (0, padding_length), mode='constant', constant_values=0)
    ###permuted_signal = np.reshape(padded_signal, (pieces, piece_length)).tolist()
    
    #sequence = list(range(0,pieces))
    #np.random.shuffle(sequence)
    
    #permuted_signal = (np.reshape(signal[:(np.shape(signal)[0]//pieces*pieces)], (pieces, piece_length)).tolist() + padded_signal
                       ##+[signal[(np.shape(signal)[0]//pieces*pieces):]])
    #permuted_signal = np.asarray(permuted_signal)[sequence]
    #permuted_signal = np.hstack(permuted_signal)
        
    #return permuted_signal
    signal_length = len(signal)
    padding_length = (pieces * piece_length) - signal_length###要补0的长度
    if padding_length < 0:  #处理pieces>signal_length的时候
        padding_length = 0

    # 在末尾用0补全信号
    if padding_length > 0:
        signal = np.pad(signal, (0, padding_length), mode='constant')

    ###将signal的shape调整为(pieces, piece_length)
    permuted_signal = np.reshape(signal, (pieces, piece_length))

    # Shuffle the segments
    sequence = np.arange(pieces)
    np.random.shuffle(sequence)
    permuted_signal = permuted_signal[sequence]

    # Flatten back to 1D
    permuted_signal = np.hstack(permuted_signal)###将所有打乱的信号水平拼接到一起

    # Trim back to original length (if padding was added)
    # Trim back to original length (if padding was added)
    if padding_length > 0:  # 如果之前补齐过
        permuted_signal = permuted_signal[:signal_length]  # 使用切片操作将permuted_signal裁剪到原始长度signal_length。
    return permuted_signal
##if __name__ == "__main__":
    ##test_permute()  # 直接运行脚本时执行测试


def time_warp(signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
    """ 
    signal: numpy array (batch x window)
    sampling freq
    pieces: number of segments along time
    stretch factor
    squeeze factor
    """
    
    ##total_time = np.shape(signal)[0]//sampling_freq
    total_time = np.shape(signal)[0] // sampling_freq
    segment_time = total_time/pieces
    sequence = list(range(0,pieces))
    stretch = np.random.choice(sequence, math.ceil(len(sequence)/2), replace = False)
    squeeze = list(set(sequence).difference(set(stretch)))
    initialize = True
    for i in sequence:
        orig_signal = signal[int(i*np.floor(segment_time*sampling_freq)):int((i+1)*np.floor(segment_time*sampling_freq))]
        ##orig_signal = orig_signal.reshape(np.shape(orig_signal)[0],1)
        orig_signal = orig_signal.astype(np.float32).reshape(-1, 1)  # 转换为 float32
        if i in stretch:
            output_shape = int(np.ceil(np.shape(orig_signal)[0]*stretch_factor))
            new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
            if initialize == True:
                time_warped = new_signal
                initialize = False
            else:
                time_warped = np.vstack((time_warped, new_signal))
        if i in squeeze:
            output_shape = int(np.ceil(np.shape(orig_signal)[0]*squeeze_factor))
            new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
            if initialize == True:
                time_warped = new_signal
                initialize = False
            else:
                time_warped = np.vstack((time_warped, new_signal))
    return time_warped
if __name__ == "__main__":
    signal = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # 示例信号
    sampling_freq = 1  # 采样频率
    pieces = 3  # 分段数
    stretch_factor = 1.5  # 拉伸因子
    squeeze_factor = 0.5  # 压缩因子
    time_warped_signal = time_warp(signal, sampling_freq, pieces, stretch_factor, squeeze_factor)  # 修复：保存返回值
    # 打印结果
    print("原始信号:", signal.flatten())
    print("时间扭曲后的信号:", time_warped_signal.flatten())


