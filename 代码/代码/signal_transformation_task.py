# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 06:48:19 2019

@author: Pritam
"""

import numpy as np
import math
import cv2


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


