# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:49:02 2020

@author: pritam
"""

import os
import tensorflow as tf
import numpy as np
import csv
from sklearn import metrics
import signal_transformation_task as stt
from mlxtend.evaluate import confusion_matrix
import time

window_size = 2560
transform_task = [0, 1, 2, 3, 4, 5, 6]

def time_mask(x, width_pct=(0.05,0.20)):
    L = x.shape[0]
    w = int(L * np.random.uniform(width_pct[0], width_pct[1]))
    if w <= 0: return x.copy()
    start = np.random.randint(0, max(1, L-w))
    y = x.copy()
    y[start:start+w] = 0.0
    return y

def crop_and_pad(x, keep_pct=(0.6,0.95)):
    L = x.shape[0]
    keep = int(L * np.random.uniform(keep_pct[0], keep_pct[1]))
    start = np.random.randint(0, L-keep+1)
    mid = x[start:start+keep]
    pad_left = np.random.randint(0, L-keep+1)
    pad_right = L-keep-pad_left
    return np.pad(mid, (pad_left, pad_right), mode='edge')

def dgw_like_warp(x, n_knots=4, max_warp=0.3):
    L = x.shape[0]
    knots = np.linspace(0, L-1, n_knots+2).astype(int)
    disp = np.zeros_like(knots, dtype=float)
    if len(knots) > 2:
        disp[1:-1] = np.random.uniform(-max_warp*L, max_warp*L, size=len(knots)-2)
    tgt = knots + disp
    tgt[0], tgt[-1] = 0, L-1
    for i in range(1, len(tgt)):
        tgt[i] = max(tgt[i], tgt[i-1]+1e-6)
    full_t = np.arange(L)
    map_t = np.interp(full_t, knots, tgt)
    map_t = np.clip(map_t, 0, L-1)
    idx0 = np.floor(map_t).astype(int)
    idx1 = np.minimum(idx0+1, L-1)
    alpha = map_t - idx0
    return (1-alpha)*x[idx0] + alpha*x[idx1]


def get_label(y, actual_batch_size):
    """ get the label or y true """

    y_label = []
    for i in range(len(transform_task)):
        label = tf.reshape(y[:, i], [actual_batch_size, 1])
        y_label.append(label)
    return y_label


def calculate_loss(y_label, logits):
    """ calculate loss of each transformtion task """
    batch_loss = []
    for i in range(len(transform_task)):
        loss = tf.reduce_mean(
            tf.compat.v1.losses.sigmoid_cross_entropy(multi_class_labels=y_label[i], logits=logits[i]))
        batch_loss.append(loss)
    return batch_loss


def get_prediction(logits):
    """ get the prediction of the model"""
    y_pred = []
    for i in range(len(transform_task)):
        pred = tf.greater(tf.nn.sigmoid(logits[i]), 0.5)
        y_pred.append(pred)
    return y_pred


def make_batch(signal_batch, noise_amount, scaling_factor, permutation_pieces, time_warping_pieces,
               time_warping_stretch_factor, time_warping_squeeze_factor):
    """
    generator: 生成包含 7 种“任务”的一个 batch（原始 + 6 个增强）
    顺序（务必保持，与 transform_task 对齐）:
    0 original
    1 noise
    2 scale
    3 time_mask (替换 negated)
    4 dgw_warp  (替换 flipped)
    5 crop_pad  (替换 permute)
    6 time_warp (沿用原实现)
    """
    import numpy as np
    import tensorflow as tf
    import signal_transformation_task as stt  # 确保顶部已 import

    for i in range(len(signal_batch)):
        signal = signal_batch[i]
        signal = np.trim_zeros(signal, 'b')
        L = len(signal)
        sampling_freq = L // 10 if L >= 10 else 1

        # 1) 现有增强（保持不变）
        noised_signal = stt.add_noise_with_SNR(signal, noise_amount=noise_amount)
        scaled_signal = stt.scaled(signal, factor=scaling_factor)

        # 2) 新增替换的三种
        time_masked_signal = stt.time_mask(signal)            # 替换 negated
        dgw_warped_signal  = stt.dgw_like_warp(signal)        # 替换 flipped
        crop_pad_signal    = stt.crop_and_pad(signal)         # 替换 permute

        # 3) 原 time_warp（长度可能变化，按旧逻辑裁切回 L）
        time_warped_signal = stt.time_warp(signal, sampling_freq,
                                           pieces=time_warping_pieces,
                                           stretch_factor=time_warping_stretch_factor,
                                           squeeze_factor=time_warping_squeeze_factor)
        if len(time_warped_signal) > L:
            tw_start = int(np.random.randint(0, len(time_warped_signal) - L))
            time_warped_signal = time_warped_signal[tw_start:tw_start + L]
        elif len(time_warped_signal) < L:
            # 保险：若实现返回变短，pad 到 L
            pad_left = 0
            pad_right = L - len(time_warped_signal)
            time_warped_signal = np.pad(time_warped_signal, (pad_left, pad_right), mode='edge')

        # 4) 统一成 (len, 1)
        signal            = signal.reshape(L, 1)
        noised_signal     = noised_signal.reshape(L, 1)
        scaled_signal     = scaled_signal.reshape(L, 1)
        time_masked_signal= time_masked_signal.reshape(L, 1)
        dgw_warped_signal = dgw_warped_signal.reshape(L, 1)
        crop_pad_signal   = crop_pad_signal.reshape(L, 1)
        time_warped_signal= time_warped_signal.reshape(L, 1)

        # 5) 注意顺序与标签对齐
        batch = [
            signal,              # 0 original_signal
            noised_signal,       # 1 noised_signal
            scaled_signal,       # 2 scaled_signal
            time_masked_signal,  # 3 time_masked_signal  (新)
            dgw_warped_signal,   # 4 dgw_warped_signal   (新)
            crop_pad_signal,     # 5 croppad_signal      (新)
            time_warped_signal   # 6 time_warped_signal
        ]

        # 与 transform_task = [0,1,2,3,4,5,6] 对齐
        labels = tf.keras.utils.to_categorical([0,1,2,3,4,5,6])

        # 与原逻辑一致：padding 到同长（基本已同长，这里是保险）
        batch = tf.keras.preprocessing.sequence.pad_sequences(batch, dtype='float32', padding='post')

        yield batch, labels



def make_total_batch(data, length, batchsize, noise_amount, scaling_factor, permutation_pieces, time_warping_pieces,
                     time_warping_stretch_factor, time_warping_squeeze_factor):
    """ calling make_batch from here, when batch size is more than one, like 64 or 32, it will make actual batch size = batch_size * len(transformed signal)
    """
    steps = length // batchsize + 1
    for counter in range(steps):

        signal_batch = data[np.mod(np.arange(counter * batchsize, (counter + 1) * batchsize), length)]

        gen_op = make_batch(signal_batch, noise_amount, scaling_factor, permutation_pieces, time_warping_pieces,
                            time_warping_stretch_factor, time_warping_squeeze_factor)
        total_batch = np.array([])
        total_labels = np.array([])
        for batch, labels in gen_op:
            total_batch = np.vstack((total_batch, batch)) if total_batch.size else batch
            total_labels = np.vstack((total_labels, labels)) if total_labels.size else labels

        yield total_batch, total_labels, counter, steps


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_weighted_loss(loss_coeff, batch_loss):
    """ calculate the weighted loss
    """
    output_loss = 0
    for i in range(len(loss_coeff)):
        temp = loss_coeff[i] * batch_loss[i]
        output_loss = output_loss + temp

    return output_loss


def fetch_all_loss(batch_loss, task_loss):
    """
    fetch individual signal transformation losses"""

    for i in range(len(transform_task)):
        task_loss[i] = np.add(task_loss[i], batch_loss[i])
    return task_loss


def fetch_pred_labels(y_preds, pred_task):
    y_preds = np.squeeze(np.asarray(y_preds, dtype=np.int32)).T
    if np.all(pred_task == -1):
        pred_task = y_preds
    else:
        pred_task = np.vstack((pred_task, y_preds))

    return pred_task


def fetch_true_labels(labels, true_task):
    if np.all(true_task == -1):
        true_task = labels
    else:
        true_task = np.vstack((true_task, labels))

    return true_task


# utils.py
def weighted_bce_loss(pos_weight):
    # 强制 pos_weight 为 float32
    pos_weight = tf.cast(pos_weight, tf.float32)

    def loss_fn(y_true, y_pred):
        # 强制 labels / logits 为 float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true, logits=y_pred, pos_weight=pos_weight
        )

    return loss_fn



def get_results_ssl(y_true, y_pred):
    accuracy = np.full((1, 7), np.nan)  #### 初始化 1×7 的准确率数组，初值为 nan
    precision = np.full((1, 7), np.nan)
    recall = np.full((1, 7), np.nan)
    f1_score = np.full((1, 7), np.nan)

    if y_true.shape == y_pred.shape:
        for i in range(len(transform_task)):
            accuracy[:, i] = np.round(metrics.accuracy_score(y_true[:, i], y_pred[:, i]), 2)
            precision[:, i] = np.round(metrics.precision_score(y_true[:, i], y_pred[:, i], labels=[0, 1]), 2)
            recall[:, i] = np.round(metrics.recall_score(y_true[:, i], y_pred[:, i], labels=[0, 1]), 2)
            f1_score[:, i] = np.round(metrics.f1_score(y_true[:, i], y_pred[:, i], labels=[0, 1]), 2)
    else:
        print("error in self supervised result calculation")

    return accuracy, precision, recall, f1_score


def write_result(accuracy, precision, recall, f1_score, epoch_number, result_dict):
    result = [accuracy, precision, recall, f1_score]
    result_dict.update({epoch_number: result})
    return result_dict


"""def write_summary(loss, total_loss, f1_score, epoch_counter, isTraining, summary_writer):

    task_name = ['original', 'noised', 'scaled', 'negated', 'flipped', 'permuted', 'timewarp']
    for i in range(len(task_name)):
        t = task_name[i]
        if isTraining:
            summary = tf.Summary(value=[tf.Summary.Value(tag="train loss/"+ t,          simple_value        =   loss[i][0])])
            summary_writer.add_summary(summary, epoch_counter)
            summary_writer.flush()

            summary = tf.Summary(value=[tf.Summary.Value(tag="train F1 score/"+ t,      simple_value        =   f1_score[0][i])])
            summary_writer.add_summary(summary, epoch_counter)
            summary_writer.flush()

        else:
            summary = tf.Summary(value=[tf.Summary.Value(tag="test loss/"+ t,           simple_value        =   loss[i][0])])
            summary_writer.add_summary(summary, epoch_counter)
            summary_writer.flush()

            summary = tf.Summary(value=[tf.Summary.Value(tag="test F1 score/"+ t,       simple_value        =   f1_score[0][i])])
            summary_writer.add_summary(summary, epoch_counter)
            summary_writer.flush()

    if isTraining:
        summary = tf.Summary(value=[tf.Summary.Value(tag="train loss/total_loss",          simple_value        =   total_loss)])
        summary_writer.add_summary(summary, epoch_counter)
        summary_writer.flush()        
    else:
        summary = tf.Summary(value=[tf.Summary.Value(tag="test loss/total_loss",           simple_value        =   total_loss)])
        summary_writer.add_summary(summary, epoch_counter)
        summary_writer.flush()

    return"""


def write_summary(loss, total_loss, f1_score, epoch_counter, isTraining, summary_writer):
    task_name = ['original', 'noised', 'scaled', 'negated', 'flipped', 'permuted', 'timewarp']

    with summary_writer.as_default():  # 使用上下文管理器，避免重复 flush()
        for i in range(len(task_name)):
            t = task_name[i]
            if isTraining:
                # 记录训练 loss 和 F1
                tf.summary.scalar(f"train_loss/{t}", loss[i][0], step=epoch_counter)
                tf.summary.scalar(f"train_F1_score/{t}", f1_score[0][i], step=epoch_counter)
            else:
                # 记录测试 loss 和 F1
                tf.summary.scalar(f"test_loss/{t}", loss[i][0], step=epoch_counter)
                tf.summary.scalar(f"test_F1_score/{t}", f1_score[0][i], step=epoch_counter)

        # 记录 total_loss
        if isTraining:
            tf.summary.scalar("train_loss/total_loss", total_loss, step=epoch_counter)
        else:
            tf.summary.scalar("test_loss/total_loss", total_loss, step=epoch_counter)

    summary_writer.flush()  # 确保所有写入操作完成
    return


def write_result_csv(kfold, epoch_number, result_store, f1_score):
    f1_score = f1_score[0]
    with open(result_store, 'a', newline='') as csvfile:
        fieldnames = ['fold', 'epoch', 'org', 'noised', 'scaled', 'neg', 'flip', 'perm', 'time_warp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {'fold': kfold, 'epoch': epoch_number, 'org': f1_score[0], 'noised': f1_score[1], 'scaled': f1_score[2],
             'neg': f1_score[3], 'flip': f1_score[4], 'perm': f1_score[5], 'time_warp': f1_score[6]})

    return


def model_result_store(y, y_pred, result_store, kfold):
    accuracy = np.round(metrics.accuracy_score(y, y_pred), 4)
    conf_mat = confusion_matrix(y_target=y, y_predicted=y_pred, binary=False)
    precision = np.round(np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis=0)), 4)
    recall = np.round(np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis=1)), 4)
    f1_score = np.round(2 * precision * recall / (precision + recall), 4)
    with open(result_store, 'a', newline='') as csvfile:
        fieldnames = ['fold', 'accuracy', 'precision', 'recall', 'f1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(
            {'fold': kfold, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1_score})
    return accuracy, precision, recall, f1_score


def current_time():
    """ taking the current system time"""
    cur_time = time.strftime("%Y_%m_%d_%H_%M", time.gmtime())
    return cur_time


def one_hot_encoding(arr, tr_index, te_index):
    num_of_class = len(np.unique(arr))
    min_val = np.min(arr)
    arr = arr - min_val
    tr_encoded_array = tf.keras.utils.to_categorical(arr[tr_index], num_classes=num_of_class)
    te_encoded_array = tf.keras.utils.to_categorical(arr[te_index], num_classes=num_of_class)
    return tr_encoded_array, te_encoded_array


def makedirs(path):
    """
    create directory on the "path name" """

    if not os.path.exists(path):
        os.makedirs(path)


def get_train_test_index(data, kf):
    train_index = []
    test_index = []
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)
    return train_index, test_index


def extract_feature(x_original, featureset_size, batch_super, extract_layer, isTrain=False, drop_out=0.0):
    feature_set = np.zeros((1, featureset_size), dtype=int)
    length = np.shape(x_original)[0]
    steps = length // batch_super + 1

    for j in range(steps):
        signal_batch = x_original[np.mod(np.arange(j * batch_super, (j + 1) * batch_super), length)]
        signal_batch = signal_batch.reshape(np.shape(signal_batch)[0], np.shape(signal_batch)[1], 1)
        fetched = extract_layer(signal_batch, training=False)
        feature_set = np.vstack((feature_set, fetched))

    x_feature = feature_set[1:length + 1]
    return x_feature


"""def extract_feature(x_original, featureset_size, batch_super, input_tensor, isTrain, drop_out, extract_layer):
    feature_set = np.zeros((1, featureset_size), dtype=int)  # 初始化 feature_set
    length = np.shape(x_original)[0]  # 获取信号长度
    steps = length // batch_super + 1  # 确定 batch 数量

    for j in range(steps):
        # 使用 np.mod() 来确保批次索引不会越界
        signal_batch = x_original[np.mod(np.arange(j * batch_super, (j + 1) * batch_super), length)]
        signal_batch = signal_batch.reshape(np.shape(signal_batch)[0], np.shape(signal_batch)[1], 1)  # reshape 输入
        # 在 TensorFlow 2.x 中，直接调用模型
        fetched = extract_layer(signal_batch, training=False)  # 直接调用模型
        feature_set = np.vstack((feature_set, fetched))  # 将结果加入 feature_set

    x_feature = feature_set[1:length + 1]  # 重新调整 feature_set 的大小以匹配原始信号的长度
    return x_feature"""
