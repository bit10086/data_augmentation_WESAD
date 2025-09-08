import os
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import logging
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from tensorflow.keras.metrics import Precision, Recall, AUC, BinaryAccuracy
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import my_model
from my_model import FeatureExtractor, SelfSupervisedModel
import utils
from utils import weighted_bce_loss
import data_preprocessing

# 设置只输出错误日志
tf.get_logger().setLevel(logging.ERROR)

dirname = os.getcwd()
data_folder = os.path.join(os.path.dirname(dirname), 'data_folder')
summaries = os.path.join(os.path.dirname(dirname), 'summaries')
output = os.path.join(os.path.dirname(dirname), 'output')
model_dir = os.path.join(os.path.dirname(dirname), 'models')

# 确保必要的目录存在
utils.makedirs(os.path.join(output, "STR_result"))
utils.makedirs(os.path.join(output, "STR_loss"))
utils.makedirs(os.path.join(model_dir))

## transformation task params
noise_param = 15
scale_param = 0.9
permu_param = 20
tw_piece_param = 9
twsf_param = 1.06
no_of_task = [
    'original_signal',     # 0
    'noised_signal',       # 1
    'scaled_signal',       # 2
    'time_masked_signal',  # 3  ← 替换 negated
    'dgw_warped_signal',   # 4  ← 替换 flipped
    'croppad_signal',      # 5  ← 强化 permuted -> 裁剪+填充
    'time_warped_signal'   # 6  (保留随机时间扭曲)
]
transform_task = [0, 1, 2, 3, 4, 5, 6]
# 可选：新增强的可调范围（如果要从脚本传给 utils 里用）
time_mask_width_pct = (0.05, 0.20)  # 随机遮挡 5%~20% 窗口
croppad_keep_pct    = (0.60, 0.95)  # 随机保留 60%~95%，剩余两侧补齐
dgw_max_warp        = 0.30          # 近似 DGW 的最大非线性位移比例

single_batch_size = len(transform_task)

## hyper parameters
batchsize = 64
actual_batch_size = batchsize * single_batch_size
log_step = 100
epoch = 100
##initial_learning_rate = 0.001
initial_learning_rate = 0.0001
##drop_rate = 0.6
drop_rate = 0.3
regularizer = 1
##L2 = 0.0001
L2 = 0.001
##lr_decay_steps = 10000
lr_decay_rate = 0.8
loss_coeff = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]
window_size = 2560
extract_data = 0
current_time = utils.current_time()
calculated_pos_weight = 6.0

if extract_data == 1:
    _ = data_preprocessing.extract_wesad_dataset(overlap_pct=0, window_size_sec=10, data_save_path=data_folder, save=1)

# 加载数据集
wesad_data = data_preprocessing.load_data(os.path.join(data_folder, 'wesad_dict.npy'))
print(f"原始数据集加载完毕，形状: {wesad_data.shape}")

# 获取所有独特的受试者ID
subject_ids = np.unique(wesad_data[:, 0])

# ===== 新增：限制折数（只跑前 N 折）=====
N_FOLDS =5   # 改成你想要的折数
subject_ids = subject_ids[:N_FOLDS]
total_subjects = len(subject_ids)

# 用来存储每折的最佳模型路径和性能
best_model_paths_per_fold = []
best_f1s_per_fold = []

# 用来存储每折下游任务的最终结果
final_downstream_accs = []
final_downstream_f1s = []

# 用于绘制曲线图的数据
fold_histories = {}

# 1. 留一受试者交叉验证主循环
for k, test_subject in enumerate(subject_ids):
    ##for k, test_subject in enumerate(subject_ids_to_run):
    print(
        f'===================== 正在开始第 {k + 1}/{total_subjects} 折 (留出受试者: {test_subject}) =====================')

    # 为每一折创建独立的 TensorBoard 目录
    str_logs = os.path.join(summaries, f"STR_subject_{int(test_subject)}", current_time)
    utils.makedirs(str_logs)
    summary_writer = tf.summary.create_file_writer(str_logs)

    # 在每折开始时，重新实例化模型和优化器，确保每次训练都是独立的
    my_self_supervised_model = SelfSupervisedModel(drop_rate=drop_rate, num_tasks=len(transform_task), hidden_nodes=128,
                                                   stride_mp=4)
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #initial_learning_rate=initial_learning_rate,  # 初始学习率
    #decay_steps=lr_decay_steps,                   # 每多少步衰减一次
    #decay_rate=lr_decay_rate,                     # 衰减系数
    #staircase=True                                # 如果为 True，则按阶梯式衰减；False 为连续衰减
    #)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 根据受试者ID划分训练集和测试集
    train_mask = wesad_data[:, 0] != test_subject
    test_mask = wesad_data[:, 0] == test_subject
    train_ECG = wesad_data[train_mask, 2:]
    test_ECG = wesad_data[test_mask, 2:]

    # 确保训练数据不为空
    if train_ECG.shape[0] == 0:
        print(f"警告：训练集为空，跳过受试者 {test_subject} 的处理。")
        continue

    train_ECG = shuffle(train_ECG, random_state=42)
    training_length = train_ECG.shape[0]
    testing_length = test_ECG.shape[0]
    # 在确定 train_ECG / training_length 之后再写：
    steps_per_epoch = int(np.ceil(training_length / batchsize))
    decay_every_n_epochs = 5  # 每 5 个 epoch 衰减一次，可调 3~10
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                  initial_learning_rate=initial_learning_rate,
                  decay_steps=steps_per_epoch * decay_every_n_epochs,
                  decay_rate=lr_decay_rate,   # 你现在是 0.9；想更明显可用 0.5~0.8
                  staircase=True
                   )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 早停机制变量
    patience = 100
    patience_counter = 0
    best_val_f1_avg = -1.0
    best_model_weights_path = os.path.join(model_dir, f"best_ssl_model_subject_{int(test_subject)}.weights.h5")

    # 用于保存本折的训练历史数据
    fold_histories[test_subject] = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [],
                                    'val_f1': []}

    # 训练循环
    print('自监督训练开始...')
    for epoch_counter in tqdm(range(epoch), desc=f"Subject {int(test_subject)} Epochs"):
        tr_total_gen_op = utils.make_total_batch(data=train_ECG, length=training_length, batchsize=batchsize,
                                                 noise_amount=noise_param, scaling_factor=scale_param,
                                                 permutation_pieces=permu_param, time_warping_pieces=tw_piece_param,
                                                 time_warping_stretch_factor=twsf_param,
                                                 time_warping_squeeze_factor=1 / twsf_param)

        # 训练步骤
        tr_epoch_loss = 0.0
        tr_task_losses = [0.0] * len(transform_task)
        tr_steps = 0
        train_preds = [[] for _ in range(len(transform_task))]
        train_labels = [[] for _ in range(len(transform_task))]

        for training_batch, training_labels_orig, _, _ in tr_total_gen_op:
            training_batch, training_labels_orig = utils.unison_shuffled_copies(training_batch, training_labels_orig)
            training_batch = training_batch.reshape(training_batch.shape[0], training_batch.shape[1], 1).astype(
                np.float32)

            with tf.GradientTape() as tape:
                predictions = my_self_supervised_model(training_batch, training=True)
                task_outputs_from_model = predictions[4:]

                # 计算总损失
                current_task_losses = []
                for i in range(len(transform_task)):
                    current_task_true_labels = training_labels_orig[:, i:i + 1]
                    current_task_predictions = task_outputs_from_model[i]
                    loss_fn = weighted_bce_loss(calculated_pos_weight)
                    loss_i = loss_fn(current_task_true_labels, current_task_predictions)
                    current_task_losses.append(tf.reduce_mean(loss_i))

                loss = utils.get_weighted_loss(loss_coeff, current_task_losses)

            gradients = tape.gradient(loss, my_self_supervised_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, my_self_supervised_model.trainable_variables))

            tr_epoch_loss += loss.numpy()
            for i in range(len(transform_task)):
                tr_task_losses[i] += current_task_losses[i].numpy()
                preds = (tf.sigmoid(task_outputs_from_model[i]).numpy() > 0.5).astype(int).flatten()
                labels = training_labels_orig[:, i:i + 1].flatten().astype(int)
                train_preds[i].extend(preds)
                train_labels[i].extend(labels)

            tr_steps += 1

        # 验证步骤
        val_epoch_loss = 0.0
        val_task_losses = [0.0] * len(transform_task)
        val_steps = 0
        val_preds = [[] for _ in range(len(transform_task))]
        val_labels = [[] for _ in range(len(transform_task))]

        te_total_gen_op = utils.make_total_batch(data=test_ECG, length=testing_length, batchsize=batchsize,
                                                 noise_amount=noise_param, scaling_factor=scale_param,
                                                 permutation_pieces=permu_param, time_warping_pieces=tw_piece_param,
                                                 time_warping_stretch_factor=twsf_param,
                                                 time_warping_squeeze_factor=1 / twsf_param)

        for testing_batch, testing_labels_orig, _, _ in te_total_gen_op:
            testing_batch = testing_batch.reshape(testing_batch.shape[0], testing_batch.shape[1], 1).astype(np.float32)
            predictions = my_self_supervised_model(testing_batch, training=False)
            task_outputs_from_model = predictions[4:]

            for i in range(len(transform_task)):
                current_task_true_labels = testing_labels_orig[:, i:i + 1]
                current_task_predictions = task_outputs_from_model[i]
                loss_fn = weighted_bce_loss(calculated_pos_weight)
                loss_i = loss_fn(current_task_true_labels, current_task_predictions)
                val_task_losses[i] += tf.reduce_mean(loss_i).numpy()

                preds = (tf.sigmoid(current_task_predictions).numpy() > 0.5).astype(int).flatten()
                labels = testing_labels_orig[:, i:i + 1].flatten().astype(int)
                val_preds[i].extend(preds)
                val_labels[i].extend(labels)

            val_steps += 1

        # 计算 epoch 级别平均指标
        avg_train_loss = tr_epoch_loss / tr_steps
        avg_val_loss = np.sum(val_task_losses) / val_steps

        train_accs, train_f1s = [], []
        for i in range(len(transform_task)):
            train_accs.append(accuracy_score(train_labels[i], train_preds[i]))
            train_f1s.append(f1_score(train_labels[i], train_preds[i], average='binary', zero_division=0))

        val_accs, val_f1s = [], []
        for i in range(len(transform_task)):
            val_accs.append(accuracy_score(val_labels[i], val_preds[i]))
            val_f1s.append(f1_score(val_labels[i], val_preds[i], average='binary', zero_division=0))

        avg_train_acc = np.mean(train_accs)
        avg_train_f1 = np.mean(train_f1s)
        avg_val_acc = np.mean(val_accs)
        avg_val_f1 = np.mean(val_f1s)

        # 存储本折训练历史数据
        fold_histories[test_subject]['train_loss'].append(avg_train_loss)
        fold_histories[test_subject]['val_loss'].append(avg_val_loss)
        fold_histories[test_subject]['train_acc'].append(avg_train_acc)
        fold_histories[test_subject]['val_acc'].append(avg_val_acc)
        fold_histories[test_subject]['train_f1'].append(avg_train_f1)
        fold_histories[test_subject]['val_f1'].append(avg_val_f1)

        tqdm.write(f"\n折 {k + 1} (受试者 {int(test_subject)}), Epoch {epoch_counter + 1}:")
        tqdm.write(f"训练总损失: {avg_train_loss:.4f} | 验证总损失: {avg_val_loss:.4f}")
        tqdm.write(f"训练平均Acc: {avg_train_acc:.4f} | 训练平均F1: {avg_train_f1:.4f}")
        tqdm.write(f"验证平均Acc: {avg_val_acc:.4f} | 验证平均F1: {avg_val_f1:.4f}")

        # 早停逻辑：保存验证集上最好的模型
        if avg_val_f1 > best_val_f1_avg:
            best_val_f1_avg = avg_val_f1
            patience_counter = 0
            # 保存当前最佳模型权重
            my_self_supervised_model.save_weights(best_model_weights_path)
            tqdm.write(f"验证F1提高至 {best_val_f1_avg:.4f}。保存模型权重。")
        else:
            patience_counter += 1
            tqdm.write(f"验证F1没有显著提高。耐心: {patience_counter}/{patience}")

        if patience_counter >= patience:
            tqdm.write(f"早停触发，在第 {epoch_counter + 1} 个epoch后停止训练。")
            break

    # 保存这折的最佳模型路径和F1分数
    best_model_paths_per_fold.append(best_model_weights_path)
    best_f1s_per_fold.append(best_val_f1_avg)

    print(f"第 {k + 1} 折训练完成。")

# 3. 找到所有折中表现最好的模型
best_fold_index = np.argmax(best_f1s_per_fold)
best_overall_model_path = best_model_paths_per_fold[best_fold_index]
print(f"所有折中表现最好的模型来自第 {best_fold_index + 1} 折，F1分数为 {best_f1s_per_fold[best_fold_index]:.4f}")

# 4. 使用全局最佳模型进行迁移学习
print('===================== 迁移学习任务开始 =====================')
# 重新实例化一个干净的模型实例
final_model = SelfSupervisedModel(drop_rate=drop_rate, num_tasks=len(transform_task), hidden_nodes=128, stride_mp=4)

# 关键修正：在加载权重前，用一个虚拟输入构建模型。
dummy_input = tf.zeros((1, window_size, 1))
_ = final_model(dummy_input)

# 加载全局最佳模型的权重
final_model.load_weights(best_overall_model_path)
print(f"已加载全局最佳模型权重，路径为: {best_overall_model_path}")

# 实例化特征提取器
feature_extractor = FeatureExtractor(final_model)

# 准备下游任务数据
x_tr_total = wesad_data[:, 2:]
y_tr_total = wesad_data[:, 1]

# 提取特征
dummy_input = tf.zeros((1, window_size, 1), dtype=tf.float32)
feature_size = feature_extractor(dummy_input, training=False).shape[-1]
x_tr_feature = utils.extract_feature(x_original=x_tr_total, featureset_size=feature_size,
                                     batch_super=batchsize, extract_layer=feature_extractor)

print("下游任务：使用 train_test_split 划分训练集和测试集...")

# 导入 train_test_split
from sklearn.model_selection import train_test_split

# 步骤 1: 使用 train_test_split 进行数据划分（例如，70% 训练，30% 测试）
# stratify=y_tr_total 确保训练集和测试集中的类别分布保持一致
x_train, x_test, y_train, y_test = train_test_split(
    x_tr_feature, y_tr_total, test_size=0.3, random_state=42, stratify=y_tr_total
)

# 步骤 2: 转换为One-Hot编码
# 确保类别数正确
num_classes = int(np.max(y_tr_total)) + 1
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

# 步骤 3: 在这个函数内部完成训练和评估，并返回acc和f1
# 这里不再需要 LOSO 循环，直接调用一次模型
acc, f1 ,_= fusion2.supervised_model_wesad(
    x_tr_feature=x_train, y_tr=y_train_onehot,
    x_te_feature=x_test, y_te=y_test_onehot,
    identifier='wesad_affect',
    kfold=0,  # 因为不再是交叉验证，可以固定为 0
    summaries=os.path.join(summaries, 'ER'),
    current_time=current_time,
    result=os.path.join(output, "STR_result")
)

# 5. 输出最终结果
print('===================== 最终结果 =====================')
print(f"下游任务的准确率: {acc:.4f}")
print(f"下游任务的F1分数: {f1:.4f}")

# 6. 绘制每一折的曲线图
def plot_history(history, subject_id, output_dir):
    """
    绘制训练和验证的损失、准确率和F1曲线图，并将图表保存为文件。
    横纵坐标从0开始，横坐标每个格子代表1。
    """
    plt.figure(figsize=(18, 5))

    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title(f'LOSO: {int(subject_id)} loss_curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 设置横纵坐标从0开始，横坐标每个格子代表1
    plt.xlim(0, len(history['train_loss']))
    plt.ylim(0, max(max(history['train_loss']), max(history['val_loss'])) * 1.1)  # 设置纵坐标的最大值稍大一些
    plt.xticks(range(0, len(history['train_loss']) + 1, 1))  # 横坐标每个格子代表1

    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.title(f'LOSO: {int(subject_id)} acc_curve')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.legend()

    # 设置横纵坐标从0开始，横坐标每个格子代表1
    plt.xlim(0, len(history['train_acc']))
    plt.ylim(0, max(max(history['train_acc']), max(history['val_acc'])) * 1.1)  # 设置纵坐标的最大值稍大一些
    plt.xticks(range(0, len(history['train_acc']) + 1, 1))  # 横坐标每个格子代表1

    # 绘制F1分数曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='train_f1')
    plt.plot(history['val_f1'], label='val_f1')
    plt.title(f'LOSO: {int(subject_id)} F1_curve')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()

    # 设置横纵坐标从0开始，横坐标每个格子代表1
    plt.xlim(0, len(history['train_f1']))
    plt.ylim(0, max(max(history['train_f1']), max(history['val_f1'])) * 1.1)  # 设置纵坐标的最大值稍大一些
    plt.xticks(range(0, len(history['train_f1']) + 1, 1))  # 横坐标每个格子代表1

    plt.tight_layout()

    # 将图表保存为文件，而不是显示在屏幕上
    save_path = os.path.join(output_dir, f'subject_{int(subject_id)}_history.png')
    plt.savefig(save_path)
    plt.close()  # 关闭图表，释放内存



# 创建一个用于保存图片的目录
figure_dir = os.path.join(output, "STR_result", "figures")
utils.makedirs(figure_dir)

print("\n正在生成每一折的性能曲线图...")
for subject_id, history in fold_histories.items():
    plot_history(history, subject_id, figure_dir)

print("曲线图已保存到以下目录: " + os.path.join(output, "STR_result", "figures"))
print("请使用 TensorBoard 查看更详细的日志和动态图。")
print("运行命令：tensorboard --logdir=" + summaries)

def plot_metrics(history, output_dir, identifier):
    """
    绘制损失、准确率和F1分数的训练和验证曲线，并保存为图像文件。
    """
    epochs = range(1, len(history['accuracy']) + 1)
    
    # 绘制 Loss 曲线
    plt.figure(figsize=(18, 5))

    # 绘制训练和验证的损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.title(f'{identifier} Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制 Accuracy 曲线
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['accuracy'], label='train_accuracy')
    plt.plot(epochs, history['val_accuracy'], label='val_accuracy')
    plt.title(f'{identifier} Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制 F1 曲线
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['f1_score'], label='train_f1')
    plt.plot(epochs, history['val_f1_score'], label='val_f1')
    plt.title(f'{identifier} F1 Score Curve')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # 调整布局，使得图表不会重叠
    plt.tight_layout()

    # 保存图表为文件
    save_path = os.path.join(output_dir, f'{identifier}_history.png')
    plt.savefig(save_path)
    plt.close()

# 获取模型训练的历史（假设你已经有了history）
history = fusion.supervised_model_wesad(
    x_tr_feature=x_train, y_tr=y_train_onehot,
    x_te_feature=x_test, y_te=y_test_onehot,
    identifier='wesad_affect',
    kfold=0,  # 因为不再是交叉验证，可以固定为 0
    summaries=os.path.join(summaries, 'ER'),
    current_time=current_time,
    result=os.path.join(output, "STR_result")
)


print("\n正在生成下游任务的性能曲线图...")
downstream_figure_dir = os.path.join(output, "STR_result", "blsa_figures")
downstream_save_path = os.path.join(downstream_figure_dir, 'downstream_history.png')
###plot_downstream_history(history, downstream_save_path)
plot_metrics(history, output_dir=downstream_figure_dir, identifier='wesad_affect')
print(f"下游任务曲线图已保存到: {downstream_save_path}")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

