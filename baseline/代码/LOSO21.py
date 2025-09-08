import os
import logging
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import data_preprocessing
import utils
from utils import weighted_bce_loss
from my_model import FeatureExtractor, SelfSupervisedModel

# ---- 新的 6 种增强 ----
import signal_transformation_task as st
print("🧠 是否使用GPU:", tf.config.list_physical_devices('GPU'))

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy("mixed_float16")

# 只输出错误
tf.get_logger().setLevel(logging.ERROR)

# 路径
dirname = os.getcwd()
data_folder = os.path.join(os.path.dirname(dirname), 'data_folder')
summaries   = os.path.join(os.path.dirname(dirname), 'summaries')
output      = os.path.join(os.path.dirname(dirname), 'output')
model_dir   = os.path.join(os.path.dirname(dirname), 'models')

# 基础目录
utils.makedirs(os.path.join(output, "STR_result"))
utils.makedirs(os.path.join(output, "STR_loss"))
utils.makedirs(model_dir)

# ----------------- 配置 -----------------
no_of_task        = st.TASK_NAMES
transform_task    = st.TASK_IDS
single_batch_size = len(transform_task)

batchsize = 64
epoch     = 100
initial_learning_rate = 1e-4
drop_rate = 0.3
lr_decay_rate = 0.8
loss_coeff = [0.195, 0.195, 0.195, 0.0125, 0.0125, 0.195, 0.195]
window_size = 2560
current_time = utils.current_time()
calculated_pos_weight = 6.0

# ----------------- 加载数据 -----------------
wesad_data = data_preprocessing.load_data(os.path.join(data_folder, 'wesad_dict.npy'))
print(f"原始数据集加载完毕，形状: {wesad_data.shape}")
subject_ids = np.unique(wesad_data[:, 0])
# ===== 新增：限制折数（只跑前 N 折）=====
N_FOLDS = 2   # 改成你想要的折数
subject_ids = subject_ids[:N_FOLDS]
total_subjects = len(subject_ids)

best_model_paths_per_fold = []
best_f1s_per_fold = []
fold_histories = {}  # 用于每折的曲线

# ----------------- 函数：每折曲线 & 下游曲线 -----------------
import matplotlib.pyplot as plt

def plot_history(history_dict, subject_id, output_dir):
    """画自监督每折的 loss/acc/f1（均为平均值）"""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(18, 5))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history_dict['train_loss'], label='train_loss')
    plt.plot(history_dict['val_loss'],   label='val_loss')
    plt.title(f'LOSO {int(subject_id)} - Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    # Acc
    plt.subplot(1, 3, 2)
    plt.plot(history_dict['train_acc'], label='train_acc')
    plt.plot(history_dict['val_acc'],   label='val_acc')
    plt.title(f'LOSO {int(subject_id)} - Acc');  plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()

    # F1
    plt.subplot(1, 3, 3)
    plt.plot(history_dict['train_f1'], label='train_f1')
    plt.plot(history_dict['val_f1'],   label='val_f1')
    plt.title(f'LOSO {int(subject_id)} - F1');   plt.xlabel('Epoch'); plt.ylabel('F1'); plt.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'subject_{int(subject_id)}_history.png')
    plt.savefig(save_path); plt.close()

def plot_metrics(history, output_dir, identifier):
    """画下游监督训练的 loss/acc/f1（来自 supervised head）"""
    os.makedirs(output_dir, exist_ok=True)
    # history 为 dict：['loss','val_loss','accuracy','val_accuracy','f1_score','val_f1_score']
    epochs = range(1, len(history['loss'])+1)

    plt.figure(figsize=(18,5))
    # Loss
    plt.subplot(1,3,1)
    plt.plot(epochs, history['loss'],     label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.title(f'{identifier} - Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()

    # Acc
    plt.subplot(1,3,2)
    plt.plot(epochs, history['accuracy'],     label='train_accuracy')
    plt.plot(epochs, history['val_accuracy'], label='val_accuracy')
    plt.title(f'{identifier} - Accuracy'); plt.xlabel('Epochs'); plt.ylabel('Accuracy'); plt.legend()

    # F1
    plt.subplot(1,3,3)
    plt.plot(epochs, history['f1_score'],     label='train_f1')
    plt.plot(epochs, history['val_f1_score'], label='val_f1')
    plt.title(f'{identifier} - F1'); plt.xlabel('Epochs'); plt.ylabel('F1'); plt.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{identifier}_history.png')
    plt.savefig(save_path); plt.close()

# ----------------- 自监督训练（LOSO） -----------------
for k, test_subject in enumerate(subject_ids):
    print(f'========== 开始第 {k+1}/{total_subjects} 折 (留出: {int(test_subject)}) ==========')

    model = SelfSupervisedModel(drop_rate=drop_rate, num_tasks=len(transform_task), hidden_nodes=128, stride_mp=4)

    train_mask = wesad_data[:, 0] != test_subject
    test_mask  = wesad_data[:, 0] == test_subject
    train_ECG  = wesad_data[train_mask, 2:]
    test_ECG   = wesad_data[test_mask, 2:]

    if train_ECG.shape[0] == 0:
        print(f"警告：训练集为空，跳过受试者 {test_subject}")
        continue

    train_ECG = shuffle(train_ECG, random_state=42)
    training_length = train_ECG.shape[0]
    testing_length  = test_ECG.shape[0]

    steps_per_epoch = int(np.ceil(training_length / batchsize))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=steps_per_epoch * 5,
        decay_rate=lr_decay_rate,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # 每折日志（用于画图）
    fold_histories[test_subject] = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  [],
        'train_f1':   [], 'val_f1':   []
    }

    patience = 10
    patience_counter = 0
    best_val_f1_avg = -1.0
    best_model_weights_path = os.path.join(model_dir, f"best_ssl_model_subject_{int(test_subject)}.weights.h5")

    print('自监督训练开始...')
    for epoch_counter in tqdm(range(epoch), desc=f"Subject {int(test_subject)}"):
        # ---- 新增强（训练）----
        tr_total_gen_op = st.make_batch(data=train_ECG, length=training_length, batchsize=batchsize)

        tr_epoch_loss = 0.0
        tr_steps = 0
        train_preds = [[] for _ in range(len(transform_task))]
        train_labels = [[] for _ in range(len(transform_task))]

        for training_batch, training_labels_orig, _, _ in tr_total_gen_op:
            training_batch, training_labels_orig = utils.unison_shuffled_copies(training_batch, training_labels_orig)
            training_batch = training_batch.reshape(training_batch.shape[0], training_batch.shape[1], 1).astype(np.float32)

            with tf.GradientTape() as tape:
                predictions = model(training_batch, training=True)
                task_outputs = predictions[4:]

                current_task_losses = []
                for i in range(len(transform_task)):
                    y_true = training_labels_orig[:, i:i+1]
                    y_pred = task_outputs[i]
                    loss_fn = weighted_bce_loss(calculated_pos_weight)
                    loss_i = tf.reduce_mean(loss_fn(y_true, y_pred))
                    current_task_losses.append(loss_i)

                loss = utils.get_weighted_loss(loss_coeff, current_task_losses)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            tr_epoch_loss += float(loss.numpy())
            for i in range(len(transform_task)):
                preds  = (tf.sigmoid(task_outputs[i]).numpy() > 0.5).astype(int).flatten()
                labels = training_labels_orig[:, i:i+1].flatten().astype(int)
                train_preds[i].extend(preds)
                train_labels[i].extend(labels)
            tr_steps += 1

        # ---- 新增强（验证）----
        te_total_gen_op = st.make_batch(data=test_ECG, length=testing_length, batchsize=batchsize)
        val_task_losses = [0.0] * len(transform_task)
        val_steps = 0
        val_preds = [[] for _ in range(len(transform_task))]
        val_labels = [[] for _ in range(len(transform_task))]

        for testing_batch, testing_labels_orig, _, _ in te_total_gen_op:
            testing_batch = testing_batch.reshape(testing_batch.shape[0], testing_batch.shape[1], 1).astype(np.float32)
            predictions = model(testing_batch, training=False)
            task_outputs = predictions[4:]

            for i in range(len(transform_task)):
                y_true = testing_labels_orig[:, i:i+1]
                y_pred = task_outputs[i]
                loss_fn = weighted_bce_loss(calculated_pos_weight)
                loss_i = tf.reduce_mean(loss_fn(y_true, y_pred)).numpy()
                val_task_losses[i] += loss_i

                preds  = (tf.sigmoid(y_pred).numpy() > 0.5).astype(int).flatten()
                labels = testing_labels_orig[:, i:i+1].flatten().astype(int)
                val_preds[i].extend(preds)
                val_labels[i].extend(labels)

            val_steps += 1

        # ---- 统计平均 ----
        avg_train_loss = tr_epoch_loss / max(1, tr_steps)
        avg_val_loss   = np.sum(val_task_losses) / max(1, val_steps)

        train_accs, train_f1s = [], []
        val_accs, val_f1s = [], []
        for i in range(len(transform_task)):
            train_accs.append(accuracy_score(train_labels[i], train_preds[i]))
            train_f1s.append(f1_score(train_labels[i], train_preds[i], average='binary', zero_division=0))
            val_accs.append(accuracy_score(val_labels[i], val_preds[i]))
            val_f1s.append(f1_score(val_labels[i], val_preds[i], average='binary', zero_division=0))

        avg_train_acc = float(np.mean(train_accs)); avg_train_f1 = float(np.mean(train_f1s))
        avg_val_acc   = float(np.mean(val_accs));   avg_val_f1   = float(np.mean(val_f1s))

        # 存到曲线字典
        fold_histories[test_subject]['train_loss'].append(avg_train_loss)
        fold_histories[test_subject]['val_loss'].append(avg_val_loss)
        fold_histories[test_subject]['train_acc'].append(avg_train_acc)
        fold_histories[test_subject]['val_acc'].append(avg_val_acc)
        fold_histories[test_subject]['train_f1'].append(avg_train_f1)
        fold_histories[test_subject]['val_f1'].append(avg_val_f1)

        tqdm.write(f"[{int(test_subject)}] Epoch {epoch_counter+1} | "
                   f"TrainLoss {avg_train_loss:.4f} | ValLoss {avg_val_loss:.4f} | "
                   f"TrainAcc {avg_train_acc:.4f} | TrainF1 {avg_train_f1:.4f} | "
                   f"ValAcc {avg_val_acc:.4f} | ValF1 {avg_val_f1:.4f}")

        # 早停 + 保存
        if avg_val_f1 > best_val_f1_avg:
            best_val_f1_avg = avg_val_f1
            patience_counter = 0
            model.save_weights(best_model_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(f"早停触发，停止在 Epoch {epoch_counter+1}")
                break

    best_model_paths_per_fold.append(best_model_weights_path)
    best_f1s_per_fold.append(best_val_f1_avg)
    print(f"第 {k+1} 折完成，最佳 ValF1 = {best_val_f1_avg:.4f}")

# 选全局最佳
best_fold_index = int(np.argmax(best_f1s_per_fold))
best_overall_model_path = best_model_paths_per_fold[best_fold_index]
print(f"全局最佳来自折 {best_fold_index+1}，F1={best_f1s_per_fold[best_fold_index]:.4f}")

# ----------------- 下游监督头（含 history，便于画图） -----------------
def supervised_model_wesad(
    x_tr_feature, y_tr,
    x_te_feature, y_te,
    identifier='wesad_affect',
    hidden=256, dropout=0.3, lr=1e-3,
    epochs=80, batch_size=128, patience=15
):
    import tensorflow as tf
    from sklearn.metrics import f1_score, accuracy_score
    import os

    num_classes = y_tr.shape[1]
    feat_dim = x_tr_feature.shape[1]

    inputs = tf.keras.Input(shape=(feat_dim,), dtype=tf.float32)
    x = tf.keras.layers.Dense(hidden, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(hidden//2, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', dtype=tf.float32)(x)
    clf = tf.keras.Model(inputs, outputs)
    clf.compile(optimizer=tf.keras.optimizers.Adam(lr),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # 自定义 callback 记录 F1
    class F1History(tf.keras.callbacks.Callback):
        def __init__(self, x_tr, y_tr, x_val, y_val):
            super().__init__()
            self.x_tr, self.y_tr = x_tr, y_tr
            self.x_val, self.y_val = x_val, y_val
            self.train_f1, self.val_f1 = [], []
        def on_epoch_end(self, epoch, logs=None):
            tr_prob = self.model.predict(self.x_tr, verbose=0)
            va_prob = self.model.predict(self.x_val, verbose=0)
            tr_pred = tr_prob.argmax(axis=1); va_pred = va_prob.argmax(axis=1)
            tr_true = self.y_tr.argmax(axis=1); va_true = self.y_val.argmax(axis=1)
            f1_tr = f1_score(tr_true, tr_pred, average='macro', zero_division=0)
            f1_va = f1_score(va_true, va_pred, average='macro', zero_division=0)
            self.train_f1.append(f1_tr); self.val_f1.append(f1_va)
            if logs is not None:
                logs['f1_score'] = f1_tr; logs['val_f1_score'] = f1_va

    # 划 20% 验证
    n = x_tr_feature.shape[0]; n_val = max(1, int(n*0.2))
    x_val, y_val = x_tr_feature[:n_val], y_tr[:n_val]
    x_train, y_train = x_tr_feature[n_val:], y_tr[n_val:]

    cb_f1 = F1History(x_train, y_train, x_val, y_val)
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=max(3, patience//5), min_lr=1e-5, verbose=1),
        tf.keras.callbacks.EarlyStopping('val_loss', patience=patience, restore_best_weights=True, verbose=1),
        cb_f1
    ]
    hist = clf.fit(x_train, y_train, validation_data=(x_val, y_val),
                   epochs=epochs, batch_size=batch_size, verbose=2, callbacks=cbs)

    te_prob = clf.predict(x_te_feature, verbose=0)
    te_pred = te_prob.argmax(axis=1); te_true = y_te.argmax(axis=1)
    acc = accuracy_score(te_true, te_pred)
    f1  = f1_score(te_true, te_pred, average='macro', zero_division=0)

    history = {
        'loss':         hist.history.get('loss', []),
        'val_loss':     hist.history.get('val_loss', []),
        'accuracy':     hist.history.get('accuracy', []),
        'val_accuracy': hist.history.get('val_accuracy', []),
        'f1_score':     cb_f1.train_f1,
        'val_f1_score': cb_f1.val_f1
    }
    return acc, f1, history

# ----------------- 迁移学习：抽特征 + 下游评估 + 画图 -----------------
print('========== 迁移学习开始 ==========')
final_model = SelfSupervisedModel(drop_rate=drop_rate, num_tasks=len(transform_task), hidden_nodes=128, stride_mp=4)
_ = final_model(tf.zeros((1, window_size, 1)))      # build
final_model.load_weights(best_overall_model_path)
feature_extractor = FeatureExtractor(final_model)

x_tr_total = wesad_data[:, 2:]
y_tr_total = wesad_data[:, 1]

dummy_input = tf.zeros((1, window_size, 1), dtype=tf.float32)
feature_size = feature_extractor(dummy_input, training=False).shape[-1]
x_tr_feature = utils.extract_feature(x_original=x_tr_total, featureset_size=feature_size,
                                     batch_super=batchsize, extract_layer=feature_extractor)

x_train, x_test, y_train, y_test = train_test_split(
    x_tr_feature, y_tr_total, test_size=0.3, random_state=42, stratify=y_tr_total
)
num_classes = int(np.max(y_tr_total)) + 1
y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test_onehot  = tf.keras.utils.to_categorical(y_test,  num_classes=num_classes)

acc, f1, sup_history = supervised_model_wesad(
    x_tr_feature=x_train, y_tr=y_train_onehot,
    x_te_feature=x_test,  y_te=y_test_onehot,
    identifier='wesad_affect'
)

print('========== 最终结果 ==========')
print(f"下游任务 Acc: {acc:.4f}")
print(f"下游任务 F1 : {f1:.4f}")

# --- 画图：每折 ---
figure_dir = os.path.join(output, "STR_result", "figures")
utils.makedirs(figure_dir)
print("\n正在生成每一折的性能曲线图...")
for subject_id, hist in fold_histories.items():
    plot_history(hist, subject_id, figure_dir)
print("每折曲线图目录:", figure_dir)

# --- 画图：下游 ---
downstream_figure_dir = os.path.join(output, "STR_result", "blsa_figures")
utils.makedirs(downstream_figure_dir)
print("\n正在生成下游任务的性能曲线图...")
plot_metrics(sup_history, output_dir=downstream_figure_dir, identifier='wesad_affect')
print("下游曲线图目录:", downstream_figure_dir)
