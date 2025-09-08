import tensorflow as tf
import utils
import keras
from tensorflow import keras
import os
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import metrics

window_size = 2560
transform_task = [0, 1, 2, 3, 4, 5, 6]


# --- 定义 ConvBlock 和 DenseBlock 为 tf.keras.layers.Layer 子类 ---


# 使用 Sequential 的方式
class ConvBlock(keras.layers.Layer):
    def __init__(self, filters, kernel_size, pool_size, stride_mp, batch_norm=False, dropout=False, dropout_rate=0.0,
                 name="conv_block", **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.stride_mp = stride_mp
        self.batch_norm_enabled = batch_norm
        self.dropout_enabled = dropout
        self.dropout_rate = dropout_rate
        self.conv1 = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')
        self.relu1 = layers.LeakyReLU()
        if self.batch_norm_enabled:
            self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding='same')
        self.relu2 = layers.LeakyReLU()
        if self.batch_norm_enabled:
            self.bn2 = layers.BatchNormalization()
        ##self.max_pool = layers.MaxPooling1D(pool_size=pool_size)
        ###self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
        ##self.max_pool = layers.MaxPooling1D(pool_size=self.pool_size)
        # 只有当 pool_size 有效且大于1时才创建 MaxPooling1D 层
        if self.dropout_enabled and self.dropout_rate > 0.0:
            self.dropout_layer = layers.Dropout(rate=self.dropout_rate)
        else:
            self.dropout_enabled = False  # 如果没有dropout_rate或dropout=False，则禁用它

    def call(self, inputs, training=True):  # 确保 training 参数传递给 Dropout 等层 (如果有的话)
        x = self.conv1(inputs)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        ###x = self.max_pool(x)
        return x


######这个 dense_block 函数就是:“全连接层 + Leaky ReLU 激活 + Dropout” 的组合模块
"""def dense_block(input_tensor, hidden_nodes, drop_rate, isTraining, name):
    reuse = tf.compat.v1.AUTO_REUSE
    ###创建一个全连接层（Dense Layer），输入为 input_tensor，输出维度为 hidden_nodes。
    # dense       = tf.layers.dense  (inputs=input_tensor, units=hidden_nodes, reuse=reuse,  name= name )
    dense = tf.keras.layers.Dense(units=hidden_nodes, name=name)(input_tensor)
    ###对 Dense 层的输出施加 Leaky ReLU 激活函数，保留负值的小梯度（相比 ReLU 可减少“神经元死亡”问题）。
    dense = tf.nn.leaky_relu(dense)
    ###训练时随机丢弃一定比例（drop_rate）的神经元，防止过拟合；测试时不会丢弃。
    # dense       = tf.layers.dropout(inputs=dense, rate=drop_rate, training=isTraining, name= name)
    dense = tf.keras.layers.Dropout(rate=drop_rate, name=name)(dense, training=isTraining)
    return dense"""


def dense_block_layer(hidden_nodes, drop_rate, name):
    # 返回一个可调用的 Keras Layer 序列
    layers = []
    layers.append(tf.keras.layers.Dense(units=hidden_nodes, name=name + '_dense'))
    layers.append(tf.keras.layers.LeakyReLU(name=name + '_relu'))
    layers.append(tf.keras.layers.Dropout(rate=drop_rate, name=name + '_dropout'))
    return tf.keras.Sequential(layers, name=name)


# --- 将 self_supervised_model 封装成 tf.keras.Model 子类 ---
class SelfSupervisedModel(tf.keras.Model):
    def __init__(self, drop_rate, num_tasks, stride_mp, hidden_nodes=128, **kwargs):
        super(SelfSupervisedModel, self).__init__(**kwargs)
        self.drop_rate = drop_rate
        ##self.hidden_nodes = hidden_nodes
        self.stride_mp = stride_mp
        self.num_tasks = num_tasks

        # 定义所有的层作为类的属性
        self.conv_layer_1 = ConvBlock(filters=32, kernel_size=32, pool_size=8, stride_mp=1, batch_norm=False,
                                      dropout=False, dropout_rate=0.0,
                                      name='conv_layer_1')  # pool_size 从 max_pool_1 移到这里
        self.conv_layer_2 = ConvBlock(filters=32, kernel_size=32, pool_size=8, stride_mp=1, batch_norm=False,
                                      dropout=False, dropout_rate=0.0, name='conv_layer_2')  # 再次使用 ConvBlock
        self.max_pool_1 = tf.keras.layers.MaxPooling1D(pool_size=8, strides=self.stride_mp, padding='valid', name='mp1')
        self.conv_layer_3 = ConvBlock(filters=64, kernel_size=16, pool_size=8, stride_mp=1, batch_norm=False,
                                      dropout=False, dropout_rate=0.0,
                                      name='conv_layer_3')  # 你需要为每个 ConvBlock 实例决定一个 pool_sizebatch_norm=False,
        self.conv_layer_4 = ConvBlock(filters=64, kernel_size=16, pool_size=8, stride_mp=1, batch_norm=False,
                                      dropout=False, dropout_rate=0.0, name='conv_layer_4')
        self.max_pool_2 = tf.keras.layers.MaxPooling1D(pool_size=8, strides=self.stride_mp, padding='same', name='mp2')
        self.conv_layer_5 = ConvBlock(filters=128, kernel_size=8, pool_size=8, stride_mp=1, batch_norm=False,
                                      dropout=False, dropout_rate=0.0, name='conv_layer_5')
        self.conv_layer_6 = ConvBlock(filters=128, kernel_size=8, pool_size=8, stride_mp=1, batch_norm=False,
                                      dropout=False, dropout_rate=0.0, name='conv_layer_6')
        # GAP 层
        # 这些池化层需要动态计算 pool_size，或者使用 GlobalMaxPooling1D / GlobalAveragePooling1D
        # 这里为了简化，我们改为 GlobalMaxPooling1D，它不需要 pool_size
        self.gap1 = tf.keras.layers.GlobalMaxPooling1D(name='GAP1')
        self.gap2 = tf.keras.layers.GlobalMaxPooling1D(name='GAP2')
        self.gap3 = tf.keras.layers.GlobalMaxPooling1D(name='GAP3')
        self.gap_final = tf.keras.layers.GlobalMaxPooling1D(name='GAP_final')  # 原来的 GAP 层
        # Flatten 层
        self.flatten = tf.keras.layers.Flatten()

        self.task_dense_layers = {}
        for i in range(self.num_tasks):  # 使用 self.num_tasks 替代硬编码的 7，更通用
            # 直接在这里创建 Sequential 模型，并作为字典的值
            self.task_dense_layers[f'task_{i}_dense_1'] = tf.keras.Sequential([
                tf.keras.layers.Dense(units=hidden_nodes, name=f'task_{i}_dense_1_layer'),
                tf.keras.layers.LeakyReLU(name=f'task_{i}_relu_1'),
                tf.keras.layers.Dropout(rate=self.drop_rate, name=f'task_{i}_dropout_1')  # 使用 self.drop_rate
            ], name=f'task_{i}_dense_block_1')

            self.task_dense_layers[f'task_{i}_dense_2'] = tf.keras.Sequential([
                tf.keras.layers.Dense(units=hidden_nodes // 2, name=f'task_{i}_dense_2_layer'),  # 示例 units
                tf.keras.layers.LeakyReLU(name=f'task_{i}_relu_2'),
                tf.keras.layers.Dropout(rate=self.drop_rate, name=f'task_{i}_dropout_2')  # 使用 self.drop_rate
            ], name=f'task_{i}_dense_block_2')

            self.task_dense_layers[f'task_{i}_output'] = tf.keras.layers.Dense(units=1,
                                                                               name=f'task_{i}_output_layer')  # 保持不变

    def call(self, inputs, training=False, drop_rate=0.0):
        main_branch = self.conv_layer_1(inputs, training=training)
        main_branch = self.conv_layer_2(main_branch, training=training)

        # conv block 1
        conv1_branch = main_branch
        conv1 = self.gap1(conv1_branch)
        ###conv1 = self.flatten(conv1) # GAP 后通常不再需要 Flatten，GlobalMaxPooling1D 已经扁平化了

        main_branch = self.max_pool_1(main_branch)
        main_branch = self.conv_layer_3(main_branch, training=training)
        main_branch = self.conv_layer_4(main_branch, training=training)

        # conv block 2
        conv2_branch = main_branch
        conv2 = self.gap2(conv2_branch)
        ###conv2 = self.flatten(conv2)
        main_branch = self.max_pool_2(main_branch)
        main_branch = self.conv_layer_5(main_branch, training=training)
        main_branch = self.conv_layer_6(main_branch, training=training)

        # conv block 3
        conv3_branch = main_branch
        conv3 = self.gap3(conv3_branch)
        ###conv3 = self.flatten(conv3)

        # Final GAP for main_branch
        final_main_branch_output = self.gap_final(main_branch)

        # Dense layer branches for each task
        tasks_output = []
        for i in range(7):
            task_branch = self.task_dense_layers[f'task_{i}_dense_1'](final_main_branch_output, training=training)
            task_branch = self.task_dense_layers[f'task_{i}_dense_2'](task_branch, training=training)
            task_output = self.task_dense_layers[f'task_{i}_output'](task_branch)
            tasks_output.append(task_output)

        # 返回所有输出
        return (conv1, conv2, conv3, final_main_branch_output, *tasks_output)  # 使用 *tasks_output 解包列表


# 构建特征提取子模型
my_self_supervised_model = SelfSupervisedModel(drop_rate=0.6, num_tasks=7, hidden_nodes=128, stride_mp=4)


class FeatureExtractor(tf.keras.Model):
    def __init__(self, full_model):
        super(FeatureExtractor, self).__init__()
        self.full_model = full_model

    def call(self, inputs, training=False):
        # 使用完整模型获取所有输出
        conv1, conv2, conv3, final_feature, *_ = self.full_model(inputs, training=training)
        return final_feature  # 只返回 final_main_branch_output，作为特征向量


# 实例化特征提取模型
###extract_layer = FeatureExtractor(my_self_supervised_model)


def supervised_model_wesad(x_tr_feature,
                           y_tr,
                           x_te_feature,
                           y_te,
                           identifier,
                           kfold,
                           result,
                           summaries,
                           current_time,
                           epoch_super=200,
                           batch_super=128,
                           lr_super=0.001,
                           hidden_nodes=512,
                           dropout=0.2,
                           L2=0):
    input_dimension = x_tr_feature.shape[1]
    output_dimension = y_tr.shape[1]
    log_dir = os.path.join(summaries, 'ER')
    result = os.path.join(result, 'ER')
    tb = keras.callbacks.TensorBoard(log_dir=log_dir)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(hidden_nodes, input_dim=input_dimension, activation='relu',
                                 kernel_regularizer=keras.regularizers.l2(L2)))
    model.add(keras.layers.Dense(hidden_nodes, activation='relu', kernel_regularizer=keras.regularizers.l2(L2)))

    if output_dimension == 2:
        model.add(keras.layers.Dense(output_dimension, activation='sigmoid'))
        op = keras.optimizers.Adam(lr=lr_super)
        ###model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])
        model.compile(loss='binary_crossentropy',
                      optimizer=op,
                      metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])
    else:
        model.add(keras.layers.Dense(output_dimension))
        model.add(keras.layers.Activation('softmax'))
        op = keras.optimizers.Adam(lr=lr_super)
        model.compile(loss='categorical_crossentropy', optimizer=op, metrics=['accuracy'])

        ###model.fit(x_tr_feature, y_tr, epochs=epoch_super, batch_size=batch_super, callbacks=[tb], verbose=1,validation_data=(x_te_feature, y_te), shuffle=True)
        history = model.fit(
            x_tr_feature, y_tr,
            epochs=epoch_super,
            batch_size=batch_super,
            callbacks=[tb],
            verbose=1,
            validation_data=(x_te_feature, y_te),
            shuffle=True
        )
        # 打印最后一轮 epoch 的指标值（训练和验证）
        # print("\nFinal epoch metrics:")
        # for metric in ['accuracy', 'precision', 'recall', 'auc']:
        # train_value = history.history.get(metric, ['N/A'])[-1]
        # val_value = history.history.get('val_' + metric, ['N/A'])[-1]
        # try:
        # train_value_fmt = f"{float(train_value):.4f}"
        # except (ValueError, TypeError):
        # train_value_fmt = str(train_value)
        # try:
        # val_value_fmt = f"{float(val_value):.4f}"
        # except (ValueError, TypeError):
        # val_value_fmt = str(val_value)

        # print(f"{metric:<10} - train: {train_value:.4f}, val: {val_value:.4f}")
        print("\nFinal epoch metrics:")
        for metric in ['accuracy', 'precision', 'recall', 'auc']:
            train_value = history.history.get(metric, [None])[-1]
            val_value = history.history.get('val_' + metric, [None])[-1]
            if isinstance(train_value, (float, int)):
                train_str = f"{train_value:.4f}"
            else:
                train_str = str(train_value)
            if isinstance(val_value, (float, int)):
                val_str = f"{val_value:.4f}"
            else:
                val_str = str(val_value)
            print(f"{metric:<10} - train: {train_str}, val: {val_str}")

        y_tr_pred = model.predict(x_tr_feature, batch_size=batch_super)
        y_te_pred = model.predict(x_te_feature, batch_size=batch_super)

        y_tr = np.argmax(y_tr, axis=1)
        y_te = np.argmax(y_te, axis=1)

        y_tr_pred = np.argmax(y_tr_pred, axis=1)
        y_te_pred = np.argmax(y_te_pred, axis=1)

        utils.model_result_store(y_tr, y_tr_pred, os.path.join(result, str("tr_" + identifier + ".csv")), kfold)
        utils.model_result_store(y_te, y_te_pred, os.path.join(result, str("te_" + identifier + ".csv")), kfold)

    return


# --- 模型测试和训练循环的示例代码（在 if __name__ == "__main__": 块中运行） ---
if __name__ == "__main__":
    print("--- 开始测试模型和层块 ---")

    # 设置随机种子，保证结果可复现
    np.random.seed(42)
    tf.random.set_seed(42)

    # 定义模型参数
    input_sequence_length = window_size  # 使用全局变量 window_size
    num_input_features = 1  # 假设是单通道时序数据（如 ECG）
    batch_size = 32
    hidden_nodes_ssm = 128  # 用于 SelfSupervisedModel
    stride_mp_ssm = 4
    learning_rate_ssm = 0.001
    num_training_steps_ssm = 100  # 减少训练步数以便快速测试
    dropout_rate_ssm = 0.5  # 训练时使用的 Dropout 比率

    # --- 测试 `conv_block_layer` ---
    print("\n测试 conv_block_layer...")
    test_conv_block = ConvBlock(filters=16, kernel_size=3, pool_size=8, stride_mp=1, name='test_conv_block')
    test_input_conv = tf.random.normal((batch_size, input_sequence_length, num_input_features))
    test_output_conv = test_conv_block(test_input_conv, training=True)  # 传入 training=True 以激活 BatchNorm/Dropout
    print(f"conv_block_layer 输出形状: {test_output_conv.shape}")
    print("conv_block_layer 测试通过。")

    # --- 测试 `dense_block_layer` ---
    print("\n测试 dense_block_layer...")
    test_dense_layer = dense_block_layer(hidden_nodes=64, drop_rate=0.3, name='test_dense_block')
    # Dense 块期望一个 2D 输入 (batch_size, features)
    test_input_dense = tf.random.normal((batch_size, 100))  # 假设 100 个输入特征
    test_output_dense = test_dense_layer(test_input_dense, training=True)  # 传入 training=True 以激活 Dropout
    print(f"dense_block_layer 输出形状: {test_output_dense.shape}")
    print("dense_block_layer 测试通过。")

    # --- 测试 `SelfSupervisedModel` 在训练循环中的行为 ---
    print("\n--- 测试 SelfSupervisedModel (预训练模拟) ---")





