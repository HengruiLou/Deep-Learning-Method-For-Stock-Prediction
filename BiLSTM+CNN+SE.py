#coding=utf-8
import sys
import time

from tensorflow import keras
import numpy as np
import pandas as pd

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation,Reshape
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, multiply, Permute, Concatenate, Add, Lambda
from tensorflow.keras.layers import Flatten, Input,Dense, Dropout, BatchNormalization,LSTM,GRU,Bidirectional,multiply,TimeDistributed,Concatenate
from tensorflow.keras.models import Sequential, load_model,Model
from tensorflow.keras.optimizers import SGD,Adam,RMSprop
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.ticker as mticker
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from math import sqrt
import tensorflow.keras.backend as K
from tensorflow.keras.activations import sigmoid
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
import os
import random




#设置gpu按需分配memory 防止溢出
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)


warnings.filterwarnings("ignore")


os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['PYTHONHASHSEED'] = str(11)
np.random.seed(11)
random.seed(11)
# tf.random.set_seed(12)

timestep = 20
INPUT_SIZE = 6
out_shape=1

class DataSolve(object):
    def __init__(self,name):
        # 读取CSV文件
        def read_csv():
            stock_data = pd.read_csv(name, encoding='gbk')
            stock_data = stock_data[::-1]  # 按日期顺序排列
            stock_data.reset_index(drop=True, inplace=True)
            datas = stock_data[['开盘价', '收盘价', '最高价', '最低价', '成交量', '成交金额']]
            return np.array(datas.values,dtype=np.float32)

        self.datas=read_csv()

    # 划分训练集 测试集
    def split(self):
        train_size=int(len(self.datas)*0.8)
        test_size=len(self.datas)-train_size
        self.train=np.array(self.datas[0:train_size],dtype=np.float32)
        self.test=np.array(self.datas[train_size:len(self.datas)],dtype=np.float32)
        # print(len(self.train))
        # print(self.train)
        # print("----------------------------------------")
        # print(len(self.test))
        # print(self.test)

    # 归一化处理
    def normalize(self):
        # z-score归一化
         self.split()
         scaler=StandardScaler()
         self.train=scaler.fit_transform(self.train)
         self.test=scaler.fit_transform(self.test)
         self.train_x=self.train[:,0:]
         self.train_y=self.train[:,1]

         self.test_x=self.test[:,0:]
         self.test_y=self.test[:,1]
         # print(self.train_x)
         # print("----------------------------------------")
         # print(self.train_y)

    def processing(self):
        self.normalize()
        self.new_train_x,self.new_train_y,self.new_test_x,self.new_test_y=[],[],[],[]
        for i in range(len(self.train_y)-timestep):
            new_x=self.train_x[i:i+timestep]
            new_y=self.train_y[i+timestep]
            self.new_train_x.append(new_x)
            self.new_train_y.append(new_y)

        for i in range(len(self.test_y)-timestep):
            new_x=self.test_x[i:i+timestep]
            new_y=self.test_y[i+timestep]
            self.new_test_x.append(new_x)
            self.new_test_y.append(new_y)

        self.new_train_x=np.vstack(self.new_train_x).reshape(-1,timestep,INPUT_SIZE)
        self.new_train_y=np.vstack(self.new_train_y).reshape(-1,out_shape)
        self.new_test_x=np.vstack(self.new_test_x).reshape(-1,timestep,INPUT_SIZE)
        self.new_test_y=np.vstack(self.new_test_y).reshape(-1,out_shape)

        return self.new_train_x,self.new_train_y,self.new_test_x,self.new_test_y


class Attention(object):

    # # Attention
    def attention_3d_block(self,inputs):
        input_dim = int(inputs.shape[2])
        a = inputs

        a_probs = Dense(input_dim, activation="softmax")(a)

        output_attention_mul = multiply([inputs, a_probs])
        return output_attention_mul

    def attach_attention_module(self, net, attention_module):
        if attention_module == 'se_block':  # SE_block
            net = self.se_block(net)
        elif attention_module == 'cbam_block':  # CBAM_block
            net = self.cbam_block(net)
        else:
            raise Exception("'{}' is not supported attention module!".format(attention_module))

        return net

    def se_block(self,input_feature, ratio=8):
        """Contains the implementation of Squeeze-and-Excitation(SE) block.
        As described in https://arxiv.org/abs/1709.01507.
        """

        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]

        se_feature = GlobalAveragePooling2D()(input_feature)
        se_feature = Reshape((1, 1, channel))(se_feature)
        assert se_feature.shape[1:] == (1, 1, channel)
        se_feature = Dense(channel // ratio,
                           activation='relu',
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')(se_feature)
        assert se_feature.shape[1:] == (1, 1, channel // ratio)
        se_feature = Dense(channel,
                           activation='sigmoid',
                           kernel_initializer='he_normal',
                           use_bias=True,
                           bias_initializer='zeros')(se_feature)
        assert se_feature.shape[1:] == (1, 1, channel)
        if K.image_data_format() == 'channels_first':
            se_feature = Permute((3, 1, 2))(se_feature)

        se_feature = multiply([input_feature, se_feature])
        return se_feature

    def cbam_block(self,cbam_feature, ratio=8):
        """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
        As described in https://arxiv.org/abs/1807.06521.
        """

        cbam_feature = self.channel_attention(cbam_feature, ratio)
        cbam_feature = self.spatial_attention(cbam_feature)
        return cbam_feature

    def channel_attention(self,input_feature, ratio=8):
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]

        shared_layer_one = Dense(channel // ratio,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                 kernel_initializer='he_normal',
                                 use_bias=True,
                                 bias_initializer='zeros')

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel // ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1, 1, channel)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1, 1, channel))(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel // ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1, 1, channel)

        cbam_feature = Add()([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])

    def spatial_attention(self,input_feature):
        kernel_size = 7

        if K.image_data_format() == "channels_first":
            channel = input_feature.shape[1]
            cbam_feature = Permute((2, 3, 1))(input_feature)
        else:
            channel = input_feature.shape[-1]
            cbam_feature = input_feature

        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
        assert avg_pool.shape[-1] == 1
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
        assert max_pool.shape[-1] == 1
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        cbam_feature = Conv2D(filters=1,
                              kernel_size=kernel_size,
                              strides=1,
                              padding='same',
                              activation='sigmoid',
                              kernel_initializer='he_normal',
                              use_bias=False)(concat)
        assert cbam_feature.shape[-1] == 1

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])


# BiLSTM模型
class ConstructModel(object):
    def __init__(self, configure_maps):
        self.__configure_maps = configure_maps

    def model(self):


        inputs=Input(shape=(timestep, INPUT_SIZE))

        BiLstm_out = Bidirectional(LSTM(20, input_shape=(timestep, INPUT_SIZE), return_sequences=True))(inputs)
        BiLstm_out = Bidirectional(LSTM(10,return_sequences=True))(BiLstm_out)

        BiLstm_out=Reshape((BiLstm_out.shape[1],BiLstm_out.shape[-1],1))(BiLstm_out)

        conv2D = Conv2D(64, (3, 3), kernel_regularizer=l1_l2(0.001, 0.001), bias_regularizer=l1_l2(0.001, 0.001))(BiLstm_out)
        conv2D=BatchNormalization()(conv2D)
        conv2D=Activation('relu')(conv2D)

        attent=Attention()
        cbam_block = "se_block"
        attention = attent.attach_attention_module(conv2D, cbam_block)
        attention=BatchNormalization()(attention)
        attention = Dropout(0.2)(attention)

        flatten = Flatten()(attention)

        # dense_1 = Dense(32,kernel_regularizer=l1_l2(0.001, 0.001), bias_regularizer=l1_l2(0.001, 0.001))(
        #     flatten)
        # dense_1 = BatchNormalization()(dense_1)
        # dense_1 = Dropout(0.2)(dense_1)
        #
        # dense_2 = Dense(16, kernel_regularizer=l1_l2(0.001, 0.001), bias_regularizer=l1_l2(0.001, 0.001))(
        #     dense_1)
        # dense_2 = BatchNormalization()(dense_2)
        # dense_2 = Dropout(0.2)(dense_2)

        out_put = Dense(self.__configure_maps["num_of_classes"])(flatten)
        model = Model(inputs=[inputs], outputs=out_put)
        rms = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-06)
        model.compile(loss="mse", optimizer=rms)
        model.summary()
        return model

class TrainModel(object):
    def __init__(self, conf):
        self.__conf = conf

    def train(self, name):
        datas = DataSolve(name)
        train_x, train_y, test_x, test_y = datas.processing()
        constructor = ConstructModel(self.__conf)
        model=constructor.model()
        keras.utils.plot_model(model, "se.png", show_shapes=True)
        model.fit(train_x,
                  train_y,
                  batch_size=self.__conf["1"]["batch_size"],
                  epochs=self.__conf["1"]["epochs"],
                  validation_data=(test_x,test_y),
                  verbose=self.__conf["verbose"],  # 每个epoch输出一行记录
                  # callbacks=[ModelCheckpoint("./model/" + name + ".h5",
                  #                            monitor='val_accuracy', verbose=1, mode='max',
                  #                            save_best_only=True, save_weights_only=False)]
                  )
        y_true,y_pred =test_y,model.predict(test_x)
        # print(model.evaluate(y_true,y_pred))
        return y_true,y_pred

class Main(object):

    std_file_name = "./logs/" + time.strftime("%Y_%b_%d_%H_%M_%S", time.gmtime()) + "_timestep" + str(
        timestep) + "_BiLSTM_CNN_SE_Stock.txt"
    sys.stdout = open(std_file_name, "w+")
    # 参数
    conf = {"num_of_classes": 1, "nesterov": True, "verbose": 2,
            "1": {
                "batch_size": 8, "epochs": 40, "learning_rate": 1e-3,
                "momentum": 0.5, "decay": 5 * 1e-6, "alpha": 1, "beta": 1,
                "input_shape": (timestep, INPUT_SIZE),  # 修改
            }}

    file_name= './data/000617.csv'
    trainer = TrainModel(conf)
    y_true,y_pred=trainer.train(file_name)
    # print(y_true)

    # 下面画出收盘价走势
    fig, ax = plt.subplots(figsize=(16, 6))
    # ax.plot(testY2,color='blue')
    # ax.plot(testPredict2_2,color='orange')

    # Visualising the results 使用Matplotlib将预测股价和实际股价的结果可视化。
    plt.plot(y_true, color='blue', label='REAL Stock Price')
    plt.plot(y_pred, color='red', label='PREDICT Stock Price')
    plt.title('TEST CLOSE PRICE')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f day'))
    plt.legend()
    plt.show()

    # 回归评价指标

    # calculate MSE 均方误差
    mse = mean_squared_error(y_true,y_pred)
    # calculate RMSE 均方根误差
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    # calculate MAE 平均绝对误差
    mae = mean_absolute_error(y_true, y_pred)
    # calculate R square
    r_square = r2_score(y_true, y_pred)
    print('均方误差mse: %.6f' % mse)
    print('均方根误差rmse: %.6f' % rmse)
    print('平均绝对误差mae: %.6f' % mae)
    print('R_square: %.6f' % r_square)


if __name__ == '__main__':
    Main()

