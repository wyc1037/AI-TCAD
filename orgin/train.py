import sys
import re
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow import keras 
import new_call
import matplotlib.pyplot as plt
import timeit
import pred
import data_program
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train(x_train,y_train,x_test,y_test,n,batch_size,epochs,layers,model_path):
    
    #model参数设置
    
    n_stocks = n #输入
    n_neurons_1 = 64 #第一层隐藏层神经元个数
    n_neurons_2 = 64 #第二层隐藏层神经元个数
    n_neurons_3 = 16 #第三层隐藏层神经元个数
    n_target = 1 #输出层神经元个数
   # 预输入X，预输出值Y
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Dense(n_stocks,
                                activation="relu",
                                input_shape=(x_train.shape[1], )
                                ,bias_initializer = 'he_normal'
                                ,kernel_initializer = 'he_normal'
                                ))
    model.add(tf.keras.layers.Dense(n_neurons_1 ,
                                activation="relu"
                                ,bias_initializer = 'he_normal'
                                ,kernel_initializer = 'he_normal'
                                ))
    model.add(tf.keras.layers.Dense(n_neurons_2 ,
                                activation="relu"
                                ,bias_initializer = 'he_normal'
                                ,kernel_initializer = 'he_normal'
                                ))
    model.add(tf.keras.layers.Dense(n_neurons_3 ,
                                activation="relu"
                                ,bias_initializer = 'he_normal'
                                ,kernel_initializer = 'he_normal'
                                ))

    model.add(tf.keras.layers.Dense(n_target))  # 最后的密集连接层，不用激活函数
    model.compile(optimizer="adam",  # 优化器
              loss="mse",  # 损失函数
              metrics=["mse"]  # 评估指标：平均绝对误差
             )
    my = new_call.my_callback(x_test,y_test,n)
    history = model.fit(x_train,  # 特征
          y_train,  # 输出
          epochs = epochs,  # 模型训练100轮
          validation_split=0.2,
          batch_size=batch_size,
          callbacks = [my],
          verbose=1  # 静默模式；如果=1表示日志模式，输出每轮训练的结果
         )
    history_dict = history.history 
    loss_values = history_dict["loss"]
    epochs = range(1,len(loss_values) + 1)
# 训练
    plt.plot(epochs,  # 循环轮数
         loss_values,  # loss取值
         "r",  # 红色
         label="loss"  
        )
    plt.title("Loss and Mse of Training")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    
    model.save(model_path)


  
if __name__ == '__main__':
    start = timeit.default_timer()
    '''
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    
    for arg in sys.argv:
        if '=' not in arg:
            continue
        val = re.split('=', arg)[1]
        if 'data' in arg:
            data = val.split('\'')[0]
          
        if 'model' in arg:
            model = val.split('\'')[0]
     '''
    data =  '../img/GAA_data/parametertrain.txt'
    filepath = '../img/GAA_data/'
    model = '../model.h5'
   # (filepath, tempfilename) = os.path.split(data)
    config_path = filepath+'/config.txt'

    data_pro = data_program.data_pro(data,config_path)
    n = data_pro.get_n()
    
    data_path = './in_data/_data/train.csv'
    batch_size = 1
    epochs = 100
    layers = 4
    
    data_train = pd.read_csv(data_path,index_col=None).astype('float32')
     
    sample_tatal_num = data_train.shape[0]
  
    train_start = 0
    train_end = int(np.floor(sample_tatal_num)) #np.floor向下取整
   
    test_start = 0
    test_end =  500
   
   #分割训练集和测试集
    data_train = data_train.iloc[np.arange(train_start, train_end), :]
    data_test = data_train.iloc[np.arange(test_start, test_end), :] 
 
    x_train = data_train.iloc[:,:n]
    y_train = data_train.iloc[:,n]
    x_test = data_test.iloc[:,:n]
    y_test = data_test.iloc[:, n]
    # 数据标准化
   # y_train = np.absolute(np.log10(y_train))
   # y_test= np.absolute(np.log10(y_test))
    
    x_min = [];y_min = [];x_max = [];y_max = []
    x_range = [];y_range = []
    x_min = x_train.min(axis=0);x_max = x_train.max(axis=0)
    x_range = x_max - x_min
    np.array(x_min)
    np.array(x_max)
    np.array(x_range)
    x_train = (x_train-x_min)/(x_range)
    
    xtest_min = [];ytest_min = [];xtest_max = [];ytest_max = []
    xtest_range = [];ytest_range = []
    xtest_min = x_test.min(axis=0);xtest_max = x_test.max(axis=0)
    xtest_range = xtest_max - xtest_min
    np.array(xtest_min)
    np.array(xtest_max)
    np.array(xtest_range)
    x_test = (x_test-xtest_min)/(xtest_range)
 
    train(x_train,y_train,x_test,y_test,n,batch_size,epochs,layers,model)#batch,epoch,layers,denses
    
    
    y_pred = pred.pred(data_test,x_test,model,n)
    
    y_true = np.power(10,-y_test)
    plt.scatter(data_test.iloc[30:50,1],y_true[30:50])
    plt.scatter(data_test.iloc[30:50,1],y_pred[30:50])
    plt.show()
    end = timeit.default_timer()
    print ("training running time:",str(end-start),"s") 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    