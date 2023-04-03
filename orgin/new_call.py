
import numpy as np
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import sympy

class my_callback(Callback):
    def __init__(self,x_all,y_all,n,showTestDetail=True):
        
        self.showTestDetail=showTestDetail
        self.predhis = []
        self.targets = []
    
        self.n = n
        self.loss = []
        self.y_all = y_all
        self.x_all = x_all
        
        
    def mape(self,y,predict):
            diff = np.abs(np.array(y) - np.array(predict))
            return np.mean(diff /y)
    def plot_loss(self):
        plt.figure(figsize=(8, 4))
        loss = np.array(self.loss)
        #x = np.arange(len(loss))
        #plt.scatter(x,loss,c = 'red',label = 'loss')
        
    def on_epoch_end(self, epoch, logs={}):
        x= sympy.symbols('x')
       # y1 = np.power(10,(self.y_all*(self.y_max-self.y_min)+self.y_min))
        #y2 = np.power(10,(self.model.predict(self.x_all)*(self.y_max-self.y_min)+self.y_min))
      #  y_test = sympy.solve(x*np.power(10,x)-y1,x)
        y_test = np.power(10,-self.y_all)
        prediction = self.model.predict(self.x_all)
        prediction = np.power(10,-prediction)

        self.predhis.append(prediction)
       
        ytrue = []
        ypred = []
        
        
        if self.showTestDetail:
            for index,item in enumerate(prediction):
              
                ytrue.append(y_test[index])
                ypred.append(prediction[index])
                print("predict",item,"=","true",y_test[index],"v=",y_test[index]-item)
                testLoss =np.abs((y_test[index]-item))/y_test[index]
                self.loss.append(testLoss)
            testloss = np.array(self.loss)
            testloss = np.mean(testloss)
            print("test loss is :{}".format(testLoss))
            x = np.arange(0,int(len(prediction)),1)
            
            
        if((logs.get('loss') < 0.05)&(testLoss<0.03)):
            print("sat \n 已经到达 99.5% 的训练精度！")
            self.model.stop_training = True
            self.restore_best_weights = True
        

      
