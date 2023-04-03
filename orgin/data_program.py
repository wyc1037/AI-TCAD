import new_read
import os
import sys

class data_pro:
    def __init__(self,data,config_path):
        
        self.get_data(data,config_path)
    def get_data(self,data,config_path):
        
        
        (filepath, tempfilename) = os.path.split(data)
        filepath = filepath+'/'
        
      
        if 'parametertrain' in tempfilename:
            writetrainpath = './in_data/'+'_train_data/'
            con_writepath = './in_data/'+'_data/'
            train_name = 'train.csv'
            if os.path.exists(con_writepath+train_name):
                 print("already have it!")
            if (os.path.exists(con_writepath+train_name)==False):
                 new_read.init_read(filepath,tempfilename,writetrainpath,con_writepath,train_name,config_path)
          
        
        name,par_num = new_read.init_get_name(filepath,tempfilename)
          
        n = par_num-1+2
        self.n = n 
          
    def get_n(self):
        return self.n
    
  
    
if __name__ == '__main__':
    config_path = '../img/GAA_data/config.txt'
    data = '../img/GAA_data/parametertrain.txt'
    #data = sys.argv[1]
    data_pro = data_pro(data,config_path)
    #print(data_pro.get_index())
    