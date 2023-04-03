import sys
import re
import os
import pandas as pd
import timeit


import tensorflow as tf

import numpy as np


#import write_pred

import linecache
import csv

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
def pred(dataout,x,model,n):
    
   
    
    pred_model = tf.keras.models.load_model(model)
    pred_y = pred_model.predict(x)
    pred_y = np.power(10,-pred_y)
    data = np.array(dataout)
    
    for i in range(len(pred_y)):   
        if data[i,n-1]<0.0:
            pred_y[i] = -pred_y[i]
            
    return pred_y
   
  
def get_pred(data,model,output):
    
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    start = timeit.default_timer()
    '''
    for arg in sys.argv:
        if '=' not in arg:
            continue
        val = re.split('=', arg)[1]
        if 'data' in arg:
            data = val.split('\'')[0]
          
        if 'model' in arg:
            model = val.split('\'')[0]
        if 'output' in arg:
            output = val.split('\'')[0]
    
    
    (filepath, tempfilename) = os.path.split(data)
    filename = tempfilename.split('.txt')[0]
    outpath = filepath + '/'
    config_path= outpath + 'config.txt'
'''
    
    xmin = []
    xmax = []
    linepara = linecache.getline(config_path,4).split(',')
    
    xmin.append(linepara[0].split(' ')[1])
    for i in linepara[1:]:
        xmin.append(i)
    
    linepara = linecache.getline(config_path,5).split(',')
    xmax.append(linepara[0].split(' ')[1])
    for i in linepara[1:]:
        xmax.append(i)
   
    n = len(xmin)+2
    
        
    vg = np.loadtxt(data, delimiter=',', skiprows=3)[:, 0].reshape(-1)
    vd = np.loadtxt(data, delimiter=',', skiprows=3)[:, 1].reshape(-1)

    para = []
    line = linecache.getline(data,2).split(',')
    
    for j in line:
        para.append(j)
    para = np.array(para,dtype = float)
    num_para = len(para)
    name = []
    for k in range(num_para):
        name.append(k)
   
    
    predpath = './in_data/'+'_pred_data/'
    dir1 = os.path.abspath(predpath)
    if not os.path.exists(dir1):
        os.makedirs(predpath)
    with open(predpath+filename+'.csv',"w",newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(name)
        for m in range(0,len(vg)):
            writer.writerow(para)
    
    data_out =  pd.read_csv(predpath+filename+'.csv')
    data_out[3] = vg
    data_out[4] = vd
    data_out.to_csv(predpath+filename+'.csv' ,mode = 'w',index=False,line_terminator="")
           
    data_out = pd.read_csv(predpath+filename+'.csv',index_col=None).astype('float32')
    x = data_out.iloc[:,:n]
    x = np.absolute(x)
    x_min = [];x_max = [];
    x_range = [];
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_min = np.array(x_min)
    x_max = np.array(x_max)
    x_range = np.array(x_range)
          
    if 'upper' in data:
       
        x_range[0:n-2] = 1
    if 'lower' in data:
        x_min[0:n-2] = 0
        
        x_range[0:n-2] = 1
    else:
        x_min[0:n-2] = xmin
        x_max[0:n-2] = xmax
    
    x_range = x_max - x_min
    x= (x-x_min)/(x_range)
    
    pred_y = pred(data_out,x,model,n)
       
    line = linecache.getline(data,1)
    line1= linecache.getline(data,2)
    line2= linecache.getline(data,3)
    
    with open(outpath+filename+'r.txt','w') as f:
        f.write(line)
        f.write(line1)
        f.write(line2)
        f.close()
    with open(outpath+filename+'r.txt','a+') as f:
                
        np.savetxt(f ,np.column_stack((vg,vd,pred_y)), delimiter=',')

    end = timeit.default_timer()
    print ("prgram running time:",str(end-start),"s") 
    
   
   
    























