import numpy as np
import linecache
import csv
import pandas as pd
import os
import glob
import kop_calculator

       
def init_get_name(filepath,filename):
    name = []
    filen = filepath+filename
    with open(filen,'r') as file:
        line = file.readline().split(',')
        for i in line:
            name.append(i)
        
    return name,len(name)

def init_get_index(filepath,filename,num):
    
    
    txt = np.loadtxt(filepath+filename, delimiter=',', skiprows=1)[:, num-1].reshape(-1)
    txt = np.array(txt,dtype = int)
    txt = txt.tolist()
    for i in range(len(txt)):
        txt[i] = str(txt[i])
    
    return txt

def init_get_data(filepath,dataname,config_path):
    filepath = filepath+dataname+'.txt'
    Vg = []
    Vd = []
    Id = []
    
    with open(filepath,'r') as file:
        for line in file.readlines()[1:] :
            new_lines = line.strip().split(',')
            
            Vg.append(new_lines[0])
            Vd.append(new_lines[1])
            Id.append(new_lines[2])
    #去除Vg，Vd=0时Ids<0

    Id = np.array(Id,dtype = float)
    
    Vd = np.array(Vd,dtype = float)
    Vg = np.array(Vg,dtype = float)
   
    
    for i in range(0,len(Id)):
       if Id[i] < 0.0:
           Id[i] = np.absolute(Id[i])
     
    Id = np.absolute(np.log10(Id))
 
    
    return Vg,Vd,Id,len(Id)



def init_get_par(filepath,filename,n1,num,n):
    filen = filepath+filename
    para = []
    line = linecache.getline(filen,n1).split(',')
    for i in line:
        if (i)!=' ': 
           para.append(i)
    #print("finish the ",n,"paramaters recieve!")
    para = para[:num-n]
    
    para = np.array(para,dtype = float)
    
    return para

def init_write_par(filepath,writename,para_name,para,num,namenum,n):
    
    file = filepath+writename+'.csv'

    with open(file,"w",newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(para_name[:namenum-n])
    
        for i in range(0,num):
            writer.writerow(para)
    #print("write para finish!")

def init_write_vi(filepath,writename,Vg,Vd,Id):
    
    csvfile = pd.read_csv(filepath+writename+'.csv')
    csvfile['Vg'] = Vg
    csvfile['Vd'] = Vd
    csvfile['Id'] = Id
   
    csvfile.to_csv( filepath+writename+'.csv',mode = 'w',index=False,line_terminator="")
    #print("write vg-vd-id finish!")
    
def init_conform(writepath,con_writepath,train_name):
    csv_list = glob.glob(writepath+'*.csv')
    file = csv_list[0]
    df = pd.read_csv(file)
    df = df.to_csv(con_writepath+train_name,index=False,line_terminator="")
    
    for i in range(1,len(csv_list)):
        file = csv_list[i]
        df = pd.read_csv(file)
        df = df.to_csv(con_writepath+train_name,index=False,header = False,mode = 'a',line_terminator="")

    
def init_read(filepath,filename,writepath,con_writepath,train_name,config_path):
    name,par_num = init_get_name(filepath,filename)
    print(par_num)
    index = init_get_index(filepath,filename,par_num)
    num_index = len(index)
    
    folder1 = os.path.exists(writepath)
    if not folder1:
        os.makedirs(writepath)
        
    for i in range(num_index): 
       
        file = writepath+str(index[i])+'.csv'
        if os.path.exists(file):
            os.remove(file)
    n = 1 
    for i in range(num_index): 
        Vg,Vd,Id,num = init_get_data(filepath,index[i],config_path)
        
        para = init_get_par(filepath,filename,i+2,par_num,n)
        init_write_par(writepath,index[i],name,para,num,par_num,n)
        init_write_vi(writepath,index[i],Vg,Vd,Id)
    folder2 = os.path.exists(con_writepath)
    if not folder2:
         os.makedirs(con_writepath)
    init_conform(writepath,con_writepath,train_name)
   
    print("Prepare for reading  data!")
  
    



        


