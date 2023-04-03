import glob
import pandas as pd
import numpy as np
def change(path,writepath):
    init_lg = 10.8
    init_tsh = 50
    init_tox = 5
    init_doping = 1e15
    with open(writepath+'parametertrain.txt','w') as f:
        f.write('lg_nw,Tsh,t_ox,Gauss_doping,index\n')
        
    csv_list = glob.glob(path+'*.csv')
    para = []
    for i in range(len(csv_list)):
        l = csv_list[i].split('GAA')
        para.append( l[1].split('_')[0])
        
    parameter = np.zeros([len(para),5])    

    for i in range(len(para)):
      
        if 'e' in para[i]:
            #print('doping')
            parameter[i][0] = init_lg
            parameter[i][1] = init_tsh
            parameter[i][2] = init_tox
            parameter[i][3] = '{:.2f}'.format(float(para[i]))
            parameter[i][4] = int(i)             
        else :
            #print('1')
         
            parameter[i][0] = '{:.2f}'.format(init_lg * float(para[i]))
            parameter[i][1] = '{:.2f}'.format(init_tsh * float(para[i]))
            parameter[i][2] = '{:.2f}'.format(init_tox *  float(para[i]))
            parameter[i][3] = '{:.3f}'.format(float(init_doping))
            parameter[i][4] = int(i) 

    df = pd.DataFrame(parameter,dtype = float)
  
    df.to_csv( writepath+'parametertrain.txt',mode = 'a+',index=False,line_terminator="",header = False)       
    for i in range(len(csv_list)):
        file = csv_list[i]
        df = pd.read_csv(file,skiprows= 2)

        with open(writepath+str(i)+'.txt','w') as f:
            f.write('vg,vd,ids\n')
        df.to_csv( writepath+str(i)+'.txt',mode = 'a+',index=False,line_terminator="")
 
if __name__ == '__main__':
    path = '../img/data/'
    writepath = '../img/GAA_data/'
    change(path,writepath)