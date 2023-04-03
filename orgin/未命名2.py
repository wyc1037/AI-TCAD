import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

import os
from skimage import io

def extract_data(img_name,min_x, max_x,min_y ,max_y,vd ):
    
    # 打开图片
    img = cv.imread(img_name)
    # 灰度化
    gary_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 二值化
    ret, binary_img = cv.threshold(gary_img, 240, 255, cv.THRESH_BINARY)
    #cv.imshow('binary_img', binary_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # 边缘提取
    xgrd = cv.Sobel(binary_img,cv.CV_16SC1,1,0)
    ygrd = cv.Sobel(binary_img,cv.CV_16SC1,0,1)
    
    egde_output = cv.Canny(xgrd,ygrd,50,150)
    #cv.imshow('canny_edge',egde_output)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # 图像像素按行和列求和
    column_sum_img = np.sum(egde_output, axis=0)
    row_sum_img = np.sum(egde_output, axis=1)
    # 排序
    #sort_column_sum = np.sort(column_sum_img)
    sort_column_sum_indices = np.argsort(column_sum_img)
    #sort_row_sum = np.sort(row_sum_img)
    sort_row_sum_indices = np.argsort(row_sum_img)
    
    row_n = len(sort_row_sum_indices)
    column_n = len(sort_column_sum_indices)
    

    min_row,max_row,min_column,max_column =0,0,0,0
    if sort_row_sum_indices[row_n-1]-sort_row_sum_indices[row_n-2]>=100:
        n = 1
        min_row = sort_row_sum_indices[row_n-1]
        max_row = sort_row_sum_indices[row_n-n]
    if sort_row_sum_indices[row_n-1]-sort_row_sum_indices[row_n-2]<100:
        n = 2
        min_row = sort_row_sum_indices[row_n-n]
        max_row = sort_row_sum_indices[row_n-n-1]
    if sort_column_sum_indices[column_n-1]-sort_column_sum_indices[column_n-2]<100:
        n = 2
        min_column = sort_column_sum_indices[ column_n-n]+15
        max_column = sort_column_sum_indices[column_n-n-1]
    if sort_column_sum_indices[column_n-1]-sort_column_sum_indices[column_n-2]>=100:
        n = 1
        min_column = sort_column_sum_indices[ column_n-n-1]+15
        max_column = sort_column_sum_indices[column_n-n]
    
   
    fc = egde_output[ min_row:max_row,min_column:max_column]
   # cv.imshow('function_curve', fc)
    cv.waitKey(0)
    cv.destroyAllWindows()
      
    n_row = max_row - min_row
    n_column = max_column - min_column
    x_axis = np.empty([n_row, n_column])
    y_axis = np.empty([n_column, n_row])
    
    x_interval = (max_x-min_x)/(n_row)
    x_value = np.arange(min_x, max_x, x_interval)
    x_value = x_value.reshape(n_row,1)
    print(x_value.shape)
    y_interval = (max_y-min_y)/(n_column)
    y_value = np.arange(max_y, min_y,-y_interval)
    y_value = y_value.reshape(n_column,1)
    
    x_axis[:,] = x_value
    y_axis[:,] = y_value
    y_axis = y_axis.T
    
    print(x_axis.shape,y_axis.shape)
    
    x_fc = x_axis.T[fc.T==255]
    y_fc = y_axis.T[fc.T==255]
  #去除重复采点  
    for i in range(len(y_fc)):
        for j in range(i+1,len(y_fc)):
            if ((x_fc[i] - x_fc[j])<1e-10)& ( x_fc[i] != 0)&( x_fc[j]!=0):

               y_fc[j] = 0
               
    index = []
    for i in range(len(y_fc)):
        if y_fc[i] <1e-7:
            index.append(i)
        if x_fc[i] <1e-5:
            index.append(i)
    
    vg = np.delete(x_fc, index)
    ids = np.delete(y_fc, index)
    
    
    plt.scatter(vg,ids)
    num = len(vg)
    data = np.empty([3, num])
    data[0] = vg
    data[1] = vd
    data[2] = ids
    return data

def write(data,path):
    
    with open(path,'w') as f:
        f.write('vg   ,vd    ,ids  \n')
    
    with open(path,'a+') as f:
        np.savetxt(f ,np.column_stack((data[0],data[1],data[2])), delimiter=',',fmt='%.5f,%.3f,%.8f')
 

   
if __name__ == '__main__':

    path = '../img/'
    writepath = path+'data/'
    folder1 = os.path.exists(writepath)
    if not folder1:
       os.makedirs(writepath)
    min_x = 0
    max_x = 0.6
    min_y = 0
    max_y =5.4e-5
    vd = 0.5
    img_path = sorted([os.path.join(path, name) for name in os.listdir(path) if name.endswith('.png')])###这里的'.tif'可以换成任意的文件后缀
    for i in range(len(img_path)):
        img_name = os.path.split(img_path[i])[-1]
        img_name = img_name.split('.')[0]
       # img = io.imread(img_path[i])##img_path[i]就是完整的单个指定文件路径了
        img_path = path+img_name+'.png'
        print(img_path)

        data = extract_data(img_path,min_x, max_x,min_y ,max_y,vd )
        print(data)
   
        file = writepath+str(i)+'.txt'
        if os.path.exists(file):
            os.remove(file)
        write(data,file)
