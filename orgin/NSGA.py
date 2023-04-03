#可以用来计算算法运行时间
import datetime
#用来绘图的神奇库
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
#数据运算，操作库
import numpy as np
class Ga_method():
    def __init__(self,DNA_Size=32,Pop_size=200,Crossover_rate=0.8,mutation_rate=0.005, N_genrations=50):
       
       pass
#基因解码　　二进制->十进制
    def translateDNA(self,pop): 
        
      pass
#得到适应度
    def get_fitness(self,pop):#适应度函数
    pass
#交叉和变异
    def crossover_and_mutation(self,pop):
    pass
#变异
    def mutate(self,child): 
    pass
#选择
    def select(self,pop,fitness):
    pass
#打印算法的相关信息
    def print_info(self,pop):
    pass
#画出动态图   
    def plot_3d(self,ax):
    pass
    def plot_2d(self,ax):
    pass
#训练　进化
    def train(self):
    pass
 