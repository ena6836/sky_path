# -*- coding: utf-8 -*-
import re
import numpy as np
#import gdal
from os.path import join,relpath
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import math
import networkx as nx
import csv
from PIL import Image
import shelve

#x, yの並び順はすべて(x, y)

def main():
    tra_seg_fin_x = np.empty(3)
    tra_seg_fin_y = np.empty(3)

    seg_trajectory = np.array([[0, 0], [0, 1]])
    tra_seg_fin_x[0] = seg_trajectory[len(seg_trajectory) - 1][0]
    tra_seg_fin_y[0] = seg_trajectory[len(seg_trajectory) - 1][1]

    seg_trajectory = np.array([[1, 1], [2, 2]])
    tra_seg_fin_x[1] = seg_trajectory[len(seg_trajectory) - 1][0]
    tra_seg_fin_y[1] = seg_trajectory[len(seg_trajectory) - 1][1]

    seg_trajectory = np.array([[3, 2], [3, 3]])
    tra_seg_fin_x[2] = seg_trajectory[len(seg_trajectory) - 1][0]
    tra_seg_fin_y[2] = seg_trajectory[len(seg_trajectory) - 1][1]

    all_trajectory = np.array([[0, 0], [0, 1], [1, 1], [2, 2], [3, 2], [3, 3]])
    len_trajectory = len(all_trajectory)    
    x_tra = np.empty(len_trajectory)
    y_tra = np.empty(len_trajectory)
    for i in range(len_trajectory):
        x_tra[i] = all_trajectory[i][0]
        y_tra[i] = all_trajectory[i][1]

    path_nodes = np.array([[0, 0], [1, 0], [2, 1], [3, 1], [3, 2], [3, 3]])
    len_path = len(path_nodes)    
    x_path = np.empty(len_path)
    y_path = np.empty(len_path)
    for i in range(len_path):
        x_path[i] = path_nodes[i][0]
        y_path[i] = path_nodes[i][1]

    path_seg_fin_x = np.empty(3)
    path_seg_fin_y = np.empty(3)

    path_seg_fin = np.array([[1, 0], [3, 1], [3, 3]])
    path_seg_fin_x[0] = path_seg_fin[0][0]
    path_seg_fin_y[0] = path_seg_fin[0][1]
    path_seg_fin_x[1] = path_seg_fin[1][0]
    path_seg_fin_y[1] = path_seg_fin[1][1]
    path_seg_fin_x[2] = path_seg_fin[2][0]
    path_seg_fin_y[2] = path_seg_fin[2][1]

    #x = np.array([0, 0, 1, 2, 3, 3])
    #y = np.array([0, 1, 1, 2, 3, 3])
    #all_trajectory = np.array([[0, 0], [0, 1], [1, 1], [2, 2], [3, 2], [3, 3]])
    #all_trajectory = np.array([[0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 3, 3]])
    #path_nodes = np.array([[0, 0], [1, 0], [2, 1], [3, 1], [3, 2], [3, 3]])
    #path_nodes = np.array([[0, 1, 2, 3, 3, 3], [0, 0, 1, 1, 2, 3]])

    plt.figure(figsize=(5,5))
    plt.plot(x_tra, y_tra)
    plt.plot(x_tra, y_tra, "r")
    plt.plot(x_path, y_path, "b")
    plt.plot(tra_seg_fin_x, tra_seg_fin_y, "ro")
    plt.plot(path_seg_fin_x, path_seg_fin_y, "bo")
    plt.show()

if __name__ == "__main__":
    main()