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
    global grid_size
    grid_size = 5.05
    global max_len
    max_len = math.sqrt(2) * grid_size
    global max_ang
    max_ang = 15

    dem_shelve = 'dem_s_rough'
    with shelve.open(dem_shelve) as db:
        dem_rough_sl = db['dem']

    shelve_x_start = 2750
    shelve_y_start = 6000
    shelve_x_end = 3750
    shelve_y_end = 7000

    path_edge_shelve = 'path_edge'
    with shelve.open(path_edge_shelve) as db:
        path_edges = db['edge']
    len_edges = len(path_edges)
    path_nodes = np.empty((len_edges + 1, 2))
    i = 0
    for ed in path_edges:
        path_nodes[i] = np.array(ed[0])
        i += 1
    path_nodes[i] = np.array(path_edges[i - 1][1])
    #np.arrayの[[x0, y0], [x1, y1], .....]という形を想定

    seg_num = 10

    seg_path = split_path(path_nodes, seg_num)

    seg_fin_nodes = find_fin_node(seg_path)

    next_initial_pos = np.array(path_nodes[0])
    initial_pos = np.empty((seg_num, 2))
    target_pos = np.empty((seg_num, 2))
    final_pos = np.empty((seg_num, 2))
    estimate_pos = np.empty((seg_num, 2))
    next_initial_pos = np.empty((seg_num, 2))

    for i in range(seg_num):
        initial_pos[i] = next_initial_pos
        #実際の初期位置
        target_pos[i]= seg_fin_nodes[i]
        #実際の目標位置
        final_pos[i] = rover_run(initial_pos, seg_path[i], i)
        #実際に到着した位置
        final_node_in_dem_all = find_closest_node_in_dem_all(final_pos[i], shelve_x_start, shelve_y_start)
        #実際に到着した位置に最も近いノード
        real_skyline = make_real_skyline(final_node_in_dem_all)
        #実際に到着した位置に最も近いノードから見える実際のスカイライン
        estimate_pos[i] = position_estimate(target_pos, real_skyline, dem_rough_sl)
        #スカイラインから推定される位置
        next_initial_pos[i] = modify_error(estimate_pos, target_pos, final_pos, i)
        #位置推定をもとに修正後の実際の位置

    draw_trajectory(seg_num, path_nodes, seg_fin_nodes)



def split_path(path_nodes, seg_num):
    num_node = len(path_nodes)
    seg_len = num_node // seg_num

    seg_path = np.empty(seg_num)

    for i in range(seg_num):
        if i != seg_num - 1:
            seg_path[i] = path_nodes[seg_len * i : seg_len * (i + 1)]
        else:
            seg_path[i] = path_nodes[seg_len * i:]

    return seg_path


def find_fin_node(seg_path):
    seg_num = len(seg_path)
    fin_nodes = np.empty((seg_num, 2))
    for i in range(seg_num):
        fin_nodes[i] = seg_path[i][len(seg_path[i]) - 1]

    return fin_nodes


def rover_run(initial_pos, nodes, seg_i):
    len_nodes = len(nodes)
    current_pos = initial_pos
    rover_trajectory = np.empty((len_nodes, 2))
    rover_trajectory[0] = current_pos

    sigma_len = 0.1 ##
    sigma_ang = 1   ##

    for i in range(len_nodes - 1):
        vector_move = nodes[i + 1] - nodes[i]
        len_error = 1 + np.random.normal(0, sigma_len)
        ang_error = math.radians(np.random.normal(0, sigma_ang))
        x_move = math.sqrt(len_error) * (vector_move[0] * math.cos(ang_error) - vector_move * math.sin(ang_error))
        y_move = math.sqrt(len_error) * (vector_move[1] * math.sin(ang_error) + vector_move * math.cos(ang_error))
        current_pos += np.array([x_move, y_move])
        rover_trajectory[i + 1] = current_pos

    shelvename = 'trajectory_' + str(seg_i)
    with shelve.open(shelvename) as db:
        db['trajectroy'] = rover_trajectory

    return current_pos

def find_closest_node_in_dem_all(pos, shelve_x_start, shelve_y_start):
    x = 0
    while x < pos[0]:
        x += 1
    if x - pos[0] > 1 / 2:
        x -= 1
    y = 0
    while y < pos[1]:
        y += 1
    if y - pos[1] > 1 / 2:
        y -= 1

    x += shelve_x_start
    y += shelve_y_start

    return np.array([x, y])


def make_real_skyline(node):
    sigma_azimuth = 1       ##
    #方位角の推定誤差
    sigma_photo = 0.001     ##
    #撮影時に生じるノイズ
    sigma_pixel = math.radians(0.5)     ##
    #解像度による誤差
    #先行研究では、skyline resolution = 0.5 deg としている

    x_center = node[0]
    y_center = node[1]

    real_skyline = skyline_with_error_azi(sigma_azimuth, x_center, y_center)
    real_skyline = add_noise(real_skyline, sigma_photo)
    real_skyline = add_noise(real_skyline, sigma_pixel)

    return real_skyline

def skyline_with_error_azi(sigma, x_center, y_center):
    dem = np.array(Image.open('\\Users\\Maenaka\\python\\20190204\\mars_spirit_landing_site\\dem.tif'))
    #dem = np.array(Image.open('/home/shuto/python/20190131/mars_spirit_landing_site/dem.tif'))

    delta_p = 1     #各方位角方向のインクリメント[m]
    max_k = 2000      #インクリメント回数
    radius_star = 3397200     #星の半径（ここでは火星）
    height_cam = 0.7    #カメラの高さ
    grid_size = 1.01
    k = 1
    azi = 0
    elevation_max = 0
    skyline = np.zeros(360)

    error = np.random.normal(0, sigma)

    while azi < 360:
        while k <= max_k:
            x_on_azi_line =  round(x_center + k * delta_p * math.cos(math.radians(azi + error)) / grid_size)
            y_on_azi_line =  round(y_center + k * delta_p * math.sin(math.radians(azi + error)) / grid_size)
            curv = radius_star - math.sqrt(radius_star ** 2 - (k * delta_p) ** 2)
            height_interest = dem[y_on_azi_line][x_on_azi_line] - curv
            height_center = dem[y_center][x_center] + height_cam
            elevation_interest = math.atan((height_interest - height_center) / k * delta_p)
            if elevation_max < elevation_interest:
                elevation_max = elevation_interest
            k += 1
        skyline[azi] = elevation_max
        elevation_max = 0
        azi += 1
        k = 0

    return skyline

def add_noise(skyline, sigma):
    real_skyline = np.zeros(360)
    azi = 0
    while azi < 360:
        noise = np.random.normal(0, sigma)
        real_skyline[azi] = skyline[azi] + noise
        azi += 1

    return real_skyline


def position_estimate(target_pos, real_skyline, dem):
    search_range = 10   ##
    #探索する正方形領域の一辺の長さ（エッジの数）
    min_x = target_pos[0] - search_range // 2
    min_y = target_pos[1] - search_range // 2
    max_x = target_pos[0] + search_range // 2
    max_y = target_pos[1] + search_range // 2

    min_diff_skyline = 99999999

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            rendered_skyline = dem[y][x][1:]
            diff_skyline = 0
            for azi in range(360):
                diff_skyline += (rendered_skyline[azi] - real_skyline[azi]) ** 2
            if min_diff_skyline > diff_skyline:
                min_diff_skyline = diff_skyline
                esitimate_pos = np.array([x, y])

    return esitimate_pos


def modify_error(estimate_pos, target_pos, final_pos, seg_i):
    sigma_len = 0.1 ##
    sigma_ang = 1   ##

    vector_move = target_pos - estimate_pos
    len_error = 1 + np.random.normal(0, sigma_len)
    ang_error = math.radians(np.random.normal(0, sigma_ang))
    x_move = math.sqrt(len_error) * (vector_move[0] * math.cos(ang_error) - vector_move * math.sin(ang_error))
    y_move = math.sqrt(len_error) * (vector_move[1] * math.sin(ang_error) + vector_move * math.cos(ang_error))

    vector_move += np.array([x_move, y_move])

    next_initial_pos = final_pos + vector_move

    return next_initial_pos

def draw_trajectory(seg_num, path_nodes, path_seg_fin):
    tra_seg_fin_x = np.empty(seg_num)
    tra_seg_fin_y = np.empty(seg_num)
    for i in range(seg_num):
        shelvename = 'trajectory_' + str(i)
        with shelve.open(shelvename) as db:
            seg_trajectory = db['trajectroy']
            if i == 0:
                all_trajectory = seg_trajectory
            else:
                all_trajectory.append(seg_trajectory)
            tra_seg_fin_x[i] = seg_trajectory[len(seg_trajectory) - 1][0]
            tra_seg_fin_y[i] = seg_trajectory[len(seg_trajectory) - 1][1]

    len_trajectory = len(all_trajectory)    
    x_tra = np.empty(len_trajectory)
    y_tra = np.empty(len_trajectory)
    for i in range(len_trajectory):
        x_tra[i] = all_trajectory[i][0]
        y_tra[i] = all_trajectory[i][1]

    len_path = len(path_nodes)    
    x_path = np.empty(len_path)
    y_path = np.empty(len_path)
    for i in range(len_path):
        x_path[i] = path_nodes[i][0]
        y_path[i] = path_nodes[i][1]

    path_seg_fin_x = np.empty(seg_num)
    path_seg_fin_y = np.empty(seg_num)
    for i in range(seg_num):
        path_seg_fin_x[i] = path_seg_fin[i][0]
        path_seg_fin_y[i] = path_seg_fin[i][1]

    plt.plot(x_tra, y_tra, "r")
    plt.plot(x_path, y_path, "b")
    plt.plot(tra_seg_fin_x, tra_seg_fin_y, "ro")
    plt.plot(path_seg_fin_x, path_seg_fin_y, "bo")
    plt.show()

if __name__ == "__main__":
    main()