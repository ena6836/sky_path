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
import pandas as pd

def main():
    #dem = np.array(Image.open('\\Users\\Maenaka\\python\\20190204\\mars_spirit_landing_site\\dem.tif'))
    #dem = np.array(Image.open('/home/shuto/python/20190131/mars_spirit_landing_site/dem.tif'))

    shelvename = 'dem_s_rough'
    with shelve.open(shelvename) as db:
        dem = db['dem']

    global y_len
    y_len = dem.shape[0]
    global x_len
    x_len = dem.shape[1]

    global grid_size
    grid_size = 5.05
    global max_len
    max_len = math.sqrt(2) * grid_size
    global max_ang
    max_ang = 15

    dijkstra(dem)


def calc_sl_dif(node, graph, dem):
    x0 = node[0]
    y0 = node[1]
    node_name = (x0, y0)

    k = 0
    if x0 == 0 and 0 < y0 < y_len - 1:
        neighbors = [(x0, y0 - 1), (x0 + 1, y0 - 1), (x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)]
    elif x0 == x_len - 1 and 0 < y0 < y_len - 1:
        neighbors = [(x0, y0 - 1), (x0, y0 + 1), (x0 - 1, y0 + 1), (x0 - 1, y0), (x0 - 1, y0 - 1)]
    elif 0 < x0 < x_len - 1 and y0 == 0:
        neighbors = [(x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1), (x0 - 1, y0 + 1), (x0 - 1, y0)]
    elif 0 < x0 < x_len - 1 and y0 == y_len - 1:
        neighbors = [(x0, y0 - 1), (x0 + 1, y0 - 1), (x0 + 1, y0), (x0 - 1, y0), (x0 - 1, y0 - 1)]
    elif x0 == 0 and y0 == 0:
        neighbors = [(x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1)]
    elif x0 == 0 and y0 == y_len - 1:
        neighbors = [(x0, y0 - 1), (x0 + 1, y0 - 1), (x0 + 1, y0)]
    elif x0 == x_len - 1 and y0 == 0:
        neighbors = [(x0, y0 + 1), (x0 - 1, y0 + 1), (x0 - 1, y0)]
    elif x0 == x_len - 1 and y0 == y_len - 1:
        neighbors = [(x0, y0 - 1), (x0 - 1, y0), (x0 - 1, y0 - 1)]
    else:
        neighbors = [(x0, y0 - 1), (x0 + 1, y0 - 1), (x0 + 1, y0), (x0 + 1, y0 + 1), (x0, y0 + 1), (x0 - 1, y0 + 1), (x0 - 1, y0), (x0 - 1, y0 - 1)]

    skyline = dem[y0][x0][1:]

    for nd in neighbors:
        x = nd[0]
        y = nd[1]
        
        skyline_nei = dem[y][x][1:]

        azi = 0
        sl_dif = 0
        for azi in range(360):
            sl_dif += (skyline[azi] - skyline_nei[azi]) ** 2

    return sl_dif

    

def calc_weight(n_current, n_adj, dem, graph):
    len_edge = np.linalg.norm(n_adj - n_current) * grid_size
    height_current = dem[n_current[1]][n_current[0]][0]
    height_adj = dem[n_adj[1]][n_adj[0]][0]
    ang = math.degrees(math.atan(abs(height_adj - height_current) / len_edge))
    sl_dif = calc_sl_dif(n_adj, graph, dem)
    
    if ang > 15.0:
        weight = 99999999
    else:
        weight = 0.3 * len_edge / max_len + 0.3 * ang / max_ang + 0.4 * (1 - sl_dif)

    return weight


def dijkstra(dem):
    x_len = dem.shape[1]
    y_len = dem.shape[0]

    graph = nx.grid_2d_graph(x_len, y_len)

    i = 0

    for y in range(0, y_len):
        for x in range(0, x_len):
            current_name = (x, y)
            graph.add_node(current_name, height = dem[y][x][0], x = x, y = y)

    for y in range(0, y_len):
        for x in range(0, x_len):
            current_name = (x, y)
            east_name = (x+1, y)
            south_name = (x, y+1)
            es_name = (x+1, y+1)
            n_current = np.array([x, y])
            n_east = np.array([x+1, y])
            n_south = np.array([x, y+1])
            n_es = np.array([x+1, y+1])
            if x != x_len - 1:
                graph.add_edge(current_name, east_name, weight = calc_weight(n_current, n_east, dem, graph))
                graph.add_edge(east_name, current_name, weight = calc_weight(n_east, n_current, dem, graph))
            if y != y_len - 1:
                graph.add_edge(current_name, south_name, weight = calc_weight(n_current, n_south, dem, graph))
                graph.add_edge(south_name, current_name, weight = calc_weight(n_south, n_current, dem, graph))
            if x != x_len - 1 and y != y_len - 1:
                graph.add_edge(current_name, es_name, weight = calc_weight(n_current, n_es, dem, graph))
                graph.add_edge(es_name, current_name, weight = calc_weight(n_es, n_current, dem, graph))
                graph.add_edge(east_name, south_name, weight = calc_weight(n_east, n_south, dem, graph))
                graph.add_edge(south_name, east_name, weight = calc_weight(n_south, n_east, dem, graph))

            i += 1
            print(x, y)

    n_start = (0, 0)
    n_goal = (x_len, y_len)

    pos = {}
    i = 0
    for y in range(y_len):
        for x in range(x_len):
            pos[(x, y)] = (x, y)
            i += 1

    selected_node = nx.dijkstra_path(graph, n_start, n_goal)
    selected_edge = []
    len_path = 0
    i = 0
    for nd in selected_node:
        if i != 0:
            selected_edge.append((pr_nd, nd))
            if pr_nd[0] == nd[0] or pr_nd[1] == nd[1]:
                len_path += 1
            else:
                len_path += math.sqrt(2)
        pr_nd = nd
        i += 1

    print(len_path * grid_size)
    print(selected_edge)

    len_path_shelve = 'path_len'
    with shelve.open(len_path_shelve) as db:
        db['len'] = len_path * grid_size

    len_path_shelve = 'path_edge'
    with shelve.open(len_path_shelve) as db:
        db['edge'] = selected_edge


    nx.draw_networkx_edges(graph, pos, edgelist=selected_edge, width=2, edge_color='b')

    nx.draw_networkx_nodes(graph, pos, node_color='w', node_size=0.1)
    #nx.draw_networkx_nodes(graph, pos, nodelist=selected_node, node_color='r', node_size=0.2)

    plt.axis('off')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig("path_proposed.png")
    plt.show()

if __name__ == "__main__":
    main()