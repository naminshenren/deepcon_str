import numpy as np
import os
from optparse import OptionParser
import sys
import networkx as nx
import node2vec
import time
import random
import math
import six.moves.cPickle as pickle
from fastdtw import fastdtw
from gensim.models import Word2Vec

cascade_file_prefix2 = r"..\data\test-net\cascade.txt"

def file_len(fname):
  lines = 0
  for line in open(fname):
    lines += 1
  return lines

def parse_graph(graph_string):
  parts = graph_string.split("\t")
  edge_strs = parts[4].split(" ")
  edge_strs2 = parts[1].split(" ")

  node_to_edges = dict()
  nx_G = nx.DiGraph()
  for each in edge_strs2:
    nx_G.add_edge(0,int(each))
  for edge_str in edge_strs:
    edge_parts = edge_str.split(":")
    source = int(edge_parts[0])
    target = int(edge_parts[1])

    if not source in node_to_edges:
      neighbors = list()
      node_to_edges[source] = neighbors
    else:
      neighbors = node_to_edges[source]
    #neighbors.append((target, get_global_degree(target)))
    nx_G.add_edge(source, target)
  return parts[0],nx_G

def return_max_degree(dic):
    return(max([dic[each] for each in dic]))

def cost(a,b):
    ep = 0.5
    m = float(max(a,b) + ep)
    mi = float(min(a,b) + ep)
    return ((m/mi) - 1)

def _randomwalkk(nx_G,i):
    pseudo_count = 0.01
    roots = list()
    roots_noleaf = list()
    str_list = list()
    #str_list.append(str(i))
    probs = list()
    probs_noleaf = list()
    weight_sum_noleaf = 0.0
    weight_sum = 0.0
    for node, weight in nx_G.out_degree_iter(weight="weight"):
        org_weight = weight
        if weight == 0: weight += pseudo_count
        weight_sum += weight
        if org_weight > 0:
            weight_sum_noleaf += weight
    for node, weight in nx_G.out_degree_iter(weight="weight"):
        org_weight = weight
        if weight == 0: weight += pseudo_count
        roots.append(node)
        prob = weight / weight_sum
        probs.append(prob)
        if org_weight > 0:
            roots_noleaf.append(node)
            prob = weight / weight_sum_noleaf
            probs_noleaf.append(prob)
    sample_total = 2000
    first_time = True
    G = node2vec.Graph(nx_G, True, 1, 1)
    G.preprocess_transition_probs()
    while True:
        if first_time:
            first_time = False
            node_list = roots
            prob_list = probs
        else:
            node_list = roots_noleaf
            prob_list = probs_noleaf
        n_sample = min(len(node_list), sample_total)
        if n_sample <= 0: break
        sample_total -= n_sample

        sampled_nodes = np.random.choice(node_list, n_sample, replace=False, p=prob_list)
        walks = G.simulate_walks(len(sampled_nodes), 14, sampled_nodes)
        for walk in walks:
            str_list.append(' '.join(str(k) for k in walk))
    return '\t'.join(str_list)

def read_graphh(i):
    graph_file = cascade_file_prefix2
    num_graphs = file_len(graph_file)
    rfile = open(graph_file, 'r')
    start_time = time.time()
    j=1
    for line in rfile:
        if(j<i):
            j=j+1
            continue
        line = line.rstrip('\r\n')
        _id,gra = parse_graph(line)
        rfile.close()
        return _id,gra
def read_walks_set():
  walks = []
  graph_walk_file = r"D:\ProgramData\Anaconda3\Scripts\DEEE\Deepcas\data\test-net\random_struct_10.txt"
  rfile = open(graph_walk_file, 'r')
  for line in rfile:
    line = line.rstrip('\r\n')
    walk_strings = line.split('\t')
    for i,walk_str in enumerate(walk_strings):
      #print(walk_str)      
      if(i == 0): continue
      walks.append(walk_str.split(" "))
  rfile.close()
  return walks
    
alist = []
for i in range(1,file_len(cascade_file_prefix2)+1):
    idd,gra = read_graphh(i)
    alist.append(gra)
    
dic = {}
for i in range(file_len(cascade_file_prefix2)):
    dic[str(i)] = {}
    for each in alist[i].nodes():
        dic[str(i)][each] = nx.shortest_path_length(alist[i], source=0,target=int(each), weight=None)

strdic = {}

for i in range(file_len(cascade_file_prefix2)):
    strdic[str(i)] = {}
    for j in range(file_len(cascade_file_prefix2)):
        if(j<=i):
            continue
        distance = 0
        _1 = return_max_degree(dic[str(i)])
        _2 = return_max_degree(dic[str(j)])
        for i1 in range(min(_1,_2)):
            node1 = [each for each in dic[str(i)] if (dic[str(i)][each]==i1)]
            node2 = [each for each in dic[str(j)] if (dic[str(j)][each]==i1)]
            degree1 = [alist[i].degree(each) for each in node1]
            degree2 = [alist[j].degree(each) for each in node2]
            #print(degree1,degree2,node1,alist[j].neighbors(node2[0]))
            dist, path = fastdtw(degree1,degree2,radius=1,dist=cost)
            distance+=dist*math.log((min(_1,_2)-i1+2))
        strdic[str(i)][str(j)] = math.exp(-distance)
        
GG = nx.DiGraph()
for i in range(file_len(cascade_file_prefix2)):
    for j in range(file_len(cascade_file_prefix2)):
        if(j<=i):
            continue
        if(strdic1[str(i)][str(j)]<0.1):
            continue
        GG.add_edge(i,j,weight=float(strdic[str(i)][str(j)]))
        GG.add_edge(j,i,weight=float(strdic[str(i)][str(j)])) 
s = _randomwalkk(GG,1)
with open(r"..\data\test-net\random_struct_14.txt","w") as f:
    f.write(s)
    
walks = read_walks_set()
from gensim.models import Word2Vec
embed_file = r"..\data\test-net\graph_vec_10_32.txt"
model = Word2Vec(walks, size=32, window=10, min_count=0, sg=1, workers=8,
                   iter=5)
model.wv.save_word2vec_format(embed_file)

        