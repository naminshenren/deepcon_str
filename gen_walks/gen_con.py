import numpy as np
import os
from optparse import OptionParser
import sys
import networkx as nx
import node2vec
import time
import random
import six.moves.cPickle as pickle
from gensim.models import Word2Vec

global_graph_file = r"..\data\test-net\global_graph.txt"
sets = ["train", "val", "test"]

node_to_degree = dict()
edge_to_weight = dict()
#pseudo_count = 0.01

def get_global_info():
  rfile = open(global_graph_file, 'r')
  for line in rfile:
    line = line.rstrip('\r\n')
    parts = line.split("\t\t")
    source = int(parts[0])
    if parts[1] != "null":
      node_freq_strs = parts[1].split("\t")
      for node_freq_str in node_freq_strs:
        node_freq = node_freq_str.split(":")
        weight = int(node_freq[1])
        target = int(node_freq[0])
        edge_to_weight[(source, target)] = weight
      degree = len(node_freq_strs)
    else:
      degree = 0
    node_to_degree[source] = degree
  rfile.close()
  return

def get_global_degree(node):
  return node_to_degree.get(node, 0)

def get_edge_weight(source, target):
  return edge_to_weight.get((source, target), 0)


def parse_graph(graph_string):
  parts = graph_string.split("\t")
  edge_strs = parts[4].split(" ")

  node_to_edges = dict()
  nx_G = nx.DiGraph()
  for edge_str in edge_strs:
    edge_parts = edge_str.split(":")
    source = int(edge_parts[0])
    target = int(edge_parts[1])

    if not source in node_to_edges:
      neighbors = list()
      node_to_edges[source] = neighbors
    else:
      neighbors = node_to_edges[source]
    neighbors.append((target, get_global_degree(target)))
    nx_G.add_edge(source, target)
  return parts[0],nx_G

def file_len(fname):
  lines = 0
  for line in open(fname):
    lines += 1
  return lines
  
 def read_graphh(i):
    graph_file = cascade_file_prefix1
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
    for node, weight in nx_G.out_degree(weight="weight"):
        org_weight = weight
        if weight == 0: weight += pseudo_count
        weight_sum += weight
        if org_weight > 0:
            weight_sum_noleaf += weight
    for node, weight in nx_G.out_degree(weight="weight"):
        org_weight = weight
        if weight == 0: weight += pseudo_count
        roots.append(node)
        prob = weight / weight_sum
        probs.append(prob)
        if org_weight > 0:
            roots_noleaf.append(node)
            prob = weight / weight_sum_noleaf
            probs_noleaf.append(prob)
    sample_total = 4000
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
def read_walks_set():
  walks = []
  graph_walk_file = r"D:\ProgramData\Anaconda3\Scripts\deepcas\data\test-net\random_content_10.txt"
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

if __name__ == "__main__":
  get_global_info()
  cascade_file_prefix1 = r"..\data\test-net\cascade.txt"
  alist = []
  dic = {}
  x = file_len(cascade_file_prefix1)
  for i in range(x):
    dic[str(i+1)] = read_graphh(i+1)
  nx_GG = nx.DiGraph()
  x = file_len(cascade_file_prefix1)
  for i in range(x):
    id1,gra1 = dic[str(i+1)]
    node1 = gra1.nodes()
    #nx_GG = nx.DiGraph()
    for j in range(x):
        if(j<=i):
            continue
        else:
            id2,gra2 = dic[str(j+1)]
            node2 = gra2.nodes()
            jiao =  [get_global_degree(a) for a in node1 if a in node2]
            we = sum(jiao)
            #alist.append(we)
            #if(we == 0):
                #we = random.randint(-20,0)  
            if(we>0):                
                nx_GG.add_edge(id1, id2, weight=we)
                nx_GG.add_edge(id2, id1, weight=we)
  pickle.dump((nx_GG), open(r'..\data\test-net\global_content_similarity.pkl','wb+'))
  alist = pickle.load(open(r'..\data\test-net\global_content_similarity.pkl','rb'))
  s = _randomwalkk(alist,1)
  with open(r"..\data\test-net\random_content_10.txt","w") as f:
    f.write(s)
  walks = read_walks_set()

  embed_file = r"..\data\test-net\content_vec_10_32.txt"
  model = Word2Vec(walks, size=32, window=10, min_count=0, sg=1, workers=8,
                   iter=5)
  model.wv.save_word2vec_format(embed_file)
  
    