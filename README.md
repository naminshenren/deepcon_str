# deepcon_str
This code is adapted from [*DeepCas* as described in the paper](https://arxiv.org/abs/1611.05373):
	
	Prediction of Information Cascades via Content and Structure Proximity Preserved Graph Level Embedding
	Xiaodong Feng, Qihang Zhao, Zhen Liu. 
	Under the second-round review of Information Sciences.

### Basic Usage


```
#### Options
You can check out the other options available to use with *DeepCas* using:<br/>
```{r, engine='bash', count_lines}
python gen_walks/gen_walks.py --help
th main/run.lua --help
```
#### Input
global_graph.txt lists each node's neighbors in the global network:

	node_id \t\t (null|neighbor_id:weight \t neighbor_id:weight...)

"\t" means tab, and "null" is used when a node has no neighbors.

cascade_(train|val|test).txt list cascades, one cascade per line:

	cascade_id \t starter_id... \t constant_field \t num_nodes \t source:target:weight... \t label...

"starter_id" are nodes who start the cascade, "num_nodes" counts the number of nodes in the cascade.

cascade.txt:cascade_train.txt+cascade_val.txt+cascade_test.txt

Since we can predict cascade growth at different timepoints, there could be multiple labels. 

## Tensorflow Implementation
### Prerequisites
Tensorflow 1.11.1

### Basic Usage
To run *DeepCas* tensorflow version on a test data set, execute the following command:<br/>
```{r, engine='bash', count_lines}
cd DeepCas
python gen_walks/gen_walks.py --dataset test-net
python gen_con.py
python gen_str.py
cd tensorflow
python preprocess.py
python run.py
```

## Citing
If you find *DeepCon+Str* useful for your research, please consider citing the following paper:
Xiaodong Feng, Qihang Zhao, Zhen Liu. Prediction of Information Cascades via Content and Structure Proximity Preserved Graph Level Embedding, 
Under the second-round review of Information Sciences.
	

## Miscellaneous

Please send any questions you might have about the code and/or the algorithm to <fengxd1988@hotmail.com>.
