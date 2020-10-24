import sys
import numpy as np
import tensorflow as tf
from model import DeepCas

import six.moves.cPickle as pickle
import gzip
tf.set_random_seed(0)
import time

NUM_THREADS = 20

tf.flags.DEFINE_float("learning_rate", 0.01, "learning_rate.")
tf.flags.DEFINE_integer("sequence_batch_size", 20, "sequence batch size.")
tf.flags.DEFINE_integer("batch_size", 32, "batch size.")
tf.flags.DEFINE_integer("n_hidden_gru", 32, "hidden gru size.")
tf.flags.DEFINE_float("l1", 5e-5, "l1.")
tf.flags.DEFINE_float("l2", 1e-8, "l2.")
tf.flags.DEFINE_float("l1l2", 1.0, "l1l2.")
tf.flags.DEFINE_string("activation", "relu", "activation function.")
tf.flags.DEFINE_integer("n_sequences", 200, "num of sequences.")
tf.flags.DEFINE_integer("training_iters", 50*3200 + 1, "max training iters.")
tf.flags.DEFINE_integer("display_step", 100, "display step.")
tf.flags.DEFINE_integer("embedding_size", 50, "embedding size.")
tf.flags.DEFINE_integer("n_input", 50, "input size.")
tf.flags.DEFINE_integer("n_steps", 10, "num of step.")
tf.flags.DEFINE_integer("n_hidden_dense1", 32, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2", 16, "dense2 size.")
tf.flags.DEFINE_string("version", "v4", "data version.")
tf.flags.DEFINE_integer("max_grad_norm", 100, "gradient clip.")
tf.flags.DEFINE_float("stddev", 0.01, "initialization stddev.")
tf.flags.DEFINE_float("emb_learning_rate", 5e-05, "embedding learning_rate.")
tf.flags.DEFINE_float("dropout_prob", 1., "dropout probability.")
tf.flags.DEFINE_string("method", "deepstr", "choose method.")

config = tf.flags.FLAGS

def get_batch(x, xv,xv_s,y, sz, avg,step, batch_size=128,_function=0):
    batch_x = np.zeros((batch_size, len(x[0]), len(x[0][0])))
    batch_avg = np.zeros((batch_size,50))
    if(_function==0 or _function==1 or _function==2 or _function==4 or _function==7):
        batch_xv = np.zeros((batch_size,32))
        batch_xv_s = np.zeros((batch_size,32))
    elif(_function==5 or _function==6):
        batch_xv = np.zeros((batch_size,64))
        batch_xv_s = np.zeros((batch_size,64))
    elif(_function==3):
        batch_xv = np.zeros((batch_size,21))
        batch_xv_s = np.zeros((batch_size,21))
    batch_y = np.zeros((batch_size, 1))
    batch_sz = np.zeros((batch_size, 1))
    start = step * batch_size % len(x)
    for i in range(batch_size):
        batch_y[i, 0] = y[(i + start) % len(x)]
        batch_sz[i, 0] = sz[(i + start) % len(x)]
        batch_x[i, :] = np.array(x[(i + start) % len(x)])
        batch_xv[i,:] = np.array(xv[(i + start) % len(xv)])
        batch_xv_s[i,:] = np.array(xv_s[(i + start) % len(xv_s)])
        batch_avg[i,:] = np.array(avg[(i + start) % len(avg)])
    return batch_x, batch_xv,batch_xv_s, batch_y, batch_sz, batch_avg

version = config.version
sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True))
_function = config.method
#model = DeepCas(config,sess, node_vec,_function)
if(_function=="deepcon" or _function=="deepstr"):
    x_train,x_vec_tr,x_vec_tr_s, y_train, sz_train, vocabulary_size,avg_train = pickle.load(open('data/data_train32.pkl','rb'))
    x_test,x_vec_te, x_vec_te_s,y_test, sz_test, _ ,avg_test= pickle.load(open('data/data_test32.pkl','rb'))
    x_val, x_vec_va,x_vec_va_s,y_val, sz_val, _ ,avg_val= pickle.load(open('data/data_val32.pkl','rb'))
elif(_function=="deepcon_str"):
    x_train,x_vec_tr,x_vec_tr_s, y_train, sz_train, vocabulary_size,avg_train = pickle.load(open('data/data_train21.pkl','rb'))
    x_test,x_vec_te, x_vec_te_s,y_test, sz_test, _,avg_test = pickle.load(open('data/data_test21.pkl','rb'))
    x_val, x_vec_va,x_vec_va_s,y_val, sz_val, _ ,avg_val= pickle.load(open('data/data_val21.pkl','rb'))
node_vec = pickle.load(open('data/node_vec.pkl', 'rb'))


training_iters = config.training_iters
batch_size = config.batch_size
display_step = int(min(config.display_step, len(sz_train)/batch_size))


np.set_printoptions(precision=2)
model = DeepCon_Str(config,sess, node_vec,_function)
start = time.time()
step = 1
best_val_loss = 20
best_test_loss = 20
_val_loss = 20
_test_loss = 20
# Keep training until reach max iterations or max_try
train_loss = []
max_try = 100
trainloss = []
valloss = []
testloss = []
bestval = []
besttest = []
patience = max_try
_len = int(input("length"))
_time = int(input("times"))
while step * batch_size <training_iters:
    batch_x,batch_xvec,batch_xvec_s, batch_y, batch_sz,batch_avg = get_batch(x_train,x_vec_tr,x_vec_tr_s, y_train, sz_train,avg_train, step, batch_size=batch_size,_function=_function)
    model.train_batch(batch_x, batch_xvec,batch_xvec_s,batch_y, batch_sz,batch_avg)
    train_loss.append(model.get_error(batch_x, batch_xvec,batch_xvec_s,batch_y, batch_sz,batch_avg))
    #print(model.get_pred(batch_x, batch_y, batch_sz))
    #print(len(batch_y))
    trainloss.append(np.mean(train_loss))
    valloss.append(_val_loss)
    testloss.append(_test_loss)
    bestval.append(best_val_loss)
    besttest.append(best_test_loss)    
    if (step % display_step == 0):
        # Calculate batch loss
        print(patience)
        val_loss = []
        val_RMSPE = []
        for val_step in range(int(len(y_val)/batch_size)):
            val_x, val_xv,val_xv_s,val_y, val_sz,val_avg = get_batch(x_val, x_vec_va,x_vec_va_s,y_val, sz_val, avg_val,val_step, batch_size=batch_size,_function=_function)
            val_loss.append(model.get_error(val_x,val_xv,val_xv_s,val_y, val_sz,val_avg))
            if(_function==0 or _function==1):
                val_pre_y = model.get_predcas(val_x,val_xv,val_y, val_sz,val_avg)
            else:     
                val_pre_y = model.get_pred(val_x,val_xv,val_xv_s,val_y, val_sz,val_avg)
            #print(list(val_pre_y),list(val_y+1))
            val_RMSPE.append(sum([(each*each)**0.5 for each in (val_pre_y-val_y)/(val_y+1)])/batch_size)
        test_loss = []
        test_RMSPE = []
        for test_step in range(int(len(y_test)/batch_size)):
            test_x, test_xv,test_xv_s,test_y, test_sz,test_avg = get_batch(x_test,x_vec_te,x_vec_te_s, y_test, sz_test,avg_test, test_step, batch_size=batch_size,_function=_function)
            test_loss.append(model.get_error(test_x,test_xv,test_xv_s,test_y, test_sz,test_avg))
            if(_function==0 or _function==1):
                test_pre_y = model.get_predcas(test_x,test_xv,test_y, test_sz,test_avg)
            else:
                test_pre_y = model.get_pred(test_x,test_xv,test_xv_s,test_y, test_sz,test_avg)
            test_RMSPE.append(sum([(each*each)**0.5 for each in (test_pre_y-test_y)/(test_y+1)])/batch_size)
            
        _val_loss = np.mean(val_loss)
        _test_loss = np.mean(test_loss)
        _test_RMSPE = np.mean(test_RMSPE)
        _val_RMSPE = np.mean(val_RMSPE)
        if (np.mean(val_loss) < best_val_loss):
            best_val_loss = np.mean(val_loss)
            best_test_loss = np.mean(test_loss)
            patience = max_try
        print("#" + str(step/display_step) + 
              ", Training Loss= " + "{:.6f}".format(np.mean(train_loss)) + 
              ", Validation Loss= " + "{:.6f}".format(np.mean(val_loss)) + 
              ", Test Loss= " + "{:.6f}".format(np.mean(test_loss)) + 
              ", RMSPE Valid Loss= " + "{:.6f}".format(_val_RMSPE) + 
              ", RMSPE Test Loss= " + "{:.6f}".format(_test_RMSPE)
             )
        train_loss = []
        patience -= 1
        if not patience:
            break
        
    step += 1
print(step)
pickle.dump((trainloss,valloss,testloss,bestval,besttest, time.time()-start),open('data/'+str(_len)+'/'+str(_time)+'/'+str(_function)+'.pkl','wb+'))
#pickle.dump((testloss,besttest,),open('data/1/'+str(_function)+'test.pkl','wb+'))
print ("Finished!\n----------------------------------------------------------------")
print ("Time:", time.time()-start)
print ("Valid Loss:", best_val_loss)
print ("Test Loss:", best_test_loss)


