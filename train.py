import argparse
import tensorflow as tf
import importlib
import os
import sys
from utils.training_utils import plot_loss_evolution, get_all_inputs, unpack_inputs, run_sess, gen_feed_dict
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from build_dataset import read_patch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='meshnet', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=1500, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=100000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00005) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():

    epoch_mean_loss = []
    test_epoch_mean_loss = []
    batch_mean_loss = []
    test_batch_mean_loss = []

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):

            nV = 27
            nv_sub = 12
            edges_ss_n = int(nv_sub * (nv_sub - 1) / 2)

            inputs_placeholder_tensors = MODEL.mesh_placeholder_inputs(BATCH_SIZE, nV, nv_sub)

            (v,
             displacement,
             deng_x_at_x0_sub,
             eng_at_x0_sub,
             eng_at_xnew_sub,
             d_n,
             alpha,
             epsilon,
             edges_ss) = inputs_placeholder_tensors

            is_training = tf.placeholder(tf.bool, shape=())
            print(is_training)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0) # this will hold global step
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            w_ij = MODEL.get_model(v=v,
                                   displacement=displacement,
                                   edges_ss=edges_ss,
                                   edges_ss_n=edges_ss_n,
                                   final_activation = tf.nn.relu, # relu worded to achieve 0.085932 loss
                                   is_training = is_training,
                                   bn_decay = bn_decay)

            energy_dif_loss, end_points,  = MODEL.mesh_get_loss(w_ij=w_ij,
                                                 deng_x_at_x0_sub=deng_x_at_x0_sub,
                                                 eng_at_x0_sub=eng_at_x0_sub,
                                                 eng_at_xnew_sub=eng_at_xnew_sub,
                                                 d_n=d_n,
                                                 alpha=alpha,
                                                 epsilon=epsilon,
                                                 BATCH=BATCH_SIZE,
                                                 nv_sub=nv_sub)

            tf.summary.scalar('loss', energy_dif_loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(energy_dif_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

        restore = False

        if restore:
            print(LOG_DIR)
            saver.restore(sess, os.path.join(LOG_DIR, "model_sub_positivedefinite_second_try_big_reduced_lr.ckpt"))
        else:
            # To fix the bug introduced in TF 0.12.1 as in
            # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
            init = tf.global_variables_initializer()
            sess.run(init, {is_training: True})

        ops = {'v': v,
               'displacement': displacement,
               'deng_x_at_x0_sub': deng_x_at_x0_sub,
               'eng_at_x0_sub': eng_at_x0_sub,
               'eng_at_xnew_sub': eng_at_xnew_sub,
               'd_n': d_n,
               'alpha': alpha,
               'epsilon': epsilon,
               'edges_ss': edges_ss,
               'is_training': is_training,
               'w_ij': w_ij,
               'loss': energy_dif_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'd_eng_at_xnew_sub' : end_points['eng_at_xnew_sub'],
               'd_energy_lin_removed' : end_points['energy_lin_removed'],
               'd_energy_dif' : end_points['energy_dif'],
               'd_Qx' : end_points['Qx'],
               'd_Qy' : end_points['Qy'],
               'Hx_final': end_points['Hx_final'],
               'Hy_final': end_points['Hy_final']}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            batch_mean_loss_epoch, running_loss_epoch = train_one_epoch(sess, ops, train_writer)
            # test_batch_mean_loss_epoch, test_running_loss_epoch = eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                # pass
                save_path = saver.save(sess, os.path.join(LOG_DIR, "test_after_clearing.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            epoch_mean_loss.extend(batch_mean_loss_epoch)
            batch_mean_loss.extend(running_loss_epoch)

    return epoch_mean_loss, batch_mean_loss, test_epoch_mean_loss, test_batch_mean_loss

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """

    is_training = True

    batch_mean_loss = []
    epoch_mean_loss = []

    patch = read_patch('/home/greg/repomeshnet/meshnet/python/data/mesh_examples/grid2xd2.mat')

    for fn in range(1):
        log_string('----' + str(fn) + '-----')
        sub = True

        if sub:
            # current_inputs = provider.loadmatDataFile('/home/greg/repomeshnet/meshnet/python/data/datasets/dataset_100.mat')
            # current_inputs = provider.loadmatDataFile('./data/datasets/small_sub_path_testb.mat')
            current_inputs = provider.loadmatDataFile('./data/datasets/test_after_cleaning.mat')
            all_inputs = get_all_inputs(current_inputs, old=False)
        else:
            if False:
                current_inputs = provider.loadmatDataFile('/home/greg/repomeshnet/meshnet/python/data/datasets/small_path_15000.mat')

        file_size = len(current_inputs['all_v'])
        num_batches = file_size // BATCH_SIZE

        total_seen = 0
        loss_sum = 0

        # num_batches = 100

        for batch_idx in range(num_batches):

            # print(batch_idx)
            start_idx = 0 # when randomly generated non eed to pick subarrays from input data.
            end_idx = BATCH_SIZE

            if sub:
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE
            else:
                current_inputs = provider.gen_samples_batch(patch, 32)

            inputs_b = unpack_inputs(all_inputs=all_inputs, start_idx=start_idx, end_idx=end_idx)
            feed_dict = gen_feed_dict(inputs_b, is_training, ops)
            all_tensors = run_sess(sess, feed_dict, ops)

            (summary,
             step,
             _,
             loss_val,
             d_energy_at_new_fV,
             d_energy_lin_removed,
             d_energy_dif,
             d_Qx,
             d_Qy,
             Hx_final,
             Hy_final
             ) = all_tensors

            total_seen += BATCH_SIZE
            loss_sum += loss_val

            if batch_idx % 10 == 0:
                train_writer.add_summary(summary, step)
                train_writer.flush()

            batch_mean_loss.append(loss_val)
        epoch_mean_loss.append(loss_sum / float(num_batches))
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))

    return epoch_mean_loss, batch_mean_loss

if __name__ == "__main__" :

    epoch_mean_loss, batch_mean_loss, test_epoch_mean_loss, test_batch_mean_loss = train()

    plot_loss_evolution(np.log(np.array(batch_mean_loss) + 1), 'LOG: batch_mean_loss losses')
    plot_loss_evolution(np.log(np.array(epoch_mean_loss) + 1), 'LOG: epoch averaged losses')

    plot_loss_evolution(np.array(batch_mean_loss[-500:]), 'LOG: batch_mean_loss losses')
    plot_loss_evolution(np.array(epoch_mean_loss[-500:]), 'LOG: epoch averaged losses')

    #----------------------------------------------------------------------------------------

    plot_loss_evolution(np.log(np.array(test_batch_mean_loss) + 1), 'TEST LOG: batch_mean_loss losses')
    plot_loss_evolution(np.log(np.array(test_epoch_mean_loss) + 1), 'TEST LOG: epoch averaged losses')

    plot_loss_evolution(np.array(test_batch_mean_loss[-500:]), 'TEST LOG: batch_mean_loss losses')
    plot_loss_evolution(np.array(test_epoch_mean_loss[-500:]), 'TEST LOG: epoch averaged losses')

    LOG_FOUT.close()