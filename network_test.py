import argparse
import tensorflow as tf
import importlib
import os
import sys
from utils.training_utils import gen_feed_dict_inference
from utils.mesh_utils import input_prep
from utils.optimization_utils import optim_res_unpack, line_search_armijio, do_optimization_result_unpack
from utils.debug_and_plot_utils import plot_optimization_sequence, concatinate_optimization_sequence_params
from utils.energy_utils import do_verbose, dirichlet_sym_gd_bfgs_nn_weak_armijo_optimizer_step, dirichlet_sym_compare_energies
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from build_dataset import read_patch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

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

def get_bn_decay(batch): # not clear what to do with this decay argument
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def extract_Hs(Hx_final_out, Hy_final_out):
    Hx = Hx_final_out[0, :, :]
    Hy = Hy_final_out[0, :, :]
    return Hx, Hy

def pack_batch(fvi, vi, subTi, BATCH):
    displacement = fvi[:, 0:2] - vi[:, 0:2]
    subvertices = np.unique(subTi)
    edges_ss = np.array(list(combinations(fvi[subvertices, ::], 2)))
    # reshape to have all the edeges in one row (4, since we look at 2d problem)
    # leave out only two coordinates z == 0
    edges_ss = edges_ss[:, :, 0:2].reshape((edges_ss.shape[0], 4))
    inputs_b = (np.array([vi[:, 0:2]] * BATCH), np.array([displacement] * BATCH), np.array([edges_ss] * BATCH))
    return inputs_b

def compare_approximations(file, sess, Hx_final, Hy_final, w_ij, ops):
    sub_patch = read_patch(file)
    verbose = True
    # Prepare input, randomness in fv is achieved by adding a noise
    vi, Ti, fv_o, subTi = input_prep(sub_patch, do_show=False, avoid_triangle_flip=True)
    B_inv_new = None
    N = 6
    BATCH = 32
    type = 'NN'

    def get_nn_hessians(fv, v, subt):
        inputs_b = pack_batch(fv, v, subt, BATCH)
        feed_dict = gen_feed_dict_inference(inputs_b, is_training=False, ops=ops)
        (Hx_final_out, Hy_final_out, w_ij_out) = sess.run((Hx_final, Hy_final, w_ij), feed_dict=feed_dict)
        (Hx, Hy) = extract_Hs(Hx_final_out, Hy_final_out)
        return (Hx, Hy)

    for i in range(N):
        dirichlet_sym_compare_energies(get_nn_hessians=get_nn_hessians,
                                       source=vi,
                                       t=Ti,
                                       target=fv_o,
                                       subt=subTi)


    plt.show()


def run_optimization(file, sess, Hx_final, Hy_final, w_ij, ops):
    """ Generate data set of N samples, with mesh taken from file"""

    sub_patch = read_patch(file)
    # all_plot_params = []
    verbose = True

    # Prepare input, randomness in fv is achieved by adding a noise
    vi, Ti, fv_o, subTi = input_prep(sub_patch, do_show=False, avoid_triangle_flip=True)
    B_inv_new = None
    N = 8
    BATCH = 32
    type = 'NN'
    for type in ('NN', 'GD', 'BFGS'):

        fvi = fv_o.copy()
        print("*" * 1000)
        print(fvi)
        print("&" * 1000)
        all_plot_params = []

        def get_nn_hessians(fv, v, subt):
            inputs_b = pack_batch(fv, v, subt, BATCH)
            feed_dict = gen_feed_dict_inference(inputs_b, is_training=False, ops=ops)
            (Hx_final_out, Hy_final_out, w_ij_out) = sess.run((Hx_final, Hy_final, w_ij), feed_dict=feed_dict)
            (Hx, Hy) = extract_Hs(Hx_final_out, Hy_final_out)
            return (Hx, Hy)

        for i in range(N):

            params = dirichlet_sym_gd_bfgs_nn_weak_armijo_optimizer_step(line_search=line_search_armijio,
                                                                         get_nn_hessians=get_nn_hessians,
                                                                         source=vi,
                                                                         t=Ti,
                                                                         target=fvi,
                                                                         subt=subTi,
                                                                         verbose=verbose,
                                                                         # Back feed the next estimate of Hessian
                                                                         B_inv_new=B_inv_new,
                                                                         type=type)

            optimization_res = do_optimization_result_unpack(params=params, t=Ti, v=vi, verbose=verbose)

            (displacement,
             p_k,  # optimization_res['step_fv'],
             p_k_n,  # optimization_res['step_fv_n'],
             _,  # optimization_res['subvertices'],
             eng_xnew_first_order_approx,
             eng_xnew_x0_diff_sub,
             source,
             t,
             source_dim,
             t_dim,
             eng_at_x0_sub,
             eng_at_xnew,
             eng_at_xnew_sub,
             deng_x_at_x0_sub,
             alpha,
             edges_sub,
             epsilon,
             sub_source_dim,  # nv_sub
             edges_sub_dim,
             sv_at_x0,
             sv_at_x0_sub,
             sv_at_xnew,
             sv_at_xnew_sub,
             fv_at_xnew,
             eng_first_order_approx_xnew_x0_diff,
             t_flip,
             x0,
             eng_first_order_approx_xnew_x0_diff_sub,
             ray_deng_x_at_x0,
             ray_deng_x_at_xnew,
             eng_plot,
             ray_deng_x_at_x0_sub,
             ray_deng_x_at_xnew_sub,
             eng_plot_sub,
             span,
             type,
             B_inv_new,
             eng_xnew_x0_diff) = optim_res_unpack(optimization_res)

            plot_params = (ray_deng_x_at_x0,
             ray_deng_x_at_xnew,
             eng_plot,
             ray_deng_x_at_x0_sub,
             ray_deng_x_at_xnew_sub,
             eng_plot_sub,
             alpha,
             span,
             type)

            do_verbose("linear_diff", eng_first_order_approx_xnew_x0_diff, verbose)
            all_plot_params.append(plot_params)
            fvi = fv_at_xnew

        sequence_params = concatinate_optimization_sequence_params(all_plot_params=all_plot_params, do_show=True)
        plot_optimization_sequence(sequence_params=sequence_params, do_show=True, type=type)

    plt.show()
    return 0

def train():

    with tf.Graph().as_default() as g:
        with tf.device('/gpu:'+str(GPU_INDEX)):

            nV = 27
            nFocus = 12
            BATCH_SIZE = 32
            edges_ss_n = int(nFocus * (nFocus - 1) / 2)

            inputs_placeholder_tensors = MODEL.mesh_placeholder_inputs(BATCH_SIZE, nV, nFocus)

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

            ops = {'v': v,
                   'displacement': displacement,
                   'alpha' : alpha,
                   'epsilon' : epsilon,
                   'edges_ss': edges_ss,
                   'is_training': is_training}

            bn_decay = get_bn_decay(0)

            w_ij = MODEL.get_model(v=v,
                                   displacement=displacement,
                                   edges_ss=edges_ss,
                                   edges_ss_n=edges_ss_n,
                                   final_activation=tf.nn.relu,
                                   is_training=is_training,
                                   bn_decay=bn_decay)

            wx_ij_flat = tf.reshape(w_ij[:, :, 0], [-1])
            wy_ij_flat = tf.reshape(w_ij[:, :, 1], [-1])

            Hx_final, Hy_final = MODEL.assemble_h(wx_ij_flat, wy_ij_flat)

            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, os.path.join(LOG_DIR, "model_sub_positivedefinite_second_try_big_reduced_lr.ckpt"))
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')
        # _ = run_optimization(os.path.join(root_path, './data/mesh_examples/grind_and_subgrid.mat'), sess, Hx_final, Hy_final, w_ij, ops)
        compare_approximations(os.path.join(root_path, './data/mesh_examples/grind_and_subgrid.mat'), sess, Hx_final, Hy_final, w_ij, ops)
    return 0

if __name__ == "__main__" :
    train()

