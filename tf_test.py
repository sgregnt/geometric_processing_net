import tensorflow as tf
import numpy as np
import numpy as np
import matplotlib
import copy
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import networkx as nx
from random import sample
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib
import copy
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import networkx as nx
from random import sample
import pickle
import scipy.io
import sys, os
sys.path.insert(0, '/home/greg/libigl/libigl/python/')
import pyigl as igl
from iglhelpers import *
from scipy.linalg import sqrtm
from itertools import combinations

#
# g = tf.Graph()
# with g.as_default():
#
#    batch = 2
#    pred = list(np.arange(12 * 11/2))
#    pred.extend(np.arange(12 * 11/2))
#    pred = np.array(pred).astype('float32')
#    d = np.arange(batch * 12).astype('float32').reshape((batch, 12, 1))
#
#    hessian = tf.Variable(initial_value=[0] * (batch * 12 * 12), dtype='float32')
#    ut = np.triu_indices(12, k=1)  # upper trianlge exluding diagonal
#
#
#    all_indices = np.arange((12 * 12)).reshape((12, 12))  # all indeces
#    all_indices_batch = np.arange((batch * 12 * 12))
#    single_batch_indexing = all_indices[ut]
#
#    all_indixing_with_batch = []
#
#    for b in range(batch):
#        b_index = np.array(single_batch_indexing) + b * (12 * 12)
#        all_indixing_with_batch.extend(b_index)
#
#    all_indixing_with_batch = np.array(all_indixing_with_batch).astype('int32')
#    # tmp = tf.tile(all_indices[ut],[batch], name=None)
#    # batch_all_indices = tf.reshape(tmp, [batch, len(all_indices[ut])])
#
#    # pred = b * (12 * 12)/2
#    tmp = tf.scatter_update(hessian, all_indixing_with_batch, pred)
#    # tmp = tf.scatter_update(hessian, all_indices[ut], pred)
#    h_mat_ut = tf.reshape(tmp, [batch, 12, 12])
#
#    # the matrix is symmetric
#    h_mat_full = h_mat_ut + tf.transpose(h_mat_ut, perm=[0,2,1])
#
#    # update diagonal
#    row_sum = -tf.reduce_sum(h_mat_full, axis=[1])
#    hessian_final = -tf.to_float(tf.linalg.set_diag(h_mat_full, row_sum))
#
#    # compute quadratic form
#    quadratic_form = tf.matmul(a=tf.matmul(a=d, b=hessian_final, transpose_a=True), b=d)
#
#    # tf.unstack()
#
# with tf.Session(graph=g) as sess:
#    sess.run(tf.initialize_all_variables())
#    print(sess.run(hessian_final))
#    print(sess.run(quadratic_form))
#    # print(sess.run(b))
if False:
   nFocus = 12
   BATCH = 32
   ut = np.triu_indices(nFocus, k=1)

   # use upper triangular mask to obtain indeces for the upper triangular part
   # of a matrix, next will have to replicate accross the batches.
   all_indices = np.arange((nFocus * nFocus)).reshape((nFocus, nFocus))
   single_batch_indexing = all_indices[ut]

   all_indixing_with_batch = []

   # get indexes for upper triangular parts of the hessian
   # for each batch, when the batches are flattened.
   for b in range(BATCH):
      # bor each batch the indices are shifted by hessian size i.e., by S-by-S.
      b_index = np.array(single_batch_indexing) + b * (nFocus * nFocus)
      all_indixing_with_batch.extend(b_index)

   all_indixing_with_batch = np.array(all_indixing_with_batch).astype('int32')
   for i in range(len(all_indixing_with_batch)):
      # print("Hx_tmp[%s] = wx_ij_flat[%s]" % (all_indixing_with_batch[i], i))
      print("a = Hx_tmp[%s].assign(wx_ij_flat[%s]) + a" % (all_indixing_with_batch[i], i))
      # print("Hy_tmp[%s] = wy_ij_flat[%s]" % (all_indixing_with_batch[i], i))
      print("b = Hy_tmp[%s].assign(wy_ij_flat[%s]) + b" % (all_indixing_with_batch[i], i))

if True:
   nFocus = 12
   BATCH = 2
   ut = np.triu_indices(nFocus, k=1)

   # use upper triangular mask to obtain indeces for the upper triangular part
   # of a matrix, next will have to replicate accross the batches.
   all_indices = np.arange((nFocus * nFocus)).reshape((nFocus, nFocus))
   single_batch_indexing = all_indices[ut]

   all_indixing_with_batch = []

   # get indexes for upper triangular parts of the hessian
   # for each batch, when the batches are flattened.
   for b in range(BATCH):
      # bor each batch the indices are shifted by hessian size i.e., by S-by-S.
      b_index = np.array(single_batch_indexing) + b * (nFocus * nFocus)
      all_indixing_with_batch.extend(b_index)

   all_indixing_with_batch = np.array(all_indixing_with_batch).astype('int32')
   print("shape = tf.constant([BATCH * nFocus * nFocus])")
   print("indices = tf.constant([")
   for i in range(len(all_indixing_with_batch)):
      # print("Hx_tmp[%s] = wx_ij_flat[%s]" % (all_indixing_with_batch[i], i))
      print("[ %s ]," % (all_indixing_with_batch[i]))
   print("])")

# patch = scipy.io.loadmat('/home/greg/newproject/data/bar4x4.mat')
# patch = scipy.io.loadmat('/home/greg/newproject/data/bar4x4_tmesh.mat')
# patch = scipy.io.loadmat('/home/greg/newproject/data/bar4x4_V_T_S_Sc_SLc_normT_detT_E.mat')
# print("hello")