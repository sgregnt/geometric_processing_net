import tensorflow as tf
import numpy as np
import utils.tf_util as tf_util

def mesh_placeholder_inputs(BATCH, vn, nv_sub):
    """ place holder for all the information read form input file.
    """

    v = tf.placeholder(tf.float32, shape=(BATCH, vn, 2))
    displacement = tf.placeholder(tf.float32, shape=(BATCH, vn, 2))
    deng_x_at_x0_sub = tf.placeholder(tf.float32, shape=(BATCH, nv_sub, 2))
    epsilon = tf.placeholder(tf.float32, shape=(BATCH, nv_sub, 2))
    eng_at_x0_sub = tf.placeholder(tf.float32, shape=(BATCH, 1))
    eng_at_xnew_sub = tf.placeholder(tf.float32, shape=(BATCH, 1))
    d_n = tf.placeholder(tf.float32, shape=(BATCH, nv_sub, 2))
    alpha = tf.placeholder(tf.float32, shape=(BATCH, 1))
    edges_ss = tf.placeholder(tf.float32, shape=(BATCH, (nv_sub * (nv_sub - 1)) / 2, 4))

    return v, displacement, deng_x_at_x0_sub, eng_at_x0_sub, eng_at_xnew_sub, d_n, alpha, epsilon, edges_ss

def get_model(v, displacement, edges_ss, edges_ss_n, is_training, final_activation, bn_decay=None):
    """Generat mesh network

    :param v: variance normalized source vertices
    :param displacement: displacement of each vertex from variance normalized source to variance normalized target
    :param edges_ss: the ij vertix, to compute w_ij, specified for the source vertices
    :param edges_ss_n: number of edges
    :param is_training: need to check what it does, exactly.
    :param final_activation: what activation to put in the final layer that drives w_ij, if range to be limited this hsould be
    sigmoidal.
    :param bn_decay:  need to check what it does, exactly.
    :return: neural network that produces w_ij to be later assambled into a hessian. 
    """

    concat_source_displacement = tf.concat([v, displacement], 2)
    batch_size = v.get_shape()[0].value
    num_point = v.get_shape()[1].value
    input_image = tf.expand_dims(concat_source_displacement, -1)

    net = tf_util.conv2d(input_image, 64, [1, 4],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                       scope='dp1')

    global_feat = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')

    global_feat_expand = tf.tile(global_feat, [1, edges_ss_n, 1, 1])
    edges_expanded = tf.expand_dims(edges_ss, axis=2)
    concat_feat = tf.concat([edges_expanded, global_feat_expand], 3)

    net = tf_util.conv2d(concat_feat, 512, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv6', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 256, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv7', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv8', bn_decay=bn_decay)

    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv9', bn_decay=bn_decay)
    dim = 2
    net = tf_util.conv2d(net, dim, [1, 1],
                         padding='VALID', stride=[1, 1], activation_fn=final_activation, is_training=is_training,
                         scope='conv10')

    w_ij = tf.squeeze(net, [2])  # BxNxC

    return w_ij


def assemble_h(wx_ij_flat, wy_ij_flat):
    """

    :param wx_ij_flat: output of the matrix for x coordinates flattend for all the batches
    :param wy_ij_flat: output of the matrix for y coordinates flattend for all the batches
    :param indices_list: list of indeces to place the w_ij in the upper part of hessian (when flattend for all batches)
    :return: Hessian matrix for x and hessian matrix for y.
    """

    BATCH = 32
    nFocus = 12

    # mask for the upper trianlge part of S-by-S matrix exluding the diagonal
    ut = np.triu_indices(nFocus, k=1)

    # use upper triangular mask to obtain indeces for the upper triangular part
    # of a matrix, next will have to replicate accross the batches.
    all_indices = np.arange((nFocus * nFocus)).reshape((nFocus, nFocus))
    single_batch_indexing = all_indices[ut]

    # place holder for indexing through flattened batches.
    all_indixing_with_batch = []

    # get indexes for upper triangular parts of the hessian
    # for each batch, when the batches are flattened.
    for b in range(BATCH):
        # bor each batch the indices are shifted by hessian size i.e., by S-by-S.
        b_index = np.array(single_batch_indexing) + b * (nFocus * nFocus)
        all_indixing_with_batch.extend(b_index)

    all_indixing_with_batch = np.array(all_indixing_with_batch).astype('int32')
    indices_list = [[ind] for ind in all_indixing_with_batch]

    shape = tf.constant([BATCH * nFocus * nFocus])
    indices = tf.constant(indices_list)

    Hx_tmp = tf.scatter_nd(indices, wx_ij_flat, shape)
    Hy_tmp = tf.scatter_nd(indices, wy_ij_flat, shape)

    # reshape H_tmp from list to the tensor of BATCH x S x S size.
    # at this point the Hessian is not symmetric only its upper part is
    # populated.
    Hx_mat_ut = tf.reshape(Hx_tmp, [BATCH, nFocus, nFocus])
    Hy_mat_ut = tf.reshape(Hy_tmp, [BATCH, nFocus, nFocus])

    Hx_tmp = tf.linalg.set_diag(Hx_mat_ut, tf.zeros((BATCH, nFocus)))
    Hy_tmp = tf.linalg.set_diag(Hy_mat_ut, tf.zeros((BATCH, nFocus)))

    clean_Hx_tmp = tf.matrix_band_part(Hx_tmp, 0, -1)
    clean_Hy_tmp = tf.matrix_band_part(Hy_tmp, 0, -1)

    # stored = tf.identity(clean_Hx_tmp)
    stored = tf.identity(wx_ij_flat)

    # populate the lower part of hessian by reflecting the upper part.
    Hx_mat_full = clean_Hx_tmp + tf.transpose(clean_Hx_tmp, perm=[0, 2, 1])
    Hx_mat_full = -Hx_mat_full
    Hy_mat_full = clean_Hy_tmp + tf.transpose(clean_Hy_tmp, perm=[0, 2, 1])
    Hy_mat_full = -Hy_mat_full

    # update diagonal to have Laplacian like structure.
    rowx_sum = -tf.reduce_sum(tf.cast(Hx_mat_full, tf.float64), axis=[1])
    rowy_sum = -tf.reduce_sum(tf.cast(Hy_mat_full, tf.float64), axis=[1])
    Hx_final = tf.linalg.set_diag(tf.cast(Hx_mat_full, tf.float64), rowx_sum)
    Hy_final = tf.linalg.set_diag(tf.cast(Hy_mat_full, tf.float64), rowy_sum)

    return (Hx_final, Hy_final)


def mesh_get_loss(w_ij, deng_x_at_x0_sub, eng_at_x0_sub, eng_at_xnew_sub, d_n, alpha, epsilon, BATCH=1, nv_sub=12):
    """s    : Source and target patch size on which we test the energy (1-ring) as
               opposed to a slightly larger patch that the networks sees.
               (hessian will be "BATCH x S x S x DATA_DIM", currently data dim = 1);

        w_ij : output of all the weight in hessian. Each w_ij is driven by the same network
               replicated for each edge. So a w_ij is a function of source patch, target
               patch and an edge. There are (S *(S-1))/2 edges in patch of size S,
               leading to all w_ij being a vector of size "BATCH x (S *(S-1))/2 x DATA_DIM"";

        gard : gradient with respect to target of E_{source} @ target.
               Vector of size "BATCH x S x DATA_DIM";

        epsilon : noise added to gradient size  "BATCH x S x DATA_DIM";

        alpha : step size  "BATCH x 1";

        energy_true : Energy value  E_{source} @ (target + alpha grad + epsilon)  "BATCH x 1".

    TODO: currently support DATA_DIM = 1"""

    end_points = {}

    # update hessian with weights where the weight go to the indexes
    # specified by all_indixing_with_batch array.
    wx_ij_flat = tf.reshape(w_ij[:, :, 0], [-1])
    wy_ij_flat = tf.reshape(w_ij[:, :, 1], [-1])

    Hx_final, Hy_final = assemble_h(wx_ij_flat, wy_ij_flat)

    # compute quadratic form
    v = tf.reshape(alpha, (BATCH, 1, 1)) * d_n + epsilon

    vx = tf.reshape(v[:, :, 0], (BATCH, nv_sub, 1))
    vy = tf.reshape(v[:, :, 1], (BATCH, nv_sub, 1))

    # compute quadratic form v^top Q v
    Qx = tf.matmul(a=tf.matmul(a=vx, b=tf.cast(Hx_final, tf.float32), transpose_a=True), b=vx)
    Qx = tf.squeeze(Qx, [2])
    Qy = tf.matmul(a=tf.matmul(a=vy, b=tf.cast(Hy_final, tf.float32), transpose_a=True), b=vy)
    Qy = tf.squeeze(Qy, [2])

    # remove the linear part
    v_flat = tf.reshape(v, [BATCH, -1, 1])
    denergy_dx_flat = tf.reshape(deng_x_at_x0_sub, [BATCH, -1, 1])

    # normalized direction times alpha: v_flat gradient, but gradient is not normalized
    tmp = tf.matmul(a=v_flat, b=denergy_dx_flat, transpose_a=True)
    energy_lin_removed = eng_at_xnew_sub - (eng_at_x0_sub - tf.squeeze(tmp, [2]))

    energy_dif = energy_lin_removed - Qx - Qy

    end_points['eng_at_xnew_sub'] = eng_at_xnew_sub
    end_points['energy_dif'] = energy_dif
    end_points['Qx'] = Qx
    end_points['Qy'] = Qy
    end_points['energy_lin_removed'] = energy_lin_removed
    end_points['Hx_final'] = Hx_final
    end_points['Hy_final'] = Hy_final

    energy_dif_loss = tf.nn.l2_loss(energy_dif)
    tf.summary.scalar('energy_dif_loss', energy_dif_loss)

    return (energy_dif_loss, end_points)


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
