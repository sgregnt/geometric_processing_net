import tensorflow as tf
import numpy as np

def train():

    with tf.Graph().as_default():
        with tf.device('/cpu'):


            def Jacobian_of_transform():
                pass


            source = tf.placeholder(tf.float32, shape=(3, ))
            is_training = tf.placeholder(tf.bool, shape=())

            w_ij = source

            W_flat = tf.Variable(initial_value=[0] * (2 * 2), dtype = 'float32')

            indices = tf.constant([[0], [1], [3]])
            updates = tf.constant([1])
            shape = tf.constant([4])
            W_flat = tf.scatter_nd(indices, w_ij, shape)

            g = tf.gradients(W_flat * W_flat, [source])

            #
            # indices = tf.constant([[1]])
            # updates = tf.constant([9, 10, 11, 12])
            # shape = tf.constant([4])
            # a = tf.scatter_nd(indices, w_ij, shape)

            # with tf.Session() as sess:
            #     print(sess.run(scatter))

            # tf.tensor_scatter_nd_update(W_flat, [1,0], (w_ij, w_ij))
            # b = W_flat[1].assign(w_ij)
            # b = W_flat[2].assign(w_ij) + b
            # b = W_flat[3].assign(w_ij) + b



            # b = a
            # c = a
            # W_flat[2].assign(4)
            # tf.assign(W_flat[1], w_ij)
            # tf.assign(W_flat[1], 4)
            # tf.assign(W_flat[2], w_ij)
            # tf.assign(W_flat[2], 4)
            # W_flat = (w_ij[0,0], w_ij[0,1], w_ij[1,0], w_ij[0,0])
            W = tf.reshape(W_flat, (2, 2))
            s, u, v = tf.linalg.svd(
                # tf.matmul(W, tf.constant(np.array([[3, 0], [1,0]]).astype('float32'))),
                W,
                full_matrices=True,
                compute_uv=True,
                name=None
            )
            a = tf.transpose(u)
            b = tf.diag(1/s)
            c = tf.matmul(b, a)
            d = tf.matmul(v, c)

            e = tf.matmul(d, W)
            f = tf.matmul(u, tf.matmul(tf.diag(s), tf.transpose(v)))
            g2 = tf.gradients(d[0], source)

            # this is summed over all singular values. So I have \sum(ds_dsource[1]) over s[0] and s[1]
            m0 = tf.hessians(s[0], source)
            m1 = tf.gradients(s[1], source)
            q = tf.gradients(m0[0][0], source)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init, {is_training: True})

        ops = {'W_flat': W_flat,
               'source' : source,
               'w_ij' : w_ij,
               'W' : W,
               'g' : g,
               'm0' : m0,
               'm1' : m1,
               's' : s,
               'q' : q,
               'a' : a,
               'b' : b,
               'c' : c,
               'd' : d,
               'e' : e,
               'f' : f,
               'g2' : g2,
               }

        for epoch in range(3):
            feed_dict = {ops['source']: [1, 2, 3]}
            res_W_flat, res_wij, res_W, cc, gg, mm0,  mm1, qq, ss, aa, bb, cc, dd, ee, ff, gg2 = sess.run([ops['W_flat'], ops['w_ij'], ops['W'],  ops['c'], ops['g'], ops['m0'], ops['m1'], ops['q'], ops['s'], ops['a'], ops['b'], ops['c'], ops['d'], ops['e'], ops['f'], ops['g2']], feed_dict=feed_dict)
            print("epoch:" ,  epoch)
            print("W_flat:", res_W_flat)
            print("wij:", res_wij)
            print("W:", res_W)
            print('g', gg)
            print('m0', mm0)
            print('m1', mm1)
            print('q', qq)
            print('s', ss)
            print('a', aa)
            print('b', bb)
            print('c', cc)
            print('d', dd)
            print('e', ee)
            print('f', ff)
            print('g2', gg2)
        1/0
if __name__ == "__main__" :
    train()