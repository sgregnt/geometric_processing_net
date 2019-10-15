import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_loss_evolution(loss_array, title):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(loss_array)
    plt.title(title)
    plt.show()


def log_arrays(d_energy_at_new_fV, d_energy_at_fV, d_energy_lin_removed, d_energy_dif, d_Qx, d_Qy):
    print("d_energy_at_new_fV", d_energy_at_new_fV)
    print("d_energy_at_fV", d_energy_at_fV)
    print("d_energy_lin_removed", d_energy_lin_removed)
    print("d_energy_lin_removed.shape", d_energy_lin_removed.shape)
    print("d_energy_dif", d_energy_dif)
    print("d_energy_dif.shape", d_energy_dif.shape)
    print("d_Qx", d_Qx)
    print("d_Qy", d_Qy)

def get_all_inputs(current_inputs, old=True):

    if old:

        (all_v,
         all_t,
         all_vn,
         all_tn,
         all_displacements,
         all_eng_at_x0_sub,
         all_eng_at_xnew_sub,
         all_deng_x_at_x0_sub,
         all_d_n,
         all_alpha,
         all_edges_ss,
         all_epsilon,
         all_nv_sub,
         all_edges_ss_n) =  (current_inputs['source'],
                            current_inputs['T'],
                            current_inputs['nV'],
                            current_inputs['nT'],
                            current_inputs['displacement'],
                            current_inputs['energy_at_fV'],
                            current_inputs['energy_at_new_fV'],
                            current_inputs['denergy_dx'],
                            current_inputs['ds'],
                            current_inputs['alpha'],
                            current_inputs['edges'],
                            current_inputs['epsilon'],
                            current_inputs['nFocus'],
                            current_inputs['en'])
    else:

        (all_v,
         all_t,
         all_vn,
         all_tn,
         all_displacements,
         all_eng_at_x0_sub,
         all_eng_at_xnew_sub,
         all_deng_x_at_x0_sub,
         all_d_n,
         all_alpha,
         all_edges_ss,
         all_epsilon,
         all_nv_sub,
         all_edges_ss_n) = (current_inputs['all_v'],
                        current_inputs['all_t'],
                        current_inputs['all_vn'],
                        current_inputs['all_tn'],
                        current_inputs['all_displacements'],
                        current_inputs['all_eng_at_x0_sub'],
                        current_inputs['all_eng_at_xnew_sub'],
                        current_inputs['all_deng_x_at_x0_sub'],
                        current_inputs['all_d_n'],
                        current_inputs['all_alpha'],
                        current_inputs['all_edges_ss'],
                        current_inputs['all_epsilon'],
                        current_inputs['all_nv_sub'],
                        current_inputs['all_edges_ss_n'])

    return     (all_v,
                 all_t,
                 all_vn,
                 all_tn,
                 all_displacements,
                 all_eng_at_x0_sub,
                 all_eng_at_xnew_sub,
                 all_deng_x_at_x0_sub,
                 all_d_n,
                 all_alpha,
                 all_edges_ss,
                 all_epsilon,
                 all_nv_sub,
                 all_edges_ss_n)

def unpack_inputs(all_inputs, start_idx, end_idx):
    """current_inputs is array of tensors with input data"""

    (all_v,
     all_t,
     all_vn,
     all_tn,
     all_displacements,
     all_eng_at_x0_sub,
     all_eng_at_xnew_sub,
     all_deng_x_at_x0_sub,
     all_d_n,
     all_alpha,
     all_edges_ss,
     all_epsilon,
     all_nv_sub,
     all_edges_ss_n) = all_inputs

    (v_b,
     t_b,
     vn_b,
     tn_b,
     displacement_b,
     eng_at_x0_sub_b,
     eng_at_xnew_sub_b,
     deng_x_at_x0_sub_b,
     d_n_b,
     alpha_b,
     edges_ss_b,
     epsilon_b,
     nv_sub_b,
     edges_ss_n_b) = (np.array(all_v[start_idx:end_idx]),
                      np.array(all_t[start_idx:end_idx]),
                      np.array(all_vn[start_idx:end_idx]),
                      np.array(all_tn[start_idx:end_idx]),
                      np.array(all_displacements[start_idx:end_idx]),
                      np.array(all_eng_at_x0_sub[start_idx:end_idx]),
                      np.array(all_eng_at_xnew_sub[start_idx:end_idx]),
                      np.array(all_deng_x_at_x0_sub[start_idx:end_idx]),
                      np.array(all_d_n[start_idx:end_idx]),
                      np.array(all_alpha[start_idx:end_idx]),
                      np.array(all_edges_ss[start_idx:end_idx]),
                      np.array(all_epsilon[start_idx:end_idx]),
                      np.array(all_nv_sub[start_idx:end_idx]),
                      np.array(all_edges_ss_n[start_idx:end_idx]))

    return (v_b,
     t_b,
     vn_b,
     tn_b,
     displacement_b,
     eng_at_x0_sub_b,
     eng_at_xnew_sub_b,
     deng_x_at_x0_sub_b,
     d_n_b,
     alpha_b,
     edges_ss_b,
     epsilon_b,
     nv_sub_b,
     edges_ss_n_b)

def gen_feed_dict(inputs_b,  is_training, ops):

    (v_b,
     t_b,
     vn_b,
     tn_b,
     displacement_b,
     eng_at_x0_sub_b,
     eng_at_xnew_sub_b,
     deng_x_at_x0_sub_b,
     d_n_b,
     alpha_b,
     edges_ss_b,
     epsilon_b,
     nv_sub_b,
     edges_ss_n_b) = inputs_b

    feed_dict = {ops['v']               : v_b,
                 ops['displacement']    : displacement_b,
                 ops['eng_at_x0_sub']   : eng_at_x0_sub_b,
                 ops['eng_at_xnew_sub'] : eng_at_xnew_sub_b,
                 ops['deng_x_at_x0_sub']: deng_x_at_x0_sub_b,
                 ops['epsilon']         : epsilon_b,
                 ops['edges_ss']           : edges_ss_b,
                 ops['d_n']             : d_n_b,
                 ops['alpha']           : alpha_b,
                 ops['is_training']     : is_training}

    return feed_dict

def gen_feed_dict_inference(inputs_b,  is_training, ops):

    (v_b,
     # t_b,
     # vn_b,
     # tn_b,
     displacement_b,
     # eng_at_x0_sub_b,
     # eng_at_xnew_sub_b,
     # deng_x_at_x0_sub_b,
     # d_n_b,
     # alpha_b,
     edges_ss_b) = inputs_b
     # epsilon_b) = inputs_b
     # nv_sub_b,
     # edges_ss_n_b) = inputs_b

    feed_dict = {ops['v']               : v_b,
                 ops['displacement']    : displacement_b,
                 # ops['eng_at_x0_sub']   : eng_at_x0_sub_b,
                 # ops['eng_at_xnew_sub'] : eng_at_xnew_sub_b,
                 # ops['deng_x_at_x0_sub']: deng_x_at_x0_sub_b,
                 # ops['epsilon']         : epsilon_b,
                 ops['edges_ss']           : edges_ss_b,
                 # ops['d_n']             : d_n_b,
                 # ops['alpha']           : alpha_b,
                 ops['is_training']     : is_training}

    return feed_dict

def run_sess(sess, feed_dict, ops):
    all_tensors = sess.run([ops['merged'],
                           ops['step'],
                           ops['train_op'],
                           ops['loss'],
                           ops['d_eng_at_xnew_sub'],
                           ops['d_energy_lin_removed'],
                           ops['d_energy_dif'],
                           ops['d_Qx'],
                           ops['d_Qy'],
                           ops['Hx_final'],
                           ops['Hy_final']],
                           feed_dict=feed_dict)

    return all_tensors