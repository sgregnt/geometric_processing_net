import numpy as np
import scipy.io
import os
from itertools import combinations
from utils.debug_and_plot_utils import hist_array_scalars
from utils.energy_utils import  do_verbose, dirichlet_sym_gd_bfgs_nn_weak_armijo_optimizer_step
from utils.mesh_utils import mesh_params, input_prep, read_patch, mesh_unpack
from utils.dirichlet_sym_utils import res_unpack
from utils.optimization_utils import step_unpack, approx_f_xnew_first_order, log, do_optimization_result_unpack, line_search_armijio

def gen_single_sample_sub(patch, line_search, do_show=True, verbose=True):
    """generate single sample based on a mesh from file
    patch, should include the large patch and subpatch

    To prepare data I need to consider the full patch. The source is the full patch (s),
    then I have target which is also the full patch. Next, I add some noice to the target
    this gives me the new target (t), then I compute gradient of the energy
    with resepct to the whole patch this gives me the d vector. I truncate this d vector
    to the subpatch, and I compute the energy of the subpatch at the E_ss(tt) and then at the
    targe E_ss(tt+d).

    The data needed for training:
    # (1) - the true energy value eng_at_xnew i.e., E_s(t+d)
    # (2) - the energy value at E_s(t)
    # (3) - the gradient at E_s(t)
    # (4) - value of d (the change between eng_at_xnew and t). D is constructed at run time.
    # (5) - value of displacement encoding of s and x.
    """

    # Prepare input, randomness in fv is achieved by adding a noise
    v, t, fv, subt = input_prep(patch, do_show=True, avoid_triangle_flip=True)

    # run line search and pick good alpha that satisfies Armijo conditions
    params = dirichlet_sym_gd_bfgs_nn_weak_armijo_optimizer_step(line_search=line_search,
                                                                 get_nn_hessians=None,
                                                                 source=v,
                                                                 t=t,
                                                                 target=fv,
                                                                 subt=subt,
                                                                 verbose=verbose,
                                                                 type='GD')

    res = do_optimization_result_unpack(params=params, t=t, v=v, verbose=verbose)
    params = (res['source'],  # v[:, 0:2],
              res['t'],  # T,
              res['source_dim'],  # v.shape[0],
              res['t_dim'],  # T.shape[0],
              res['displacement'],  # displacement,
              res['eng_at_x0_sub'],  # eng_at_x0_sub,
              res['eng_at_xnew'],  # eng_at_xnew,
              res['eng_at_xnew_sub'],  # eng_at_xnew_sub,
              res['deng_x_at_x0_sub'],  # deng_x_at_x0_sub[subvertices, 0:2],
              res['p_k'][res['subvertices'], 0:2],  # d_n, this is ugly.
              res['alpha'],  # alpha,
              res['edges_sub'],  # edges_ss,
              res['epsilon'],  # epsilon[subvertices, 0:2],
              res['sub_source_dim'],  # len(subvertices),
              res['edges_sub_dim'],  # edges_ss.shape[0],
              res['sv_at_x0'],  # sv_at_x0,
              res['sv_at_x0_sub'],  # sv_at_x0_sub,
              res['sv_at_xnew'],  # sv_at_xnew,
              res['sv_at_xnew_sub'],  # sv_at_xnew_sub,
              res['xnew'],  # fv_new,
              res['eng_first_order_approx_xnew_x0_diff'],  # linear_diff,
              res['t_flip'],  # detT,
              res['eng_xnew_x0_diff_sub'],  # sub_dif,
              res['x0'],  # fv,
              res['eng_first_order_approx_xnew_x0_diff_sub'])  # sub_linear_diff

    return params

def gen_dataset_sub(file, N):
    """ Generate data set of N samples, with mesh taken from file"""

    all_source = []
    all_t = []
    all_source_dim = []
    all_t_dim = []
    all_displacements = []
    all_eng_at_x0_sub = []
    all_eng_at_xnew_sub = []
    all_deng_x_at_x0_sub = []
    all_step_n = []
    all_alpha = []
    all_edges_sub = []
    all_epsilon = []
    all_sub_source_dim = []
    all_edges_sub_dim = []
    all_sv_at_x0 = []
    all_sv_at_xnew = []
    all_t_flip = []
    all_eng_xnew_x0_diff_sub = []
    all_eng_first_order_approx_xnew_x0_diff_sub = []

    verbose = True
    sub_patch = read_patch(file)
    for i in range(N):
        params = gen_single_sample_sub(line_search=line_search_armijio, patch=sub_patch, do_show=False, verbose=True)

        (source,
         t,
         source_dim,
         t_dim,
         displacement,
         eng_at_x0_sub,
         eng_at_xnew,
         eng_at_xnew_sub,
         deng_x_at_x0_sub,
         step_n,
         alpha,
         edges_sub,
         epsilon,
         sub_source_dim,
         edges_sub_dim,
         sv_at_x0,
         sv_at_x0_sub,
         sv_at_xnew,
         sv_at_xnew_sub,
         fv_at_xnew,
         eng_first_order_approx_xnew_x0_diff,
         t_flip,
         eng_xnew_x0_diff_sub,
         fv_at_x0,
         eng_first_order_approx_xnew_x0_diff_sub) = params

        do_verbose("eng_first_order_approx_xnew_x0_diff", eng_first_order_approx_xnew_x0_diff, verbose)

        if t_flip:
            do_verbose("Triangle flipped", t_flip, True)
            print("!+!+!" * 100)
            pass
        else:
            if (((eng_xnew_x0_diff_sub > 20) or (eng_xnew_x0_diff_sub < 0.2)) or ((eng_first_order_approx_xnew_x0_diff_sub < 0.5) or (eng_first_order_approx_xnew_x0_diff_sub > 15))):
                do_verbose("eng_first_order_approx_xnew_x0_diff_sub", eng_first_order_approx_xnew_x0_diff_sub, True)
                do_verbose("Something bad happened, eng_xnew_x0_diff_sub", eng_xnew_x0_diff_sub, True)
                print("-!-!-!" * 100)
                pass
            else:
                all_source.append(source)
                all_t.append(t)
                all_source_dim.append(source_dim)
                all_t_dim.append(t_dim)
                all_displacements.append(displacement)
                all_eng_at_x0_sub.append([eng_at_x0_sub])
                all_eng_at_xnew_sub.append([eng_at_xnew_sub])
                all_deng_x_at_x0_sub.append(deng_x_at_x0_sub)
                all_step_n.append(step_n)
                all_alpha.append([alpha])
                all_edges_sub.append(edges_sub)
                all_epsilon.append(epsilon)
                all_sub_source_dim.append(sub_source_dim)
                all_edges_sub_dim.append(edges_sub_dim)
                all_sv_at_x0.append(sv_at_x0)
                all_sv_at_xnew.append(sv_at_xnew)
                all_t_flip.append(t_flip)
                all_eng_xnew_x0_diff_sub.append(eng_xnew_x0_diff_sub)
                all_eng_first_order_approx_xnew_x0_diff_sub.append(eng_first_order_approx_xnew_x0_diff_sub)

            print("-----------------------", i)

    params = (all_source,
              all_t,
              all_source_dim,
              all_t_dim,
              all_displacements,
              all_eng_at_x0_sub,
              all_eng_at_xnew_sub,
              all_deng_x_at_x0_sub,
              all_step_n,
              all_alpha,
              all_edges_sub,
              all_epsilon,
              all_sub_source_dim,
              all_edges_sub_dim,
              all_sv_at_x0,
              all_sv_at_xnew,
              all_t_flip,
              all_eng_xnew_x0_diff_sub,
              all_eng_first_order_approx_xnew_x0_diff_sub)

    return params

if __name__ == "__main__":

    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.')
    dataset = gen_dataset_sub(os.path.join(root_path, './data/mesh_examples/grind_and_subgrid.mat'), N=100)

    (all_source,
     all_t,
     all_source_dim,
     all_t_dim,
     all_displacements,
     all_eng_at_x0_sub,
     all_eng_at_xnew_sub,
     all_deng_x_at_x0_sub,
     all_step_n,
     all_alpha,
     all_edges_sub,
     all_epsilon,
     all_sub_source_dim,
     all_edges_sub_dim,
     all_sv_at_x0,
     all_sv_at_xnew,
     all_t_flip,
     all_eng_xnew_x0_diff_sub,
     all_eng_first_order_approx_xnew_x0_diff_sub) = dataset

    (source, Ts, nVs, nTs, displacements, energy_at_fVs_sub, energy_at_new_fVs_sub, denergy_dx, ds, alphas,
     edges, epsilons, nFocus, en, fV_svs, new_fV_svs, detTs, sub_difs, sub_linear_diffs) = dataset

    dataset = {'all_v': all_source,
               'all_t': all_t,
               'all_vn': all_source_dim,
               'all_tn': all_t_dim,
               'all_displacements': all_displacements,
               'all_eng_at_x0_sub': all_eng_at_x0_sub,
               'all_eng_at_xnew_sub': all_eng_at_xnew_sub,
               'all_deng_x_at_x0_sub': all_deng_x_at_x0_sub,
               'all_d_n': all_step_n,
               'all_alpha': all_alpha,
               'all_edges_ss': all_edges_sub,
               'all_epsilon': all_epsilon,
               'all_nv_sub': all_sub_source_dim,
               'all_edges_ss_n': all_edges_sub_dim}

    hist_array_scalars(sub_difs, 'energy at sub_diffs')
    hist_array_scalars(sub_linear_diffs, 'energy at sub_linear_diffs')

    scipy.io.savemat(os.path.join(root_path, './data/datasets/test_after_cleaning.mat'), dataset)
    read_dataset = scipy.io.loadmat(os.path.join(root_path, './data/datasets/test_after_cleaning.mat'))
    print("dataset_length", len(Ts))
    print("dataset_length", len(read_dataset['all_alpha']))
    print("A small verification that the dataset was saved - reading back nVs:", read_dataset['all_vn'])
    print("Done.")