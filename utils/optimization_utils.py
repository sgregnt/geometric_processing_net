import numpy as np
import sys
sys.path.insert(0, './utils')
from utils.mesh_utils import check_triangle_flip, show, mesh_unpack
from utils.dirichlet_sym_utils import res_unpack
from itertools import combinations
from utils.debug_and_plot_utils import do_verbose

def BFGS(y, s, inv_B):
    """ Calculate inverse Hessian for BFGS algorithm"""

    # I need to flatten here the vectors from a matrix form to actual vectors.
    s = s.flatten()
    y = y.flatten()
    ys = np.inner(y, s)
    inv_By = np.dot(inv_B, y)
    yinv_By = np.inner(y, inv_By)
    inv_B_new = inv_B + (ys + yinv_By) * np.outer(s, s) / ys ** 2
    inv_B_new -= (np.outer(inv_By, s) + np.outer(s, inv_By)) / ys
    return inv_B_new


def get_nn_direction(Hx, Hy, subvertices, deng_x_at_x0):

    Hx_inv = np.linalg.pinv(Hx)
    Hy_inv = np.linalg.pinv(Hy)

    # NOTE: very important: I think I didn;t teach neural network
    # correctly, should have used alpha p_k, instead used alpha * p_k_n?
    # this I have to verify later.

    corr = deng_x_at_x0 / np.linalg.norm(deng_x_at_x0)

    Hx_inv_dir = np.dot(Hx_inv, -corr[subvertices, 0])
    Hy_inv_dir = np.dot(Hy_inv, -corr[subvertices, 1])

    H_inv_dir = -corr.copy()
    H_inv_dir[subvertices, 0] = Hx_inv_dir
    H_inv_dir[subvertices, 1] = Hy_inv_dir

    return H_inv_dir


def step_unpack(step_dic):

    return(step_dic['x0'],
           step_dic['xnew'],
           step_dic['alpha'],
           step_dic['p_k'],
           step_dic['p_k_n'],
           step_dic['type'])


def line_search_armijio(init_alpha, p_k, mesh, mesh_sub, all_at_x0,
                        compute_all_from_x, compute_all_from_x_sub, c1=0.001, c2=0.9, verbose=False,
                        type='GD'):

    """ Run line search along 'deng_x_at_x0_n' direction until Armijo conditions are met."""

    (_, _, _, _, _, _, _, _, _, _, sls, t, _) = mesh_unpack(mesh)
    (_, _, _, _, _, _, _, _, _, _, slssub, subt, subvertices) = mesh_unpack(mesh_sub)

    alpha = init_alpha
    look_for_armijio_flag = True

    (eng_at_x0,
     deng_sv_at_x0,
     deng_x_at_x0,
     deng_x_at_x0_n,
     sv_at_x0,
     dsv_x_at_x0,
     fv_at_x0) = res_unpack(all_at_x0)

    # Cycle until Armijo conditions are met.
    while look_for_armijio_flag:

        alpha = alpha * 0.9

        do_verbose('alpha', alpha, verbose)
        step_fv = alpha * p_k
        fv_at_xnew = fv_at_x0 + step_fv

        if check_triangle_flip(fv_at_xnew, t, verbose=True):
            show(fv_at_x0, t, fv_at_xnew, False)
            continue

        all_at_xnew = compute_all_from_x((fv_at_xnew, t, sls))
        all_at_xnew_sub = compute_all_from_x_sub((fv_at_xnew, subt, slssub))

        # Objective and gradients at x0
        (eng_at_xnew,
         deng_sv_at_xnew,
         deng_x_at_xnew,
         deng_x_at_xnew_n,
         sv_at_xnew,
         dsv_x_at_xnew,
         fv_at_xnew) = res_unpack(all_at_xnew)

        # check armijio weak condition
        look_for_armijio_flag = check_armijo(eng_at_x0=eng_at_x0,
                                             c1=c1,
                                             deng_x_at_x0=deng_x_at_x0,
                                             p_k=p_k,
                                             alpha=alpha,
                                             eng_at_xnew=eng_at_xnew,
                                             deng_x_at_xnew=deng_x_at_xnew,
                                             c2=c2)

    p_k_n = p_k / np.linalg.norm(p_k)
    step = {'x0' : fv_at_x0,
            'xnew' : fv_at_xnew,
            'alpha' : alpha,
            'p_k' : p_k,
            'p_k_n' : p_k_n,
            'type' : type}

    return step, all_at_xnew, all_at_xnew_sub

def check_armijo(eng_at_x0, c1, deng_x_at_x0, p_k, alpha, eng_at_xnew, deng_x_at_xnew, c2):
    """ Check if Armijo conditions are satisfied"""

    d = alpha * p_k
    # Sufficient decrease
    # f(x_k  + alpha p_k)  =< f(x_k) + c_1  alpha df(x_k) p_k
    if ( eng_at_xnew < eng_at_x0 + c1 * np.matmul(d.flatten(), deng_x_at_x0.flatten())):
        print("sufficient decrease", eng_at_xnew, eng_at_x0 + c1 * np.matmul(d.flatten(), deng_x_at_x0.flatten()))
    else:
        print("INsufficient decrease", eng_at_xnew, eng_at_x0 + c1 * np.matmul(d.flatten(), deng_x_at_x0.flatten()))
        return True

    # Curvature condition
    # df(x_k + alpha p_k) p_k >= c_2 df(x_k) p_k
    if (np.matmul(deng_x_at_xnew.flatten(), d.flatten()) > c2 * np.matmul(d.flatten(), deng_x_at_x0.flatten())):
        print("curvature condition met",
              np.matmul(deng_x_at_xnew.flatten(),
                        d.flatten()),
              c2 * np.matmul(d.flatten(),
                             deng_x_at_x0.flatten()))
    else:
        print("curvature condition was NOT met",
              np.matmul(deng_x_at_xnew.flatten(),
              d.flatten()),
              c2 * np.matmul(deng_x_at_x0.flatten(),
                             d.flatten()))
        return True

    strong_curvature = False
    if strong_curvature:
        if np.abs(np.matmul(deng_x_at_xnew.flatten(),
                            d.flatten())) > np.abs(c2 * np.matmul(d.flatten(), deng_x_at_x0.flatten())):

            print("strong curvature condition met",
                  np.matmul(deng_x_at_xnew.flatten(), d.flatten()),
                  c2 * np.matmul(d.flatten(), deng_x_at_x0.flatten()))
        else:
            print("strong curvature condition was NOT met",
                   np.matmul(deng_x_at_xnew.flatten(), d.flatten()),
                  c2 * np.matmul(deng_x_at_x0.flatten(), d.flatten()))
            return True

    return False

def log(alpha, eng_xnew_x0_diff, eng_at_x0, eng_at_x0_sub, eng_at_xnew, eng_at_xnew_sub, linpart, first_order_approx_xnew_x0_diff_sub, eng_xnew_x0_diff_sub, verbose):

    do_verbose('alpha', alpha, verbose)
    do_verbose('eng_at_xnew', eng_at_xnew, verbose)
    do_verbose('eng_at_x0', eng_at_x0, verbose)
    do_verbose('eng_at_xnew_sub', eng_at_xnew_sub, verbose)
    do_verbose('eng_at_x0_sub', eng_at_x0_sub, verbose)
    do_verbose('eng_xnew_x0_diff',eng_xnew_x0_diff, verbose)
    do_verbose('eng_at_x0_sub - eng_at_xnew_sub', eng_at_x0_sub - eng_at_xnew_sub, verbose)
    do_verbose('eng_at_xnew - linpart', eng_at_xnew - linpart, verbose)
    do_verbose('first_order_approx_xnew_x0_diff_sub', first_order_approx_xnew_x0_diff_sub, verbose)
    do_verbose('eng_xnew_x0_diff_sub', eng_xnew_x0_diff_sub, verbose)

def optim_res_unpack(optimization_res):
    return (optimization_res['displacement'],
    optimization_res['p_k'],
    optimization_res['p_k_n'],
    optimization_res['subvertices'],
    optimization_res['eng_xnew_first_order_approx'],
    optimization_res['eng_xnew_x0_diff_sub'],
    optimization_res['source'],
    optimization_res['t'],
    optimization_res['source_dim'],
    optimization_res['t_dim'],
    optimization_res['eng_at_x0_sub'],
    optimization_res['eng_at_xnew'],
    optimization_res['eng_at_xnew_sub'],
    optimization_res['deng_x_at_x0_sub'],
    optimization_res['alpha'],
    optimization_res['edges_sub'],
    optimization_res['epsilon'],
    optimization_res['sub_source_dim'],
    optimization_res['edges_sub_dim'],
    optimization_res['sv_at_x0'],
    optimization_res['sv_at_x0_sub'],
    optimization_res['sv_at_xnew'],
    optimization_res['sv_at_xnew_sub'],
    optimization_res['xnew'],
    optimization_res['eng_first_order_approx_xnew_x0_diff'],
    optimization_res['t_flip'],
    optimization_res['x0'],
    optimization_res['eng_first_order_approx_xnew_x0_diff_sub'],
    optimization_res['ray_deng_x_at_x0'],
    optimization_res['ray_deng_x_at_xnew'],
    optimization_res['eng_plot'],
    optimization_res['ray_deng_x_at_x0_sub'],
    optimization_res['ray_deng_x_at_xnew_sub'],
    optimization_res['eng_plot_sub'],
    optimization_res['span'],
    optimization_res['type'],
    optimization_res['B_inv_new'],
    optimization_res['eng_xnew_x0_diff'])

def approx_f_xnew_first_order(f_x0, df_x0, step):
    return (f_x0 + np.matmul(step.flatten(), df_x0.flatten()))

def get_edges(fv, subvertices):

    # The edges used for Hessian prediction of the mesh_sub are only edges of the mesh_sub.
    # Reshape to have all the edegs in one row (4, since we look at 2d problem)
    # leave out only two coordinates since z == 0.
    edges_ss = np.array(list(combinations(fv[subvertices, ::], 2)))
    edges_ss = edges_ss[:, :, 0:2].reshape((edges_ss.shape[0], 4))

    return edges_ss

def get_p_k_direction(B_inv_new, type, mesh_sub, Hx, Hy, all_at_x0):

    deng_x_at_x0 = all_at_x0['deng_x_at_x']
    deng_x_at_x0_n = all_at_x0['deng_x_at_x_n']

    if type == 'BFGS':

        # Initialize BFGS descent direction on the first run
        if B_inv_new is None:
            p_k = -deng_x_at_x0
        else:
            p_k = np.dot(B_inv_new, -deng_x_at_x0.flatten()).reshape(-1, 3)

        init_alpha = 1

    elif type == 'NN':
        init_alpha = 1
        # NOTE: very important it used to be deng_x_at_x0_n instead of deng_x_at_x0
        # I fixed it to deng_x_at_x0, for the correct Hessian this  gives the
        # minimum of quadratic approximation.
        p_k = get_nn_direction(Hx, Hy, mesh_sub['vertices'], deng_x_at_x0)

    elif type == 'GD':
        init_alpha = 1
        # NOTE: very important should I use gradient direction or absolute value of gradient?
        p_k = -deng_x_at_x0_n

    elif type == 'RAND':
        p_k = 0 * all_at_x0['deng_x_at_x']
        p_k[:, 0:2] = 0.1 * np.random.randn(p_k.shape[0], 2)
        init_alpha = 1

    else:
        print("Wrong optimization type", type, "should be GD, BFGS or NN")
        1 / 0  # if wrong optimization type, break!

    return p_k, init_alpha

def do_optimization_result_unpack(params, v, t, verbose=True):
    optimization_res = {}

    (all_at_x0,
     all_at_xnew,
     all_at_x0_sub,
     all_at_xnew_sub,
     mesh,
     mesh_sub,
     plot_params,
     step,
     B_inv_new) = params

    (ray_deng_x_at_x0,
     ray_deng_x_at_xnew,
     eng_plot,
     ray_deng_x_at_x0_sub,
     ray_deng_x_at_xnew_sub,
     eng_plot_sub,
     span) = plot_params

    (_, _, alpha, p_k, p_k_n, type) = step_unpack(step)

    tmp = np.cross(mesh['A'], mesh['C'])[:, -1]
    t_flip = sum(tmp > 0) > 0

    (eng_at_x0,
     deng_sv_at_x0,
     deng_x_at_x0,
     deng_x_at_x0_n,
     sv_at_x0,
     dsv_x_at_x0,
     fv_at_x0) = res_unpack(all_at_x0)

    (eng_at_xnew,
     deng_sv_at_xnew,
     deng_x_at_xnew,
     deng_x_at_xnew_n,
     sv_at_xnew,
     dsv_x_at_xnew,
     fv_at_xnew) = res_unpack(all_at_xnew)

    (eng_at_x0_sub,
     deng_sv_at_x0_sub,
     deng_x_at_x0_sub,
     deng_x_at_x0_sub_n,
     sv_at_x0_sub,
     dsv_x_at_x0_sub,
     _) = res_unpack(all_at_x0_sub)

    (eng_at_xnew_sub,
     deng_sv_at_xnew_sub,
     deng_x_at_xnew_sub,
     deng_x_at_xnew_sub_n,
     sv_at_xnew_sub,
     dsv_x_at_xnew_sub,
     _) = res_unpack(all_at_xnew_sub)

    eng_xnew_first_order_approx = approx_f_xnew_first_order(f_x0=eng_at_x0, df_x0=deng_x_at_x0, step=alpha * p_k)
    eng_first_order_approx_xnew_x0_diff = eng_at_xnew - eng_xnew_first_order_approx
    eng_xnew_x0_diff = eng_at_xnew - eng_at_x0

    # A few notes:
    # (1) This difference is expected to be positive, due to strong Wolf
    #     conditions.
    #
    # (2) The direction of descent is set by the gradient on the whole
    #     mesh therefore one uses "step=step_fv", and we do first order
    #     expansion of eng(fv[sub-vertices] + alpha * step_fv[sub-vertices]).
    #
    # (3) No need to select sub-vertices, since deng_x_at_x0_sub has zeros at
    #     the vertices that do not appear in the mesh_sub.
    #
    eng_xnew_first_order_approx_sub = approx_f_xnew_first_order(f_x0=eng_at_x0_sub, df_x0=deng_x_at_x0_sub,
                                                                step=alpha * p_k)

    eng_first_order_approx_xnew_x0_diff_sub = eng_at_xnew_sub - eng_xnew_first_order_approx_sub
    eng_xnew_x0_diff_sub = eng_at_x0_sub - eng_at_xnew_sub

    # NOTE: important, should this be v or fv? I currently think this should be
    # fixed to V, since we can ask about vertices that are not present,
    # it makes sense then to refer to the source, and make the weights depend only
    # on the displacement not the actual values.
    #
    # edges_sub = get_edges(fv=fv_at_x0, subvertices=mesh_sub['vertices'])
    edges_sub = get_edges(fv=v, subvertices=mesh_sub['vertices'])

    log(alpha,
        eng_xnew_x0_diff,
        eng_at_x0,
        eng_at_x0_sub,
        eng_at_xnew,
        eng_at_xnew_sub,
        eng_xnew_first_order_approx,
        eng_first_order_approx_xnew_x0_diff_sub,
        eng_xnew_x0_diff_sub,
        verbose)

    optimization_res['displacement'] = fv_at_x0[:, 0:2] - v[:, 0:2]
    optimization_res['p_k'] = p_k
    optimization_res['p_k_n'] = p_k_n
    optimization_res['subvertices'] = mesh_sub['vertices']
    optimization_res['eng_xnew_first_order_approx'] = eng_xnew_first_order_approx
    optimization_res['eng_xnew_x0_diff_sub'] = eng_xnew_x0_diff_sub
    optimization_res['source'] = v[:, 0:2]
    optimization_res['t'] = t
    optimization_res['source_dim'] = v.shape[0]
    optimization_res['t_dim'] = t.shape[0]
    optimization_res['eng_at_x0_sub'] = eng_at_x0_sub
    optimization_res['eng_at_xnew'] = eng_at_xnew
    optimization_res['eng_at_xnew_sub'] = eng_at_xnew_sub
    optimization_res['deng_x_at_x0_sub'] = deng_x_at_x0_sub[mesh_sub['vertices'], 0:2]
    optimization_res['alpha'] = alpha
    optimization_res['edges_sub'] = edges_sub
    optimization_res['epsilon'] = deng_x_at_x0_sub[mesh_sub['vertices'], 0:2] * 0
    optimization_res['sub_source_dim'] = len(mesh_sub['vertices'])
    optimization_res['edges_sub_dim'] = edges_sub.shape[0]
    optimization_res['sv_at_x0'] = sv_at_x0
    optimization_res['sv_at_x0_sub'] = sv_at_x0_sub
    optimization_res['sv_at_xnew'] = sv_at_xnew
    optimization_res['sv_at_xnew_sub'] = sv_at_xnew_sub
    optimization_res['xnew'] = fv_at_xnew # this is repeated twice
    optimization_res['eng_first_order_approx_xnew_x0_diff'] = eng_first_order_approx_xnew_x0_diff
    optimization_res['t_flip'] = t_flip
    optimization_res['x0'] = fv_at_x0
    optimization_res['eng_first_order_approx_xnew_x0_diff_sub'] = eng_first_order_approx_xnew_x0_diff_sub
    optimization_res['ray_deng_x_at_x0'] = ray_deng_x_at_x0
    optimization_res['ray_deng_x_at_xnew'] = ray_deng_x_at_xnew
    optimization_res['eng_plot'] = eng_plot
    optimization_res['ray_deng_x_at_x0_sub'] = ray_deng_x_at_x0_sub
    optimization_res['ray_deng_x_at_xnew_sub'] = ray_deng_x_at_xnew_sub
    optimization_res['eng_plot_sub'] = eng_plot_sub
    optimization_res['span'] = span
    optimization_res['type'] = type
    optimization_res['B_inv_new'] = B_inv_new
    optimization_res['eng_xnew_x0_diff'] = eng_xnew_x0_diff

    return optimization_res

if __name__ == "__main__":

    t1 = True
    if t1:
        y = np.array([1,2,1]) - np.array([2,1,2])
        grad = np.array([1,2,1])
        s = np.array([1,1,1])
        inv_B_new = BFGS(y=y, s=s, inv_B=np.eye(3))
        p_k = np.dot(inv_B_new, -grad)
        p_k_n = p_k/ np.linalg.norm(p_k)
        print("inv_B y : ", np.matmul(inv_B_new, y), "should be equal to s", s)
        B = np.linalg.inv(inv_B_new)
        print("-Bp_k", np.matmul(-B, np.transpose(p_k)), "Should be equal to grad", grad)