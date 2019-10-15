import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, './utils')
from utils.mesh_utils import is_flip, add_noise, extract_tet, show, mesh_params, input_prep, read_patch, compute_sv, compute_deng_x, compute_dsv_x, mesh_unpack
from utils.dirichlet_sym_utils import dirichlet_sym,  dirichlet_sym_get_compute_functions, dirichlet_sym_compute_all_from_x, res_unpack
from utils.optimization_utils import step_unpack, BFGS, get_p_k_direction, approx_f_xnew_first_order
from utils.debug_and_plot_utils import do_verbose, calculate_plots, plot_line_search_results, test, get_engs
import scipy.io
import os


def NN_H(deng_x_at_x0, deng_x_at_xnew, s, inv_H):
    """ Calculate inverse Hessian for BFGS algorithm"""

    direction = -np.dot(inv_H, deng_x_at_x0.flatten())
    return direction.reshape(-1, 3), inv_H


def dirichlet_sym_gd_bfgs_nn_weak_armijo_optimizer_step(line_search, get_nn_hessians, source, t, target, subt=None, verbose=True, B_inv_new=None, type='GD'):
    """ Run line search along gradient direction. Currently for dirichlet energy only."""

    mesh = mesh_params(source, t)
    mesh_sub = mesh_params(source, subt)

    (_, _, compute_eng_from_x) = dirichlet_sym_get_compute_functions(mesh['detT'])
    (_, _, compute_eng_from_x_sub) = dirichlet_sym_get_compute_functions(mesh_sub['detT'])

    compute_all_from_x = dirichlet_sym_compute_all_from_x(mesh['detT'])
    compute_all_from_x_sub = dirichlet_sym_compute_all_from_x(mesh_sub['detT'])

    all_at_x0 = compute_all_from_x((target, mesh['t'], mesh['sls']))
    all_at_x0_sub = compute_all_from_x_sub((target, mesh_sub['t'], mesh_sub['sls']))

    if type == 'NN':
        Hx, Hy = get_nn_hessians(fv=target, v=source, subt=subt)
    else:
        Hx = None
        Hy = None

    p_k, init_alpha = get_p_k_direction(B_inv_new, type, mesh_sub, Hx, Hy, all_at_x0)

    if type == 'BFGS':
        if B_inv_new is None:
            B_inv_new = np.eye(all_at_x0['deng_x_at_x'].flatten().shape[0])

    params = line_search(init_alpha=init_alpha,
                         mesh=mesh,
                         p_k=p_k,
                         mesh_sub=mesh_sub,
                         all_at_x0=all_at_x0,
                         compute_all_from_x=compute_all_from_x,
                         compute_all_from_x_sub=compute_all_from_x_sub,
                         c1=0.001,
                         c2=0.9,
                         verbose=verbose,
                         type=type)

    (step,
     all_at_xnew,
     all_at_xnew_sub) = params

    (eng_at_xnew,
     deng_sv_at_xnew,
     deng_x_at_xnew,
     deng_x_at_xnew_n,
     sv_at_xnew,
     dsv_x_at_xnew,
     fv_at_xnew) = res_unpack(all_at_xnew)

    (_, _, alpha, p_k, p_k_n, type) = step_unpack(step)

    if type == 'BFGS':
        inv_B_new = BFGS(y=(deng_x_at_xnew - all_at_x0['deng_x_at_x']), s=alpha * p_k, inv_B=B_inv_new)
    else:
        inv_B_new = None

    # If 'do_show' is active will plot line  figure with objective
    # and slopes for both the full mesh and the sub mesh
    plot_params = calculate_plots(res_unpack=res_unpack,
                                  mesh_unpack=mesh_unpack,
                                  step_unpack=step_unpack,
                                  step=step,
                                  mesh=mesh,
                                  mesh_sub= mesh_sub,
                                  all_at_x0=all_at_x0,
                                  all_at_xnew=all_at_xnew,
                                  all_at_x0_sub=all_at_x0_sub,
                                  all_at_xnew_sub=all_at_xnew_sub,
                                  compute_eng_from_x=compute_eng_from_x,
                                  compute_eng_from_x_sub=compute_eng_from_x_sub,
                                  do_show=True)

    plot_line_search_results(step_unpack=step_unpack, plot_params=plot_params, step=step, do_show=False)

    return (all_at_x0,
            all_at_xnew,
            all_at_x0_sub,
            all_at_xnew_sub,
            mesh,
            mesh_sub,
            plot_params,
            step,
            inv_B_new)

def dirichlet_sym_compare_energies(get_nn_hessians, source, t, target, subt=None):
    """ Compare how good the energy is approximated by various methods"""

    mesh = mesh_params(source, t)
    mesh_sub = mesh_params(source, subt)

    (_, _, compute_eng_from_x) = dirichlet_sym_get_compute_functions(mesh['detT'])
    (_, _, compute_eng_from_x_sub) = dirichlet_sym_get_compute_functions(mesh_sub['detT'])

    compute_all_from_x = dirichlet_sym_compute_all_from_x(mesh['detT'])
    compute_all_from_x_sub = dirichlet_sym_compute_all_from_x(mesh_sub['detT'])

    all_at_x0 = compute_all_from_x((target, mesh['t'], mesh['sls']))
    all_at_x0_sub = compute_all_from_x_sub((target, mesh_sub['t'], mesh_sub['sls']))

    Hx, Hy = get_nn_hessians(fv=target, v=source, subt=subt)

    target = source
    Vxy = source[:, 0:2]
    fVxy = target[:, 0:2].copy()
    fV = target.copy() * 0
    displacement_scale = np.sqrt(np.var(Vxy - fVxy))
    fV[:, 0:2] = add_noise(fVxy, displacement_scale * 0.5)

    p_k = 0.7 * all_at_x0['deng_x_at_x'] / np.linalg.norm(all_at_x0['deng_x_at_x'])
    p_k = 0 * all_at_x0['deng_x_at_x']
    p_k[:, 0:2] = 0.1 * np.random.randn(p_k.shape[0], 2)
    # p_k[2, 0:2] = 0.2 * np.abs(np.random.randn(1, 2))
    fV = target + p_k
    # regenerate fV untill no flips occure
    if True:
        while is_flip(fV, t):
            1/0
            fV[:, 0:2] = add_noise(fVxy, displacement_scale * 1)

    displacement_scale = np.sqrt(np.var(Vxy - fVxy))
    p_k = fV - target
    p_k_n = p_k/ np.linalg.norm(p_k)
    c = np.linalg.norm(p_k)
    show(target, t, fV, True)
    # show(target, t, target + p_k, True)

    step = {'x0' : all_at_x0['x'],
            'xnew' : None,
            'alpha' : 1,
            'p_k' : p_k,
            'p_k_n' : p_k_n,
            'type' : 'dummy'}

    span = (-0.15, 0.15)

    def compute_energy_estimate(x):
        (fv, t, sls) = x[0], x[1], x[2]
        p_k = fv - all_at_x0['x']
        eng_xnew_first_order_approx = approx_f_xnew_first_order(f_x0=all_at_x0['eng_at_x'],
                                                                df_x0=all_at_x0['deng_x_at_x'],
                                                                step=p_k)
        corr_p_k = p_k

        p_k_x_sub = corr_p_k[mesh_sub['vertices'], 0].flatten()
        p_k_y_sub = corr_p_k[mesh_sub['vertices'], 1].flatten()

        square_x = np.matmul(p_k_x_sub, np.matmul(Hx, p_k_x_sub))
        square_y = np.matmul(p_k_y_sub, np.matmul(Hy, p_k_y_sub))

        return eng_xnew_first_order_approx + square_x + square_y

    def compute_energy_estimate_sub(x):
        (fv, t, sls) = x[0], x[1], x[2]
        p_k = fv - all_at_x0['x']
        eng_xnew_first_order_approx = approx_f_xnew_first_order(f_x0=all_at_x0_sub['eng_at_x'],
                                                                df_x0=all_at_x0_sub['deng_x_at_x'],
                                                                step=p_k)
        corr_p_k = p_k

        # for the reminder of the mesh just do x^2
        p_k_others = p_k.copy()
        p_k_others[mesh_sub['vertices'], :] = 0

        p_k_x_sub = corr_p_k[mesh_sub['vertices'], 0].flatten()
        p_k_y_sub = corr_p_k[mesh_sub['vertices'], 1].flatten()

        square_x = np.matmul(p_k_x_sub, np.matmul(Hx, p_k_x_sub))
        square_y = np.matmul(p_k_y_sub, np.matmul(Hy, p_k_y_sub))

        return eng_xnew_first_order_approx + square_x + square_y + np.matmul(p_k_others.flatten(), p_k_others.flatten())

    eng_plot = get_engs(compute_eng_from_x=compute_eng_from_x, fv=step['x0'], t=t, sls=mesh['sls'], d=p_k_n, span=span)
    eng_estimate_plot = get_engs(compute_eng_from_x=compute_energy_estimate, fv=step['x0'], t=t, sls=mesh['sls'], d=p_k_n, span=(-0.15, 0.15))

    eng_estimate_plot_sub = get_engs(compute_eng_from_x=compute_energy_estimate_sub, fv=step['x0'], t=subt, sls=mesh_sub['sls'], d=p_k_n, span=(-0.25, 0.25))
    eng_plot_sub = get_engs(compute_eng_from_x=compute_eng_from_x_sub, fv=step['x0'], t=subt, sls=mesh_sub['sls'], d=p_k_n, span=(-0.25, 0.25))

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(eng_estimate_plot[0], eng_estimate_plot[1], '-b', label='eng_estimate')
    # ax.plot(eng_plot[0], eng_plot[1], '-g', label='eng_true')
    # plt.legend(loc='center left')
    # plt.title('compare estimates with energy curve')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(eng_estimate_plot_sub[0], eng_estimate_plot_sub[1], '-k', label='eng_sub_estimate')
    ax.plot(eng_plot_sub[0], eng_plot_sub[1], '-m', label='eng_sub_true')
    plt.legend(loc='center left')
    plt.title('compare estimates with energy curve sub patch')


def gradient_approximation_quality(true, original, gradient, alpha, title):
    """ The difference between energy value at xnew and estiamted energy value by linear approximation"""

    diff =  np.squeeze(true) - ( np.squeeze(original) - np.trace(
                                                        np.matmul(np.transpose(gradient, (0,2,1)), gradient),
                                                        axis1= 1, axis2=2
                                                        ) * np.squeeze(alpha))
    plt.hist(diff, bins=100)
    plt.title(title + ' ' + str(np.mean(diff)))
    plt.show()

    plt.hist(diff- np.mean(diff), bins=100)
    plt.title(title + ' mean removed')
    plt.show()


if __name__ == "__main__":

        t1 = True
        t2 = True

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # t1 - Test that dirichlet energy calcualtion and the derivatives here in python
        # match the calculation done on Matlab.
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if t1:
            dir = '/home/greg/newproject/data/'
            filename = 'good_skew_bar4x4_V_T_S_Sc_SLc_normT_detT_E_fV_Acen_Bcen_Ccen_E_dE_t_v_grad_SV.mat'
            patch = scipy.io.loadmat(os.path.join(dir, filename))

            v, t, fv = extract_tet(patch)

            # Add dummy third dimension to fv
            fv_tmp = fv.copy()
            fv = v * 0
            fv[:, 0:2] = fv_tmp

            show(v, t, fv, do=True)
            (A, B, C, Acen, Bcen, Ccen, detT, normT, t_mesh_M, S, sls, _, _) = mesh_unpack(mesh_params(v, t))

            (sv, uc, vc) = compute_sv(fv, t, sls)
            dsv_x = compute_dsv_x(fv, t, sls, uc, vc)

            (energy, deng_sv) = dirichlet_sym(sv[:, 0], sv[:, 1], np.array(list(0.5 * detT)))

            deng_x = compute_deng_x(deng_sv=deng_sv, dsv_x=dsv_x, fv=fv, t=t)

            test(patch['Acen'], patch['Bcen'], patch['Ccen'], patch['S'], patch['SLc'], patch['detT'], patch['normT'],
                 patch['SV'], patch['E'], patch['grad'], patch['dE'], Acen,
                 Bcen, Ccen, S, sls, detT, normT, sv[:, 0:2], energy, np.transpose(deng_sv), deng_x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #  t2 - Test of draw_energy_along_gradient
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        if t2:
            np.random.seed(seed=47)
            root_path = os.path.dirname(os.path.abspath(__file__))
            file = os.path.join(root_path, '../data/mesh_examples/grind_and_subgrid.mat')
            patch = read_patch(file)
            v, t, fv, subt = input_prep(patch, do_show=True, avoid_triangle_flip=True)
            show(v, t, fv, do=True)
            params = dirichlet_sym_gd_bfgs_nn_weak_armijo_optimizer_step(v, t, fv, subt, do_show=True)

            (alpha,
             eng_at_x0,
             deng_sv_at_x0,
             deng_x_at_x0,
             sv_at_x0,
             deng_x_at_x0_n,
             step_fv,
             eng_at_xnew,
             sv_at_xnew,
             eng_at_xnew_sub,
             sv_at_xnew_sub,
             eng_at_x0_sub,
             deng_x_at_x0_sub,
             sv_at_x0_sub) = params

            verbose = True
            do_verbose('alpha', alpha, verbose)
            do_verbose('eng_at_x0', eng_at_x0, verbose)
            do_verbose('deng_sv_at_x0', deng_sv_at_x0, verbose)
            do_verbose('deng_x_at_x0', deng_x_at_x0, verbose)
            do_verbose('sv_at_x0', sv_at_x0, verbose)
            do_verbose('deng_x_at_x0_n', deng_x_at_x0_n, verbose)
            do_verbose('step_fv', step_fv, verbose)
            do_verbose('eng_at_xnew', eng_at_xnew, verbose)
            do_verbose('sv_at_xnew', sv_at_xnew, verbose)
            do_verbose('eng_at_xnew_sub', eng_at_xnew_sub, verbose)
            do_verbose('sv_at_xnew_sub', sv_at_xnew_sub, verbose)
            do_verbose('eng_at_x0_sub', eng_at_x0_sub, verbose)
            do_verbose('deng_x_at_x0_sub',deng_x_at_x0_sub, verbose)
            do_verbose('sv_at_x0_sub', sv_at_x0_sub, verbose)