import numpy as np
import matplotlib.pyplot as plt

def do_verbose(title, data, verbose):
    """ Print variable value for debugging"""

    if verbose:
        print(title, ':', data)

def get_engs(compute_eng_from_x, fv, t, sls, d, span):
    """ Computes values of energy starting from fv in the direction d. I.e., E[fv + alpha * d]"""

    xs = np.linspace(span[0], span[1], 100)
    fvs = [fv + x * d for x in xs]
    ys = [compute_eng_from_x ((fv, t, sls)) for fv in fvs]

    return xs, np.array(ys)

def directional_derivative(dir_n, grad):
    return np.matmul(dir_n.flatten(), grad.flatten())

def calculate_plots(res_unpack, mesh_unpack, step_unpack, step, mesh, mesh_sub, all_at_x0, all_at_xnew, all_at_x0_sub, all_at_xnew_sub,
                    compute_eng_from_x, compute_eng_from_x_sub, do_show):

    """ When do_show is active plots a graph showing the energy along gradient and line search
    results for the whole mesh and for the submesh. Main purpose of this function is
    for debugging and verification of the calculations."""

    (_, _, _, _, _, _, _, _, _, _, sls, t, _) = mesh_unpack(mesh)
    (_, _, _, _, _, _, _, _, _, _, slssub, subt, _) = mesh_unpack(mesh_sub)
    (_, _, alpha, p_k, p_k_n, type) = step_unpack(step)

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

    if do_show:

        # Derivative at the targets
        # using p_k instead of p_k_n since the x axis is in terms of p_k not interms of normalized
        # p_k_n, this in turn is done to insure that at the alpha we recover the energy ant the next step.
        deng_x_at_x0_slope = directional_derivative(dir_n=p_k, grad=deng_x_at_x0)
        deng_x_at_xnew_slope = directional_derivative(dir_n=p_k, grad=deng_x_at_xnew)

        deng_x_at_x0_sub_slope = directional_derivative(dir_n=p_k, grad=deng_x_at_x0_sub)
        deng_x_at_xnew_sub_slope = directional_derivative(dir_n=p_k, grad=deng_x_at_xnew_sub)

        # x axis span
        span = (0, alpha)

        # Line that passes through eng_at_x0 and has the deng_x_at_x0_slope deng_x_at_x0
        ray_deng_x_at_x0 = get_ray(y0=eng_at_x0, x0=0, slope=deng_x_at_x0_slope, span=span)
        ray_deng_x_at_xnew = get_ray(y0=eng_at_xnew, x0=alpha, slope=deng_x_at_xnew_slope, span=span)

        # Energy value along the ray given by tangent at x of submesh
        eng_plot = get_engs(compute_eng_from_x=compute_eng_from_x, fv=fv_at_x0, t=t, sls=sls, d=p_k, span=span)

        ray_deng_x_at_x0_sub = get_ray(y0=eng_at_x0_sub, x0=0, slope=deng_x_at_x0_sub_slope, span=span)
        ray_deng_x_at_xnew_sub = get_ray(y0=eng_at_xnew_sub, x0=alpha, slope=deng_x_at_xnew_sub_slope, span=span)

        eng_plot_sub = get_engs(compute_eng_from_x=compute_eng_from_x_sub, fv=fv_at_x0,
                                t=subt, sls=slssub, d=p_k, span=span)

        plot_params = (ray_deng_x_at_x0,
                       ray_deng_x_at_xnew,
                       eng_plot,
                       ray_deng_x_at_x0_sub,
                       ray_deng_x_at_xnew_sub,
                       eng_plot_sub,
                       span)

        return plot_params

def get_ray(y0, x0, slope, span):
    """ Line that goes through a given point with a given slope"""

    ts = np.linspace(span[0], span[1], 100)
    ys = y0 + (ts - x0) * slope
    not_extreme = ys > 10
    ts = ts[not_extreme]
    ys = ys[not_extreme]
    return (ts, ys)

def plot_line_search_results(step_unpack, step, plot_params, do_show):
    """ When do_show is active plots a graph showing the energy along gradient and line search
    results for the whole mesh and for the submesh. Main purpose of this function is
    for debugging and verification of the calculations."""

    if do_show:
        (ray_deng_x_at_x0, ray_deng_x_at_xnew, eng_plot, ray_deng_x_at_x0_sub, ray_deng_x_at_xnew_sub, eng_plot_sub, _) = plot_params

        (_, _, alpha, step_fv, step_fv_n, type) = step_unpack(step)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # ax.plot(ray_deng_x_at_x0_sub[0][-20:], ray_deng_x_at_x0_sub[1][-20:], '-r', label='eng_sub_grad')
        ax.plot(ray_deng_x_at_x0_sub[0], ray_deng_x_at_x0_sub[1], '-g', label='eng_sub_grad')
        # ax.plot(ray_deng_x_at_xnew_sub[0][-20:], ray_deng_x_at_xnew_sub[1][-20:], '-y', label='eng_sub_grad_new')
        ax.plot(ray_deng_x_at_xnew_sub[0], ray_deng_x_at_xnew_sub[1], '-c', label='eng_sub_grad_new')
        ax.plot(eng_plot_sub[0], eng_plot_sub[1], '-b', label='eng_sub')
        plt.axvline(x=alpha)
        plt.legend(loc='center left')
        plt.title('Sub patch dynamics ' + type)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # ax.plot(ray_deng_x_at_x0[0][-20:], ray_deng_x_at_x0[1][-20:], '-g', label="eng_grad")
        ax.plot(ray_deng_x_at_x0[0], ray_deng_x_at_x0[1], '-g', label="eng_grad")
        ax.plot(ray_deng_x_at_xnew[0], ray_deng_x_at_xnew[1], '-c', label="eng_grad_new")
        ax.plot(eng_plot[0], eng_plot[1], '-b', label='eng')
        plt.legend(loc='center left')
        plt.axvline(x=alpha)
        plt.title('Full patch dynamics for ' + type)
        plt.show()

def test(Acen_ref, Bcen_ref, Ccen_ref,
         S_ref, SLc_ref, detT_ref,
         normT_ref, sv_ref, energy_ref,
         grad_ref, denergy_dx_ref, Acen,
         Bcen, Ccen, S,
         sls, detT, normT,
         sv, energy, deng_sv, denergy_dx):
    """ A primitive test to check if python is correct, its just a comparison between
    python calcualtion with matlab calculations"""

    def check_if_close(a, epsilon=10 ** -10):
        if a < epsilon:
            return '| GOOD'
        else:
            return '| BAD'

    print("Acen", np.sum(Acen_ref - Acen), check_if_close(np.sum(Acen_ref - Acen)))
    print("Bcen", np.sum(Bcen_ref - Bcen), check_if_close(np.sum(Bcen_ref - Bcen)))
    print("Ccen", np.sum(Ccen_ref - Ccen), check_if_close(np.sum(Ccen_ref - Ccen)))
    print("S", np.sum(S_ref - S), check_if_close(np.sum(S_ref - S)))
    print("sls", np.sum(SLc_ref - sls), check_if_close(np.sum(SLc_ref - sls)))
    print("detT", np.sum(detT_ref - detT), check_if_close(np.sum(detT_ref - detT)))
    print("normT", np.sum(normT_ref - normT), check_if_close(np.sum(normT_ref - normT)))
    print("sv", np.sum(sv_ref - sv), check_if_close(np.sum(sv_ref - sv)))
    print("energy", np.sum(energy_ref - energy), check_if_close(np.sum(energy_ref - energy)))
    print("deng_sv, normalization be triangle are is different so quanitties differ", np.sum(grad_ref - deng_sv))
    print("denergy_dx", np.sum(denergy_dx_ref - denergy_dx[:, 0:2]),
          check_if_close(np.sum(denergy_dx_ref - denergy_dx[:, 0:2])))


def hist_array_scalars(array, title):
    """ Plot histogram for array of scalars"""

    plt.hist(array, bins=100)
    plt.title(title)
    plt.show()

def hist_singular_values(svs, title):
    """ Plot histogram for singular values"""

    sv = np.array(svs)[:, :, 0:2].flatten()
    plt.hist(sv, bins=100)
    plt.title(title)
    plt.show()


def plot_optimization_sequence(sequence_params, do_show, type):

    if do_show:

        (ray_deng_x_at_x0, ray_deng_x_at_xnew, eng_plot, ray_deng_x_at_x0_sub, eng_plot_sub, all_alpha) = sequence_params

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(ray_deng_x_at_x0[0], ray_deng_x_at_x0[1], '.g', label="deng_x_at_x0", linewidth=2)
        ax.plot(ray_deng_x_at_xnew[0], ray_deng_x_at_xnew[1], '.c', label="deng_x_at_xnew", linewidth=2)
        plt.plot((0, np.max(eng_plot[0])), (33.661, 33.661), 'k-')
        ax.plot(eng_plot[0], eng_plot[1], '-b', label='eng')
        plt.legend(loc='upper left')

        for alpha in all_alpha:
            plt.axvline(x=alpha)
        plt.title('Full patch dynamics ' + type)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for alpha in all_alpha:
            plt.axvline(x=alpha)
        ax.plot(ray_deng_x_at_x0_sub[0], ray_deng_x_at_x0_sub[1], '.y', label='deng_x_at_x0_sub', linewidth=2)
        ax.plot(eng_plot_sub[0], eng_plot_sub[1], '-m', label='eng_sub')
        plt.legend(loc='upper left')
        plt.title('Sub-patch dynamic ' + type)



def concatinate_optimization_sequence_params(all_plot_params, do_show):


    (all_ray_deng_x_at_x0,
     all_ray_deng_x_at_xnew,
     all_eng_plot,
     all_ray_deng_x_at_x0_sub,
     _,
     all_eng_plot_sub,
     all_alpha,
     all_spans,
     _) = all_plot_params[0]

    # axis shift
    s = 0
    # keep only the last elements
    llen = 20
    all_ray_deng_x_at_x0 = (all_ray_deng_x_at_x0[0][:llen], all_ray_deng_x_at_x0[1][:llen])
    all_ray_deng_x_at_x0_sub = (all_ray_deng_x_at_x0_sub[0][:llen], all_ray_deng_x_at_x0_sub[1][:llen])
    all_ray_deng_x_at_xnew = (all_ray_deng_x_at_xnew[0][-llen:], all_ray_deng_x_at_xnew[1][-llen:])
    all_eng_plot = (all_eng_plot[0], all_eng_plot[1])
    all_eng_plot_sub = (all_eng_plot_sub[0], all_eng_plot_sub[1])

    all_alpha = [all_alpha]
    s = s + all_alpha[0]

    for i in range(1, len(all_plot_params)):


        (ray_deng_x_at_x0, ray_deng_x_at_xnew, eng_plot, ray_deng_x_at_x0_sub, _, eng_plot_sub, alpha, span, _) = all_plot_params[i]


        all_ray_deng_x_at_x0 = (np.concatenate((all_ray_deng_x_at_x0[0],
                                                ray_deng_x_at_x0[0][:llen] + s), axis=0),
                                np.concatenate((all_ray_deng_x_at_x0[1], ray_deng_x_at_x0[1][:llen]), axis=0))

        all_ray_deng_x_at_xnew = (np.concatenate((all_ray_deng_x_at_xnew[0],
                                                  ray_deng_x_at_xnew[0][-llen:] + s), axis=0),
                                  np.concatenate((all_ray_deng_x_at_xnew[1], ray_deng_x_at_xnew[1][-llen:]), axis=0))

        all_eng_plot = (np.concatenate((all_eng_plot[0], eng_plot[0] + s), axis=0),
                        np.concatenate((all_eng_plot[1], eng_plot[1]), axis=0))

        all_ray_deng_x_at_x0_sub = (np.concatenate((all_ray_deng_x_at_x0_sub[0],
                                                    ray_deng_x_at_x0_sub[0][:llen] + s), axis=0),
                                    np.concatenate((all_ray_deng_x_at_x0_sub[1],
                                                    ray_deng_x_at_x0_sub[1][:llen]), axis=0))

        all_eng_plot_sub = (np.concatenate((all_eng_plot_sub[0], eng_plot_sub[0] + s), axis=0),
                            np.concatenate((all_eng_plot_sub[1], eng_plot_sub[1]), axis=0))

        all_alpha.append(alpha + s)
        s = s + alpha

    return (all_ray_deng_x_at_x0, all_ray_deng_x_at_xnew, all_eng_plot, all_ray_deng_x_at_x0_sub, all_eng_plot_sub, all_alpha)
