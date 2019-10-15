import numpy as np
import sys
sys.path.insert(0, './utils')
from utils.mesh_utils import mesh_params, compute_sv, compute_deng_x, compute_dsv_x, compute_sv_dsv_x, mesh_unpack

def dirichlet_sym_eng(s1, s2, area):
    """ Compute dirichlet energy its value and derivative with respect to singular values
    per triangle
    """

    raw_energy = (s1 ** 2 + s2 ** 2 + s1 ** (-2) + s2 ** (-2))
    raw_energy = raw_energy.reshape((s1.shape[0], 1))
    energy = np.sum(raw_energy * area)

    return energy

def dirichlet_sym_deng_sv(s1, s2, area):
    """ Compute dirichlet energy its value and derivative with respect to singular values
    per triangle
    """

    raw_ds1 = (-2 * s1 ** (-3) + 2 * s1)
    raw_ds1 = raw_ds1.reshape((raw_ds1.shape[0], 1))
    raw_ds2 = (-2 * s2 ** (-3) + 2 * s2)
    raw_ds2 = raw_ds2.reshape((raw_ds2.shape[0], 1))

    deng_sv = np.array([raw_ds1 * area, raw_ds2 * area])
    deng_sv = deng_sv.reshape((2, deng_sv.shape[1]))

    return deng_sv

def dirichlet_sym(s1, s2, area):
    """ Compute dirichlet energy its value and derivative with respect to singular values
    per triangle
    """

    raw_energy = (s1 ** 2 + s2 ** 2 + s1 ** (-2) + s2 ** (-2))
    raw_energy = raw_energy.reshape((s1.shape[0], 1))
    energy = np.sum(raw_energy * area)

    raw_ds1 = (-2 * s1 ** (-3) + 2 * s1)
    raw_ds1 = raw_ds1.reshape((raw_ds1.shape[0], 1))
    raw_ds2 = (-2 * s2 ** (-3) + 2 * s2)
    raw_ds2 = raw_ds2.reshape((raw_ds2.shape[0], 1))

    deng_sv = np.array([raw_ds1 * area, raw_ds2 * area])
    deng_sv = deng_sv.reshape((2, deng_sv.shape[1]))

    return (energy, deng_sv)

def dirichlet_sym_eng_deng_sv_deng_x_sv_at_fv(v, t, fv):
    """ Calculate energy and its derivative with respect to singular
    values (two values per triangle) and  with respect to vertices
    """

    # Calculate different useful quantities for mesh processing
    (A, B, C, Acen, Bcen, Ccen, detT, normT, t_mesh_M, S, sls, _, _) = mesh_unpack(mesh_params(v, t))

    # Calculate gradient d(E_s)dx @ fv == E_s @ v + displacement
    (sv, uc, vc) = compute_sv(fv, t, sls)
    dsv_x = compute_dsv_x(fv, t, sls, uc, vc)

    # sv, dsv_x = compute_singular_values_and_derivative(fv, t, sls)
    s1_at_fv, s2_at_fv = sv[:, 0], sv[:, 1]

    (eng_at_fv, deng_sv_at_fV) = dirichlet_sym(s1=s1_at_fv, s2=s2_at_fv, area=np.array(list(0.5 * detT)))
    deng_x_at_fv = compute_deng_x(deng_sv=deng_sv_at_fV, dsv_x=dsv_x, fv=fv, t=t)

    return (eng_at_fv, deng_sv_at_fV, deng_x_at_fv, sv)

def dirichlet_sym_get_compute_functions(detT):
    """ Generates function with with one can compute derivatives and values of objective for symmetric dirichlet energy"""

    # Function returning dirichlet energy (with area substituted)
    # Some magic: x is matrix with singular values of the size  trinagles - by - 3, slo I have to extract
    # the first two columns and split the matrix by the columns.
    compute_eng_from_sv = lambda x: dirichlet_sym_eng(*(np.hsplit(x[:, [0, 1]], 2)), np.array(list(0.5 * detT)))

    # Function returning derivative of dirichlet energy (with area substituted)
    # Some magic: x is matrix with singular values of the size  trinagles - by - 3, slo I have to extract
    # the first two columns and split the matrix by the columns.
    compute_deng_sv_from_sv = lambda x: dirichlet_sym_deng_sv(*(np.hsplit(x[:, [0, 1]], 2)), np.array(list(0.5 * detT)))

    # Calculate singular values from fv, unpack only the first two singular values
    # first get only the singular values part trinagles -by- 3, extract the first two columns
    # and split the matrix by the columns.
    compute_eng_from_x = lambda x: dirichlet_sym_eng(*(np.hsplit(
                                                               compute_sv(fv=x[0], t=x[1], sls=x[2])[0][:, [0, 1]]
                                                               , 2)
                                                         ), np.array(list(0.5 * detT)))

    return(compute_eng_from_sv, compute_deng_sv_from_sv, compute_eng_from_x)

def dirichlet_sym_obejective_and_gradient_at_x(x, t, sls, detT):
    """ Calcualted all derivatives and values of objective for symmetric dirichlet energy at a given point x"""

    (_, compute_deng_sv_from_sv, compute_eng_from_x) = dirichlet_sym_get_compute_functions(detT)
    (sv_at_x0, dsv_x_at_x0) = compute_sv_dsv_x(fv=x, t=t, sls=sls)
    eng_at_x0 = compute_eng_from_x((x, t, sls))
    deng_sv_at_x0 = compute_deng_sv_from_sv(sv_at_x0)
    deng_x_at_x0 = compute_deng_x(deng_sv=deng_sv_at_x0, dsv_x=dsv_x_at_x0, fv=x, t=t)
    deng_x_at_x0_n = deng_x_at_x0/np.linalg.norm(deng_x_at_x0)

    res = {'eng_at_x': eng_at_x0, 'deng_sv_at_x' : deng_sv_at_x0, 'deng_x_at_x' : deng_x_at_x0, 'deng_x_at_x_n' : deng_x_at_x0_n, 'sv_at_x' : sv_at_x0, 'dsv_x_at_x' : dsv_x_at_x0, 'x': x}
    return res

def res_unpack(res):
    return (res['eng_at_x'], res['deng_sv_at_x'], res['deng_x_at_x'], res['deng_x_at_x_n'], res['sv_at_x'], res['dsv_x_at_x'], res['x'])

def dirichlet_sym_compute_all_from_x(detT):
    """ Generates function that can be used to calculat all derivatives and values of objective for symmetric dirichlet energy at a given point x"""

    compute_all_from_x = lambda x: dirichlet_sym_obejective_and_gradient_at_x(x=x[0], t=x[1], sls=x[2], detT=detT)
    return compute_all_from_x
