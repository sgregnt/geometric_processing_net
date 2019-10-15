import numpy as np
import scipy.io
from scipy.linalg import sqrtm
import sys, os
from utils.debug_and_plot_utils import do_verbose
sys.path.insert(0, '/home/greg/libigl/libigl/python/')
from iglhelpers import *
import pyigl as igl

def centralize(X):
    """centralize vertices in the mesh data"""
    return X - np.mean(X, axis=0)

def add_noise(X, scale=1):
    """add random gaussian noise to the vertices in the mesh"""
    return X + np.random.randn(X.shape[0], X.shape[1]) * scale

def read_patch(file):
    """read disctionary in the mesh file"""

    return scipy.io.loadmat(file)

def extract_tet(patch):
    """extract mesh parameters and convert to type suitable for libigl"""
    return (patch['V'].astype('float64'), patch['T'].astype('int32') - 1, patch['fV'].astype('float64'))

def beautify_data(V, T, fV):
    """wrapper to centralize V and fV"""

    fV = centralize(fV)
    V = centralize(V)
    V_with_fV = np.vstack((V, fV))
    return (V, T, fV, V_with_fV)

def gradient_helper(u, v, sls, dfVcx):
    """ Helper function with gradient evaluation of singular values
    # ds_1 / dfVx_ij and  ds_2 / dfVx_ij in 3-by-3 diagonal matrix
    """

    I = np.eye(3)
    dA = np.matmul(dfVcx, sls)
    dS = I * np.matmul(np.matmul(np.transpose(u), dA), v)

    return dS

def compute_deng_x(deng_sv, dsv_x, fv, t):
    """ Given two differencials: that of energy with respect to singular values
    and that of singular values with respect to the vertices, compute
    gradient of energy with respect to vertices.
    """

    # Arrange denergy /ds_i1 dD /ds_i2 into a vector
    energy_ds1_ds2 = np.transpose(deng_sv).reshape((2 * t.shape[0]))
    denergy_dx = np.zeros((dsv_x.shape[1], 3))

    for x in range(fv.shape[0]):  # cycle through vertices x
        for i in range(3):  # cycle through x_1, x_2, x_3

            # Arrange dS_i1 dx dS_i2 dx into vectors
            ds1_dxi = dsv_x[:, x, i, 0, 0]
            ds2_dxi = dsv_x[:, x, i, 1, 1]

            # Combine them into alternatinge single vector
            combined_ds1_ds2_dxi = [None] * (len(ds1_dxi) + len(ds2_dxi))
            combined_ds1_ds2_dxi[::2] = ds1_dxi
            combined_ds1_ds2_dxi[1::2] = ds2_dxi
            combined_ds1_ds2_dxi = np.array(combined_ds1_ds2_dxi)

            denergy_dx[x, i] = np.matmul(energy_ds1_ds2, combined_ds1_ds2_dxi)

    return denergy_dx

def compute_sv(fv, t, sls):
    """ Compute singular values and derivatives of a map
    defined by fv on triangles t, Slc captures the structure of the source mesh
    see "mesh_params" for description

    sls should match t! (for sub mesh the function cannot berun by substitution of t to subt, sls should
    also be changed accordingly)
    """

    # Looking at the shape[2] due transpose take in the previous step
    tn = sls.shape[2]
    dfc = np.zeros((3, 3, tn))
    sv = np.zeros((tn, 3))

    df = fv.shape[1]
    dv = fv.shape[1]
    tn = t.shape[0]

    uc = np.zeros((df, df, tn))
    vc = np.zeros((dv, df, tn))

    # Maybe I need to change the order of elements in fVc
    for c in range(tn):
        fVc = np.transpose(fv[t[c], :])
        dfc[:, :, c] = - np.matmul(fVc, sls[:, :, c])

        uc[:, :, c], sv[c,], vc[:, :, c] = np.linalg.svd(dfc[:, :, c])

    return (sv, uc, vc)


def compute_dsv_x(fv, t, sls, uc, vc):
    """ Compute singular values derivatives Sv with respecto to the vertices fv on triangles t,
    sls captures the structure of the source mesh
    see "mesh_params" for description

    fv is the new position of the vertices
    t are the triangles
    sls are the descriptors of original triangles (check out mesh_param for details)
    Uc are left singular vector
    Vc are right singular vectors

    Check out  compute_singular_values for details
    """

    # Looking at the shape[2] due transpose take in the previous step
    tn = t.shape[0]

    # Compute gradient of singular values with respect to vertices fv

    # The gradient of first singular value is stored in dsv_x[:, :, 0, 0]
    # of the second in dsv_x[:, :, 1, 1], of the third (==0) is in dsv_x[:, :, 2, 2]
    dsv_x = np.zeros((tn, fv.shape[0], fv.shape[1], 3, 3))

    # Cycle through vertices x
    for i in range(fv.shape[0]):

        # Cycle through x_1, x_2, x_3
        for j in range(fv.shape[1]):
            cur_dfv_x = fv * 0
            cur_dfv_x[i, j] = 1

            # Cycle through triangles
            for c in range(tn):
                cur_dfv_xc = np.transpose(cur_dfv_x[t[c], :])

                # ds_1 / dfVx_ij and  ds_2 / dfVx_ij in 3-by-3 diagonal matrix
                dsv_cur_x = gradient_helper(u=uc[:, :, c], v=vc[:, :, c], sls=sls[:, :, c], dfVcx=-cur_dfv_xc)

                # Triangle c, gradient with respect to $fVxc_ij$
                dsv_x[c, i, j, :, :] = dsv_x[c, i, j, :, :] + dsv_cur_x

    return dsv_x

def compute_sv_dsv_x(fv, t, sls):
    """ Compute derivative of singular values with respect to x"""

    (sv, Uc, Vc) = compute_sv(fv, t, sls)
    dsv_x = compute_dsv_x(fv, t, sls, Uc, Vc)

    return sv, dsv_x

def beautify_data_cov_normalized(V, T, fV):
    """wrapper to normalize data so that the vertices have
     unit covariance (identity 2 by 2 metrix for mesh in 2D)
     """

    fV = centralize(fV)
    V = centralize(V)

    cov_V = np.cov(np.transpose(V))
    sqrt_cov_V = sqrtm(cov_V)

    V_cov_normalized = np.matmul(np.linalg.pinv(sqrt_cov_V), np.transpose(V))
    V_cov_normalized = np.transpose(V_cov_normalized)

    fV_cov_normalized = np.matmul(np.linalg.pinv(sqrt_cov_V), np.transpose(fV))
    fV_cov_normalized = np.transpose(fV_cov_normalized)

    V_with_fV = np.vstack((V_cov_normalized, fV_cov_normalized))

    return (V_cov_normalized, T, fV_cov_normalized, V_with_fV)


def all_2_e(V, fV, T):
    """convert numpy to Eigen datatypes used for visualization"""

    return (p2e(V), p2e(fV), p2e(T))

def normalize_rows(X):
    """return matrix with normalized rows"""
    row_norms = np.linalg.norm(X, axis=1).reshape(X.shape[0], 1)
    m = X.shape[1]
    Xnorm = X / row_norms
    return Xnorm

def is_flip(V, T):
    """return true if some triangles have flipped,
     otherwise returns false"""

    A = V[T[:, 1], :] - V[T[:, 0], :]
    B = V[T[:, 2], :] - V[T[:, 1], :]
    C = V[T[:, 0], :] - V[T[:, 2], :]

    detT = np.cross(A, C)[:, -1]

    if (sum(detT > 0)) > 0:
        return True
    else:
        return False

def mesh_params(V, T):
    """extract various useful parameters for each triangle"""
    tn = T.shape[0]

    # A = V(T(:, 2),:) - V(T(:, 1),:);
    # B = V(T(:, 3),:) - V(T(:, 2),:);
    # C = V(T(:, 1),:) - V(T(:, 3),:);

    A = V[T[:, 1], :] - V[T[:, 0], :]
    B = V[T[:, 2], :] - V[T[:, 1], :]
    C = V[T[:, 0], :] - V[T[:, 2], :]

    # t_mesh.Acen = (V(T(:, 2),:) + V(T(:, 1),:)) / 2;
    # t_mesh.Bcen = (V(T(:, 3),:) + V(T(:, 2),:)) / 2;
    # t_mesh.Ccen = (V(T(:, 1),:) + V(T(:, 3),:)) / 2;

    Acen = (V[T[:, 1], :] + V[T[:, 0], :]) / 2.0
    Bcen = (V[T[:, 2], :] + V[T[:, 1], :]) / 2.0
    Ccen = (V[T[:, 0], :] + V[T[:, 2], :]) / 2.0

    # detT = norms(cross(A, C), [], 2);
    detT = np.linalg.norm(np.cross(A, C), axis=1).reshape(tn, 1)

    # normT = cross(A, B, 2);
    normT = np.cross(A, B)

    # normLen = norms(normT, [], 2);
    normLen = np.linalg.norm(normT, axis=1).reshape(tn, 1)

    # normT = normT. / normLen;
    normT = normT / normLen

    # t_mesh.M = (1 / 2) * abs(detT);
    t_mesh_M = (1 / 2) * abs(detT)

    # S = zeros(tn, 3, 3);
    S = np.zeros((tn, 3, 3))

    # S(:,:, 1) = normalize_rows(cross(B, normT, 2)). * norms(B, [], 2);
    # S(:,:, 2) = normalize_rows(cross(C, normT, 2)). * norms(C, [], 2);
    # S(:,:, 3) = normalize_rows(cross(A, normT, 2)). * norms(A, [], 2);

    S[:, :, 0] = normalize_rows(np.cross(B, normT)) * np.linalg.norm(B, axis=1).reshape(tn, 1)
    S[:, :, 1] = normalize_rows(np.cross(C, normT)) * np.linalg.norm(C, axis=1).reshape(tn, 1)
    S[:, :, 2] = normalize_rows(np.cross(A, normT)) * np.linalg.norm(A, axis=1).reshape(tn, 1)

    # t_mesh.SLc = permute(S ./ detT,[3 2 1])
    SLc = np.transpose(S / detT.reshape(tn, 1, 1), (2, 1, 0))

    mesh_dic = {'A' : A,
                'B' : B,
                'C' : C,
                'Acen' : Acen,
                'Bcen' : Bcen,
                'Ccen': Ccen,
                'detT' : detT,
                'normT': normT,
                't_mesh_M' : t_mesh_M,
                'S' : S,
                'sls' : SLc,
                't': T,
                'vertices' : np.unique(T)}

    return mesh_dic

def mesh_unpack(mesh_dic):
    return (mesh_dic['A'],
     mesh_dic['B'],
     mesh_dic['C'],
     mesh_dic['Acen'],
     mesh_dic['Bcen'],
     mesh_dic['Ccen'],
     mesh_dic['detT'],
     mesh_dic['normT'],
     mesh_dic['t_mesh_M'],
     mesh_dic['S'],
     mesh_dic['sls'],
     mesh_dic['t'],
     mesh_dic['vertices'])

def show_all(eV, eT, efV):
    """open viewer show original mesh and how f moves the mesh"""

    red = p2e(np.array([[1, 0, 0]]).astype('float64'))
    blue = p2e(np.array([[0, 0, 1]]).astype('float64'))
    viewer = igl.glfw.Viewer()
    viewer.data(0).show_lines = True
    viewer.data(0).set_mesh(eV, eT)
    viewer.data(0).add_points(efV, blue)
    viewer.data(0).add_edges(eV, efV, red)

    return viewer

def input_prep(patch, do_show=True, avoid_triangle_flip=False):
    """extract mesh information from a patch
    avoid_triangle_flip is set true will generate data until no flipped triangle ave appeared
    """

    V, T, fV = extract_tet(patch)

    # temporary to fix one of the problematic triangles
    fV[3, :] = fV[3, :] + (-0.1, -0.1, 0)


    # run centralization
    V = centralize(V)
    fV = centralize(fV)

    # add dummy third dimension
    if fV.shape[1] == 2:
        fV_tmp = fV.copy()
        fV = V * 0
        fV[:, 0:2] = fV_tmp

    # add noise to fV
    Vxy = V[:, 0:2]
    fVxy = fV[:, 0:2].copy()
    displacement_scale = np.sqrt(np.var(Vxy - fVxy))
    fV[:, 0:2] = add_noise(fVxy, displacement_scale * 1)

    # regenerate fV untill no flips occure
    if avoid_triangle_flip:
         while is_flip(fV, T):
            fV[:, 0:2] = add_noise(fVxy, displacement_scale * 1)

    # print("aa- fV[:, 0:2]", aa- fV[:, 0:2])
    fV = centralize(fV)

    # normalize covariance of vertices and use same transformation on fV
    V, T, fV, _ = beautify_data_cov_normalized(V, T, fV)


    if 'subT' in patch.keys():
        subT = patch['subT'].astype('int32') - 1
    else:
        subT = None

    return V, T, fV, subT

def show_mesh(eV, eT):
    """open viewer show original mesh and how f moves the mesh"""
    viewer = igl.glfw.Viewer()
    viewer.data(0).show_lines = True
    viewer.data(0).set_mesh(eV, eT)
    return viewer

def show(V, T, fV, do=False, show_source=True):
    eV, eT, efV = all_2_e(V, T, fV)

    if do:
        if show_source:
            viewer = show_all(eV, eT, efV)
            viewer.launch()
        viewer = show_mesh(efV, eT)
        viewer.launch()

def check_triangle_flip(fv, t, verbose=False):
    """ Check if any of the triangle has flipped"""

    (A, B, C, _, _, _, _, _, _, _, _, _, _) = mesh_unpack(mesh_params(fv, t))
    detT = np.cross(A, C)[:, -1]

    if (sum(detT > 0)) > 0:
        do_verbose("triangle flipped, sum(detT > 0)", sum(detT > 0), verbose)
        return True
    else:
        return False

def get_triangle_one_rign(T, ref_t):
    """ get one ring for triangle ref_t
    consists of all triangles that share
    at least one vertex with ref_t
    """

    tmp = []
    for ref_v in ref_t:
        tmp.extend([t for t in T if ref_v in t])
    t_all = np.array(tmp)
    t_unique = np.unique(t_all, axis=0)
    return t_unique

def get_patch_one_rign(T, ref_T):
    """ get one ring for triangle ref_t
    consists of all triangles that share
    at least one vertex with ref_t
    """

    tmp = []
    for ref_t in ref_T:
        for ref_v in ref_t:
            tmp.extend([t for t in T if ref_v in t])
    t_all = np.array(tmp)
    t_unique = np.unique(t_all, axis=0)
    return t_unique

def renumber_mesh(V, fV, new_T):
    """renumbers vertices according to new_T so
    the mesh starts with zero vector.
    """

    hash = np.array(list(range(max(np.unique(new_T.flatten(), axis=0)) + 1)))
    hash[:] = -1
    i = 0
    sub_v = []
    for t in new_T:
        for v in t:
            if hash[v] == -1:
                sub_v.append(v)
                hash[v] = i
                i += 1

    new_V = []
    new_fV = []
    for v in sub_v:
        new_V.append(V[v])
        new_fV.append(fV[v])

    updated_new_T = []
    for t in new_T:
        triangle = []
        for v in t:
            triangle.append(hash[v]+1) # numberinf in T starts from 1 not zero.
        updated_new_T.append(triangle)
    new_V = np.array(new_V)
    new_fV = np.array(new_fV)
    updated_new_T = np.array(updated_new_T)
    return (new_V, new_fV, updated_new_T)


def store_patch(V, fV, T, subT, filepath):
    scipy.io.savemat(filepath,  {'V':V, 'fV' : fV, 'T':T, 'subT':subT})