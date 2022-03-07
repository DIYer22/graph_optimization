#!/usr/bin/env python3

import boxx
from boxx import np

import scipy
import sklearn.cluster
from tqdm import tqdm


def norma(feat):
    minn = feat.min()
    return (feat - minn) / (feat.max() - minn)


def softmax(feat):
    exp = np.exp(feat)
    prob = exp / exp.sum(-1, keepdims=True)
    return prob


def argsort_axis0(key):
    if key.ndim == 1:
        return np.argsort(key)
    unique_key, new_value_to_sort = np.unique(key, return_inverse=1, axis=0)
    sort_idx = np.argsort(new_value_to_sort)
    return sort_idx


def np_group_by(key, values, sorted=False):
    is_single_value = isinstance(values, np.ndarray)
    if is_single_value:
        values = [values]
    if not sorted:
        sort_idx = argsort_axis0(key)
        key = key[sort_idx]
        values = [v[sort_idx] for v in values]

    unique_key, return_index = np.unique(key, return_index=True, axis=0)
    split_idx = return_index[1:]
    groups = [np.split(v, split_idx) for v in values]
    if is_single_value:
        groups = groups[0]
    return unique_key, groups


def embedding_to_affinity(embedding, distance_type="l2"):
    # embedding shape == 100, 82, 4  复杂度和内存就受不了了
    is_2d = embedding.ndim == 3
    if is_2d:
        embedding = to_1d(embedding)
    if distance_type == "l1":
        affinity = -(embedding[None] - embedding[:, None]).sum(-1)
    if distance_type == "l2":
        affinity = -((embedding[None] - embedding[:, None]) ** 2).sum(-1) ** 0.5
    # if distance_type == "none_linear":
    #     affinity = 2 - ((embedding[None] - embedding[:, None]) ** 2).sum(-1) ** 0.5
    return norma(affinity)


def spectral_clustering(affinity, n_clusters=10):  # return partitioned cluster index
    import scipy.sparse

    if isinstance(affinity, np.ndarray):
        affinity = scipy.sparse.coo_matrix(affinity)
    solver = None
    solver = "arpack"
    solver = "lobpcg"
    solver = "amg"
    sc = sklearn.cluster.SpectralClustering(
        n_clusters, affinity="precomputed", eigen_solver=solver
    ).fit(affinity)
    cluster_idx = sc.labels_
    return cluster_idx


def desnse_crf(prob, embedding, itern=15, compat=2.0, visn=0):  # reutrn new_nxk_prob
    import pydensecrf.densecrf

    embedding *= 100
    dcrf = pydensecrf.densecrf.DenseCRF(prob.shape[0], prob.shape[-1])
    unary_energy = -np.log(np.ascontiguousarray(prob.T).clip(1e-5, 1))
    dcrf.setUnaryEnergy(unary_energy)
    dcrf.addPairwiseEnergy(
        np.ascontiguousarray(embedding.T),
        compat=compat,
        normalization=pydensecrf.densecrf.NORMALIZE_SYMMETRIC,
        kernel=pydensecrf.densecrf.DIAG_KERNEL,
    )

    Q, tmp1, tmp2 = dcrf.startInference()
    for idx in range(itern):
        dcrf.stepInference(Q, tmp1, tmp2)
        if __name__ == "__main__" and visn and not idx % (itern // visn):
            kl = dcrf.klDivergence(Q) / (H * W)
            cluster_idx = np.argmax(Q, axis=0)
            cluster_idx2d = to_2d(cluster_idx)
            Q2d = to_2d(np.array(Q).T)
            print("klDivergence", kl)
            boxx.show(cluster_idx2d, Q2d)
    new_prob = np.array(Q).T
    boxx.mg()
    return new_prob


def graph_conv_by_affinity(feat, affinity, momenta=0.6, order=1, itern=10, visn=0):
    if isinstance(affinity, np.ndarray):
        affinity = scipy.sparse.coo_matrix(affinity)

    sorted_idx = np.argsort(affinity.row)
    row = affinity.row[sorted_idx]
    col = affinity.col[sorted_idx]
    data = affinity.data[sorted_idx]
    assert data.min() >= 0 and data.max() <= 1

    indirect_rcds = []
    for order_idx in range(1, order):
        indirect_rcds.clear()
        row_to_cd = np.array([None] * affinity.shape[0])
        unique_row, (split_col, split_data) = np_group_by(row, [col, data])
        # (n, [col(neighbor*int), data(neighbor*float)])
        _cds = np.zeros((affinity.shape[0],), dtype=object)
        _cds[:] = list(zip(split_col, split_data))
        row_to_cd[unique_row] = _cds
        order_rate = 1
        for row_idx, (direct_col, direct_data) in enumerate(row_to_cd):
            indirect_cds = row_to_cd[direct_col]
            for direct_idx, (indirect_col, indirect_data) in enumerate(indirect_cds):
                weight = direct_idx
                indirect_rcds.append(
                    (
                        np.ones_like(indirect_col) * row_idx,
                        indirect_col,
                        indirect_data * direct_data[direct_idx] * order_rate,
                    )
                )

        indirect_row = np.concatenate([rcd[0] for rcd in indirect_rcds])
        indirect_col = np.concatenate([rcd[1] for rcd in indirect_rcds])
        indirect_data = np.concatenate([rcd[2] for rcd in indirect_rcds])

        row = np.append(row, indirect_row)
        col = np.append(col, indirect_col)
        data = np.append(data, indirect_data)

        # remove replica
        unique_edge, groups = np_group_by(np.array([row, col]).T, data)

        row = unique_edge[:, 0]
        col = unique_edge[:, 1]
        data = np.array([gp.max() for gp in groups])

    unique_row, counts = np.unique(row, return_counts=1)
    compact_n = max(counts)
    compact_col_idx = np.concatenate(list(map(np.arange, counts))) % compact_n
    compact = np.zeros((affinity.shape[0], compact_n), affinity.dtype)

    col_idx_to_compact_shape = -np.ones_like(compact, np.int32)
    col_idx_to_compact_shape[row, compact_col_idx] = col

    new_feat = feat.copy()
    for idx in range(itern):
        # compute by batchs for save memory
        batch = 200000 * 20 * 90 // compact_n // feat.shape[-1]
        new_feat_ = new_feat.copy()
        for batch_idx in tqdm(range(int(np.ceil(affinity.shape[0] / batch)))):
            slice_on_row = boxx.sliceInt[batch_idx * batch : batch_idx * batch + batch]
            # feat_ = feat[slicee]
            slice_on_coo = (batch_idx * batch <= row) & (
                row < batch_idx * batch + batch
            )
            col_idx_to_compact_shape_ = col_idx_to_compact_shape[slice_on_row]
            feat_in_compact_shape_ = feat[col_idx_to_compact_shape_]

            compact[row[slice_on_coo], compact_col_idx[slice_on_coo]] = data[
                slice_on_coo
            ]
            compact_ = compact[slice_on_row]
            weight = compact_ / compact_.sum(-1, keepdims=True)
            aggregation = (weight[..., None] * feat_in_compact_shape_).sum(-2)
            new_feat_[slice_on_row] = new_feat[slice_on_row] * momenta + aggregation * (
                1 - momenta
            )
        new_feat = new_feat_

        if __name__ == "__main__" and visn and not idx % (itern // visn):
            new_feat2d = to_2d(new_feat)
            boxx.show(new_feat2d.argmax(0), new_feat2d[1], feat2d[1])
    boxx.mg()
    return new_feat


def test_graph_conv_complexity():
    N = 200000
    K = 100
    C = 90
    data = np.random.rand(N * K).astype(np.float32)
    row = np.linspace(0, N - 0.1, N * K).astype(np.int32)
    col = np.random.randint(0, N, (N * K))
    affinity = scipy.sparse.coo_matrix((data, (row, col)), shape=(N, N))
    feat = np.linspace(0, 1, N)[:, None].astype(np.float32)
    feat = np.concatenate([feat] * C, -1)
    new_feat = graph_conv_by_affinity(
        feat, affinity, momenta=0.5, order=1, itern=1, visn=0,
    )
    boxx.loga(feat)
    boxx.loga(new_feat)
    boxx.mg()


if __name__ == "__main__":
    from boxx import *

    # test_graph_conv_complexity(); 1 / 0

    H, W, NLABELS = 55, 40, 2
    # H, W = np.int32(np.array([H, W]) * 1.3)
    print("H, W, H*W:", H, W, H * W)
    to_2d = lambda x, hw=(H, W): x.T.reshape(-1, *hw)
    to_1d = lambda x: x.reshape(x.shape[0], -1).T

    ys, xs = np.mgrid[:H, :W]
    feat_ = 0.8 - ((xs / W - 0.5) ** 2 + (ys / H - 0.5) ** 2) ** 0.5
    feat_[H * 4 // 9 : H // 2, W * 4 // 9 : W // 2,] *= 0.6
    feat_ = norma(feat_)
    feat2d = np.array([1 - feat_, feat_]).astype(np.float32)
    feat = to_1d(feat2d)
    prob = softmax(feat)
    prob2d = to_2d(prob)

    embedding_pos2d = np.array([xs / W, ys / H]).astype(np.float32)
    embedding_pos = to_1d(embedding_pos2d)

    print("带有噪音的 feature 可视化")
    boxx.show(feat2d[1], feat2d.argmax(0))

    if "spectral_clustering" and 10:
        embedding = np.concatenate((embedding_pos, feat), -1)

        affinity = ((embedding_pos[None] - embedding_pos[:, None]) ** 2).sum(-1) ** 0.5
        affinity += ((feat[None] - feat[:, None]) ** 2).sum(-1) ** 0.5 * 2
        affinity = norma(-affinity)

        cluster_idx = spectral_clustering(affinity, 2)
        cluster_idx2d = to_2d(cluster_idx)
        print("spectral_clustering")
        boxx.show(cluster_idx2d)

    if "desnse_crf" and 10:
        embedding = embedding_pos
        new_prob = desnse_crf(prob, embedding, itern=15, visn=0)
        new_prob2d = to_2d(new_prob)
        print("desnse_crf: result mask, new_prob2d, prob2d")
        boxx.show(new_prob2d.argmax(0), new_prob2d[1], prob2d[1])

    if "graph_conv" and 10:
        embedding = embedding_pos
        dis = (
            (xs.flatten()[None] - xs.flatten()[:, None]) ** 2
            + (ys.flatten()[None] - ys.flatten()[:, None]) ** 2
        ) ** 0.5
        mask = dis < H / 30
        affinity = embedding_to_affinity(embedding)
        affinity[np.eye(len(feat), dtype=np.bool)] = 0
        affinity[~mask] = 0
        print(mask.sum() / affinity.size)

        new_feat = graph_conv_by_affinity(
            feat, affinity, momenta=0, order=3, itern=3, visn=0,
        )

        new_feat2d = to_2d(new_feat)
        print("graph_conv: result mask, new_feat2d, feat2d")
        boxx.show(new_feat2d.argmax(0), new_feat2d[1], feat2d[1])
    embedding2d = to_2d(embedding)
