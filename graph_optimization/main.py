#!/usr/bin/env python3

import boxx
from boxx import np

# import scipy.linalg.decomp
import sklearn.cluster


def norma(feat):
    minn = feat.min()
    return (feat - minn) / (feat.max() - minn)


def softmax(feat):
    exp = np.exp(feat)
    prob = exp / exp.sum(-1, keepdims=True)
    return prob


def embedding_to_affinity(embedding, distance_type="l2"):
    # embedding shape == 100, 82, 4  复杂度和内存就受不了了
    is_2d = embedding.ndim == 3
    if is_2d:
        embedding = to_1d(embedding)
    if distance_type == "l1":
        affinity = 2 - (embedding[None] - embedding[:, None]).sum(-1)
    if distance_type == "l2":
        affinity = 2 - ((embedding[None] - embedding[:, None]) ** 2).sum(-1) ** 0.5
    # if distance_type == "none_linear":
    #     affinity = 2 - ((embedding[None] - embedding[:, None]) ** 2).sum(-1) ** 0.5
    return affinity


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
    dcrf.addPairwiseEnergy(np.ascontiguousarray(embedding.T), compat=compat)

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


if __name__ == "__main__":
    H, W, NLABELS = 40, 30, 2
    to_2d = lambda x, hw=(H, W): x.T.reshape(-1, *hw)
    to_1d = lambda x: x.reshape(x.shape[0], -1).T

    ys, xs = np.mgrid[:H, :W]
    feat_ = 0.8 - ((xs / W - 0.5) ** 2 + (ys / H - 0.5) ** 2) ** 0.5
    feat_[H * 4 // 9 : H // 2, W * 4 // 9 : W // 2,] *= 0.6
    feat_ = norma(feat_)
    feat2d = np.array([1 - feat_, feat_]).astype(np.float32)
    feat = to_1d(feat2d)

    embedding_pos2d = np.array([xs / W, ys / H]).astype(np.float32)
    embedding_pos = to_1d(embedding_pos2d)

    print("带有噪音的 feature 可视化")
    boxx.show(feat2d[1], feat2d.argmax(0))
    if "spectral_clustering" and 10:
        embedding = np.concatenate((embedding_pos, feat), -1)
        # affinity = embedding_to_affinity(embedding, "none_linear")
        affinity = ((embedding_pos[None] - embedding_pos[:, None]) ** 2).sum(-1) ** 0.5
        affinity += ((feat[None] - feat[:, None]) ** 2).sum(-1) ** 0.5 * 2
        affinity = norma(-affinity)

        cluster_idx = spectral_clustering(affinity, 2)
        cluster_idx2d = to_2d(cluster_idx)
        print("spectral_clustering")
        boxx.show(cluster_idx2d)

    if "desnse_crf" and 10:
        embedding = np.concatenate((embedding_pos ** 3, feat * 0), -1)
        # embedding = np.concatenate((embedding_pos * 10, feat * 0), -1)
        prob = softmax(feat)
        new_prob = desnse_crf(prob, embedding, itern=15, visn=0)
        new_prob2d = to_2d(new_prob)
        prob2d = to_2d(prob)
        print("desnse_crf: result mask, new_prob2d, prob2d")
        boxx.show(new_prob2d.argmax(0), new_prob2d[1], prob2d[1])

    embedding2d = to_2d(embedding)
