import numpy as np
import torch
import faiss
import torch.nn as nn
from sklearn.cluster import KMeans
import math


def CIndex(pred, ytime_test, ystatus_test):
    N_test = ystatus_test.shape[0]
    ystatus_test = np.squeeze(ystatus_test)
    ytime_test = np.squeeze(ytime_test)
    theta = np.squeeze(pred)
    concord = 0.
    total = 0.
    eav_count = 0
    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        concord = concord + 1
                    elif theta[j] == theta[i]:
                        concord = concord + 0.5
                        eav_count = eav_count + 1
                        print("相等的对数")
                        print(eav_count)
    if total == 0:
        return 0
    # print("concord:", concord,  "total:", total)
    return concord / total


def AUC(pred, ytime_test, ystatus_test):
    N_test = ystatus_test.shape[0]
    ystatus_test = np.squeeze(ystatus_test)
    ytime_test = np.squeeze(ytime_test)
    theta = np.squeeze(pred)
    total = 0
    count = 0

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                # if ytime_test[i] < quantile_2 < ytime_test[j]:
                if ytime_test[i] < 365 * 1 < ytime_test[j]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        count = count + 1
                    elif theta[j] == theta[i]:
                        count = count + 0.5

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                # if ytime_test[i] < quantile_2 < ytime_test[j]:
                if ytime_test[i] < 365 * 5 < ytime_test[j]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        count = count + 1
                    elif theta[j] == theta[i]:
                        count = count + 0.5

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                # if ytime_test[i] < quantile_2 < ytime_test[j]:
                if ytime_test[i] < 365 * 10 < ytime_test[j]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        count = count + 1
                    elif theta[j] == theta[i]:
                        count = count + 0.5

    if total == 0:
        return 0

    return count / total


def run_kmeans(x, number_cluster, temperature, gpu):

    print('performing kmeans clustering')
    results = {'can2cluster': [], 'centroids': [], 'density': []}

    for seed, num_cluster in enumerate(number_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]  
        nd = x.shape[0]
        k = int(num_cluster)
        if nd <= k:
            break
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = gpu
        with torch.cuda.device(gpu):
            torch.cuda.empty_cache()
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        can2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for can, i in enumerate(can2cluster):
            Dcluster[i].append(D[can][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = temperature * density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda(gpu)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        can2cluster = torch.LongTensor(can2cluster).cuda(gpu)
        density = torch.Tensor(density).cuda(gpu)

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['can2cluster'].append(can2cluster)

    return results





