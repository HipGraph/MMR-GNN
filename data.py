import scipy
import numpy as np
from sklearn import cluster as sk_cluster


class Clustering:

    def cluster(x, alg="KMeans", n_clusters=3, return_obj=False, **kwargs):
        """
        Arguments
        ---------
        x : ndarray with shape=(N, D) for N samples on D variables
        alg : str
        kwargs : dict

        Returns
        -------
        cluster_index : ndarray with shape=(N,)

        """
        debug = kwargs.get("debug", 0)
        if alg == "KMeans":
            clustering = sk_cluster.KMeans(n_clusters, random_state=0, **kwargs).fit(x)
            cluster_index = clustering.labels_
        elif alg == "Agglomerative":
            clustering = sk_cluster.AgglomerativeClustering(n_clusters, compute_distances=True, **kwargs).fit(x)
            if debug:
                print(clustering.children_)
                print(clustering.distances_)
            cluster_index = clustering.labels_
        elif alg == "DBSCAN":
            clustering = sk_cluster.DBSCAN(min_samples=5).fit(x)
            cluster_index = clustering.labels_
        elif alg == "Random":
            rng = np.random.default_rng(0)
            clustering = rng
            cluster_index = rng.integers(0, n_clusters, size=len(x))
        else:
            raise NotImplementedError(alg)
        if return_obj:
            return clustering, cluster_index
        return cluster_index
        

class Probability:

    def get_lims(x, lims=[-3,3]):
        if lims is None:
            lims = np.stack((np.min(x, 0), np.max(x, 0)), -1)
        elif not isinstance(lims, np.ndarray):
            if isinstance(lims[0], str) and "%" in lims[0]: # percentile
                lims = [np.percentile(x, float(lims[0][:-1])), lims[1]]
            if isinstance(lims[1], str) and "%" in lims[1]: # percentile
                lims = [lims[0], np.percentile(x, float(lims[1][:-1]))]
        return lims

    def compute_histograms(x, bins=10, lims=None, **kwargs):
        """
        Arguments
        ---------
        x : ndarray with shape=(N, D) for N samples on D variables
        bins : int or ndarray with shape(B,)
        lims : None or ndarray with shape=(2,) or shape=(D, 2)

        Returns
        -------
        histograms : ndarray with shape=(D, B) for D variables

        """
        return_bins = kwargs.get("return_bins", False)
        debug = kwargs.get("debug", 0)
        N, D = x.shape
        n_bins = bins
        if not isinstance(bins, int): # bins defines bin edges
            n_bins = len(bins) - 1
        lims = Probability.get_lims(x, lims)
        if debug:
            print("Data.__init__", "bins =", bins, "lims =", lims)
        if not isinstance(lims, np.ndarray):
            lims = np.array(lims)
        if lims.ndim == 1:
            lims = np.tile(lims, (x.shape[-1], 1))
        elif not lims.ndim == 2 or not lims.shape == (D, 2):
            raise ValueError(lims.shape)
        histograms = np.zeros((D, n_bins))
        bin_edges = np.zeros((D, n_bins+1))
        for i in range(D):
            histograms[i,:], bin_edges[i,:], binnumber = scipy.stats.binned_statistic(
                x[:,i], x[:,i], "count", bins, lims[i,:]
            )
        if return_bins:
            return histograms, bin_edges
        return histograms

    def compute_correlations(x, **kwargs):
        """
        Arguments
        ---------
        x : ndarray with shape=(N, D) for N samples on D variables

        Returns
        -------
        corrs : ndarray with shape=(D, B) for D variables

        """
        debug = kwargs.get("debug", 0)
        N, D = x.shape
        corrs = np.corrcoef(x, rowvar=False, **kwargs)
        return corrs

    def transform(x, rep="histogram", **kwargs):
        if rep == "histogram":
            bins = kwargs.get("bins", 12)
            lims = kwargs.get("lims", [-3,3])
            x = Probability.compute_histograms(x, bins, lims, **kwargs)
        elif rep == "correlation":
            x = Probability.compute_correlations(x, **kwargs)
        else:
            raise NotImplementedError(rep)
        return x


def compute_histograms(x, bins=12, lims=[-3,3]):
    n_temporal, n_spatial, n_feature = x.shape
    hists = np.concatenate(
        [Probability.compute_histograms(x[:,:,i], bins=bins, lims=lims) for i in range(n_feature)], -1
    )
    return hists


def compute_correlations(x):
    n_temporal, n_spatial, n_feature = x.shape
    corrs = Probability.compute_correlations(np.reshape(x, (n_temporal, -1)))
    return np.reshape(corrs, (n_spatial, n_feature, n_spatial, n_feature))


def compute_representation(x, rep="histogram", **kwargs):
    # Handle arguments
    n_temporal, n_spatial, n_feature = x.shape
    bins, lims = kwargs.get("bins", 12), kwargs.get("lims", [-3,3])
    debug = kwargs.get("debug", 0)
    if debug:
        print("representation =", rep)
        print("bins =", bins)
        print("lims =", lims)
    # Start
    if rep == "histogram":
        x = compute_histograms(x, bins, lims)
        if debug:
            print("x =", x.shape)
            if debug > 1:
                print(x)
        x = np.reshape(x, (x.shape[0], -1))
        if debug:
            print("x =", x.shape)
            if debug > 1:
                print(x)
    elif rep == "correlation":
        x = compute_correlations(x)
        if len(n_feature) > 1:
            raise ValueError("Must select one feature for correlation. Received %d." % (n_feature))
        x = np.reshape(x, (x.shape[0], -1))
        if debug:
            print("x =", x.shape)
            if debug > 1:
                _x = np.round(x, 3)
                print(_x[...,:10])
    else:
        raise NotImplementedError(rep)
    return x


def cluster(x, alg="Agglomerative", n_clusters=3, rep="histogram", **kwargs):
    # Handle arguments
    debug = kwargs.get("debug", 0)
    # Start
    #  Derive representation of data x
    x = compute_representation(x, rep, **kwargs)
    #  Compute clusters
    clustering, cluster_index = Clustering.cluster(x, alg, n_clusters, return_obj=1)
    return x, clustering, cluster_index
