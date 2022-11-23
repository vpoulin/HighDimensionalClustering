import numpy as np
import pandas as pd
import scipy

# Produce the directed knn graph.
# Uses umap functions - so knn may be approximate
def knn_digraph(X, k, graph_type='ig'):
    import umap
    knn_indices, knn_dists, _ = umap.umap_.nearest_neighbors(
        X, n_neighbors=k, metric='euclidean', metric_kwds={},
        angular=True, random_state=0
    )

    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = umap.umap_.smooth_knn_dist(
        knn_dists,
        float(k),
        local_connectivity=float(1),
    )

    rows, cols, vals, dists = umap.umap_.compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists=True
    )

    A = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
        )
    A.eliminate_zeros()
    if(graph_type=='nx'):
        G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
        bi_dir =  {(u,v):True for (u,v) in G.edges() if ((v,u) in G.edges())}
        nx.set_edge_attributes(G, False, 'bi_directional')
        nx.set_edge_attributes(G, bi_dir, 'bi_directional') 
    else:
        G = ig.Graph.Weighted_Adjacency(A)
        for e in G.es:
            e["bi_directional"] = G.are_connected(e.target, e.source)

    return(G)

def graph_edge_class_from_labels(G, labels, attribute_name = 'internal'): 
    for e in G.es:
        e[attribute_name] = (labels[e.target] == labels[e.source])        
    return(G)



