import umap
import igraph as ig
import numpy as np
import scipy

def get_graph_UX(X,
                 n_neighbors,
                 random_state,
                 metric,
                 local_connectivity=1.0):
    """
    A subset of the paramters from umap.umap_.fuzzy_simplicial_set. Note that this will set `apply_set_operations` to False.

    Returns
    --------
    The weighted UMAP graph U(X) as an igraph object with sigmas and rhos as vertex attributes
    """
    A, sigmas, rhos, dists = umap.umap_.fuzzy_simplicial_set(X, 
                                                             n_neighbors=n_neighbors, 
                                                             random_state=random_state, 
                                                             metric=metric, 
                                                             local_connectivity=local_connectivity,
                                                             return_dists=True,
                                                             apply_set_operations=False)
    G = ig.Graph.Weighted_Adjacency(A, 'directed')
    G.vs['sigmas'] = sigmas
    G.vs['rhos'] = rhos
    return G

def get_VX_from_UX(UX, set_op_mix_ratio=1.0):
    """
    From the matrix of U(X), get the igraph of V(X). 
    
    set_op_mix_ratio is the same as the parameter of umap.umap_.fuzzy_simplicial_set. That is:
    
    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.
    """
    result = UX
    transpose = result.transpose()

    prod_matrix = result.multiply(transpose)

    result = (
        set_op_mix_ratio * (result + transpose - prod_matrix)
        + (1.0 - set_op_mix_ratio) * prod_matrix
    )
    G = ig.Graph.Weighted_Adjacency(result, 'undirected')
    return G


def adjacency_to_distance_matrix(M, inf_size=1000, noise_factor=0.1, random_seed=42):
    """
    Take an adjacency matrix and turn it into a distance matrix where there is distance 1 when there is an edge and otherwise distance is inf_size. 
    Also perturbs the distances with a random noise factor to simulate randomly breaking the tiebreak for equidistant points.
    
    Define a noise_factor to use to create the random noise.
    
    Note: This assumes that the adjacency matrix is a binary matrix.
    """
    rng = np.random.default_rng(random_seed)
    N = M.copy()
    N[N < noise_factor] = inf_size
    N = N +  ((rng.random((N.shape[0], N.shape[1])) *noise_factor) - (noise_factor/2))
    return N

def randomly_remove_edges(M, proportion, kind="undirected", random_seed=42):
    '''
    Given an adjacency matrix M, keep only a given portion of entries. If undirected, assume input and output matrix is symmetric.
    '''
    rng = np.random.default_rng(random_seed)
    if kind=="directed":
        entries = [(i, j) for i, j in zip(*scipy.sparse.csr_matrix(M).nonzero())]
    elif kind=="undirected":
        entries = [(i, j) for i, j in zip(*scipy.sparse.csr_matrix(M).nonzero()) if i < j]
    else:
        raise ValueError(f"{kind} must be one of 'directed' or 'undirected'")
    choices = rng.choice(np.arange(len(entries)), size=int(proportion*len(entries)), replace=False)
    row_ind, col_ind = list(zip(*[entries[x] for x in choices]))
    if kind=="directed":
        data = np.ones(len(choices))
        mat = scipy.sparse.csr_matrix((data, (list(row_ind), list(col_ind))), shape=M.shape)
    elif kind=="undirected":
        data = np.ones(len(choices)*2)
        mat = scipy.sparse.csr_matrix((data, (list(row_ind)+list(col_ind), list(col_ind)+list(row_ind))), shape=M.shape)
    return mat