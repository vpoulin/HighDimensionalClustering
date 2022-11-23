def construct_graph(raw_data, n_neighbors, metric='euclidean', keep_k = True):
    A, sigmas, rhos, dists = umap.umap_.fuzzy_simplicial_set(X=raw_data, 
                                                         n_neighbors=n_neighbors, 
                                                         random_state=0, 
                                                         metric=metric, 
                                                         return_dists=True)
    G = nx.from_scipy_sparse_matrix(A, edge_attribute='umap_weight')
    
    nx.set_edge_attributes(G, 0, 'highdim_dist')
    nx.set_edge_attributes(G, dists, 'highdim_dist')
    nx.set_node_attributes(G, dict(zip(G.nodes, sigmas)), 'sigma')
    nx.set_node_attributes(G, dict(zip(G.nodes, rhos)), 'rho')
    
    G_dir = G.to_directed()
    
    # compute neighbours' ranks
    node_dicts = dict()
    for n in G_dir.nodes():
        x = {(u,v):e for u,v,e in G_dir.out_edges(n, data='highdim_dist')}
        node_dicts[n] = {e:i for i,e in enumerate(sorted(x, key=x.get))}
    highdim_rank = {key:val for n,d in node_dicts.items() for key,val in d.items()}
    nx.set_edge_attributes(G_dir, highdim_rank, 'highdim_rank')
    if(keep_k):
        edges_rm = [(u,v) for u,v,e in G_dir.edges(data=True) if e['highdim_rank']>=(n_neighbors-1)]
        G_dir.remove_edges_from(edges_rm)
        
    bi_dir =  {(u,v):True for (u,v) in G_dir.edges() if ((v,u) in G_dir.edges())}
    nx.set_edge_attributes(G_dir, False, 'bi_directional')
    nx.set_edge_attributes(G_dir, bi_dir, 'bi_directional') 
    return(G_dir)

## Run on networkx graphs instead of igraph
def run_graph_clustering_algorithm(algo_list, G, weight = 'umap_weight'):
    n_points = G.number_of_nodes()
    clustering_results = dict()
    clustering_labels = dict()
    for algo in algo_list:
        #print(f"Running {algo}...")
        if algo == 'Louvain':
            clustering_results[algo] = community_louvain.best_partition(G)
        elif algo == 'Louvain \n+ weight':
            clustering_results[algo] = community_louvain.best_partition(G, weight=weight)
        elif algo == 'Label \nPropagation':
            lp_graph_list = list(label_propagation_communities(G)) 
            clustering_results[algo] = {v:com for com, comSet in enumerate(list(lp_graph_list)) for v in comSet}
        elif algo == 'ECG':
            clustering_results[algo] = ecg(G, ens_size=32).partition
        elif algo == 'Leiden':
            iG = ig.Graph.from_networkx(G)
            clustering_results[algo] = {i:v for i, v in enumerate(la.find_partition(iG, la.ModularityVertexPartition).membership)}
        elif algo == 'Leiden \n+ weight':
            iG = ig.Graph.from_networkx(G)
            clustering_results[algo] = {i:v for i, v in enumerate(la.find_partition(iG, la.ModularityVertexPartition, weights=weight).membership)} 
        clustering_labels[algo] = np.array([clustering_results[algo][i] for i in range(n_points)])
    return(clustering_results, clustering_labels)       


### Plot graphs with cluster colors

# node_dist_to_color = {
#     -1: "white",
#     0: "tab:brown",
#     1: "tab:red",
#     2: "tab:orange",
#     3: "tab:olive",
#     4: "tab:green",
#     5: "tab:blue",
#     6: "tab:purple",
#     7: "tab:cyan",
#     8: "tab:gray",
#     9: "tab:pink",
#     10: "darkgray",
#     11: "yellow",
# }

# degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
# dmax = max(degree_sequence)

# fig = plt.figure("Degree of a random graph", figsize=(8, 14))
# # Create a gridspec for adding subplots of different sizes
# axgrid = fig.add_gridspec(11, 4)

# ax3 = fig.add_subplot(axgrid[0:3, :])
# Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
# pos = nx.spring_layout(Gcc, seed=10396953)
# nx.draw_networkx_nodes(Gcc, pos, ax=ax3, node_size=20, 
#                        node_color=[node_dist_to_color[nd] for nd in digits.target])
# nx.draw_networkx_edges(Gcc, pos, ax=ax3, alpha=0.4)
# ax3.set_title(f"True classes ({max(digits.target)+1} classes)")
# ax3.set_axis_off()


# ax0 = fig.add_subplot(axgrid[3:6, :])
# Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
# pos = nx.spring_layout(Gcc, seed=10396953)
# nx.draw_networkx_nodes(Gcc, pos, ax=ax0, node_size=20, 
#                        node_color=[node_dist_to_color[nd] for nd in lv_graph_labels])
# nx.draw_networkx_edges(Gcc, pos, ax=ax0, alpha=0.4)
# ax0.set_title(f"Louvain clusters ({max(lv_graph_labels)+1} classes)")
# ax0.set_axis_off()

# ax4 = fig.add_subplot(axgrid[6:9, :])
# Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
# pos = nx.spring_layout(Gcc, seed=10396953)
# nx.draw_networkx_nodes(Gcc, pos, ax=ax4, node_size=20, 
#                        node_color=[node_dist_to_color[nd] for nd in hd_umap_labels])
# nx.draw_networkx_edges(Gcc, pos, ax=ax4, alpha=0.4)
# ax4.set_title(f"HDBSCAN-UMAP clusters ({max(hd_umap_labels)+1} classes)")
# ax4.set_axis_off()

# ax1 = fig.add_subplot(axgrid[9:, :2])
# ax1.plot(degree_sequence, "b-", marker="o")
# ax1.set_title("Degree Rank Plot")
# ax1.set_ylabel("Degree")
# ax1.set_xlabel("Rank")

# ax2 = fig.add_subplot(axgrid[9:, 2:])
# ax2.bar(*np.unique(degree_sequence, return_counts=True))
# ax2.set_title("Degree histogram")
# ax2.set_xlabel("Degree")
# ax2.set_ylabel("# of Nodes")

# fig.tight_layout()
# plt.show()

## On NETWORKX : this comes from teh UMAP_graph_stydy.ipynb
def enrich_graph_edge_properties(G, vertex_high_representation=None, vertex_low_representation=None, vertex_labels = dict(), edge_binary=True):
    
    all_triangles =  [(n,nbr) for n in G for nbr, nbr2 in itertools.combinations(G[n],2) if (nbr in G[nbr2] and n<nbr)]
    edge_triangles = collections.Counter(all_triangles)
    nx.set_edge_attributes(G, 0, 'nb_triangles')
    nx.set_edge_attributes(G, edge_triangles, 'nb_triangles')    
    
    edge_sigma_sum = {e:(sigmas[e[0]]+sigmas[e[1]]) for e in G.edges}
    nx.set_edge_attributes(G, edge_sigma_sum, 'sigma_sum')
    edge_rho_sum = {e:(rhos[e[0]]+rhos[e[1]]) for e in G.edges}
    nx.set_edge_attributes(G, edge_rho_sum, 'rho_sum')
        
    for name, labels in vertex_labels.items():
        if(edge_binary):
            edge_internal = {e:(1 if (labels[e[0]]==-1 or labels[e[1]]==-1)
                    else 1 if((labels[e[0]]!=-1) and labels[e[0]]==labels[e[1]]) 
                    else 0)
                    for e in G.edges}
        else:
            edge_internal = {e:(-1 if (labels[e[0]]==-1 or labels[e[1]]==-1)
                    else 1 if(labels[e[0]]==labels[e[1]]) 
                    else 0)
                    for e in G.edges}
        nx.set_edge_attributes(G, edge_internal, name)
        
    if(vertex_low_representation is not None):
        edge_lowdim_dist = {e:euclidean( vertex_low_representation[e[0]], vertex_low_representation[e[1]] )
                    for e in G.edges}
        nx.set_edge_attributes(G, edge_lowdim_dist, 'lowdim_dist')
        
    if(vertex_high_representation is not None):
        edge_highdim_dist = {e:euclidean( vertex_high_representation[e[0]], vertex_high_representation[e[1]] )
                    for e in G.edges}
        nx.set_edge_attributes(G, edge_highdim_dist, 'highdim_dist')
        
    return(G)

## Construct digraph from umap graph
def construct_graph(raw_data, n_neighbors, metric='euclidean', keep_k = True):
    A, sigmas, rhos, dists = umap.umap_.fuzzy_simplicial_set(X=raw_data, 
                                                         n_neighbors=n_neighbors, 
                                                         random_state=0, 
                                                         metric=metric, 
                                                         return_dists=True)
    G = nx.from_scipy_sparse_matrix(A, edge_attribute='umap_weight')
    
    nx.set_edge_attributes(G, 0, 'highdim_dist')
    nx.set_edge_attributes(G, dists, 'highdim_dist')
    nx.set_node_attributes(G, dict(zip(G.nodes, sigmas)), 'sigma')
    nx.set_node_attributes(G, dict(zip(G.nodes, rhos)), 'rho')
    
    G_dir = G.to_directed()
    
    # compute neighbours' ranks
    node_dicts = dict()
    for n in G_dir.nodes():
        x = {(u,v):e for u,v,e in G_dir.out_edges(n, data='highdim_dist')}
        node_dicts[n] = {e:i for i,e in enumerate(sorted(x, key=x.get))}
    highdim_rank = {key:val for n,d in node_dicts.items() for key,val in d.items()}
    nx.set_edge_attributes(G_dir, highdim_rank, 'highdim_rank')
    if(keep_k):
        edges_rm = [(u,v) for u,v,e in G_dir.edges(data=True) if e['highdim_rank']>=(n_neighbors-1)]
        G_dir.remove_edges_from(edges_rm)
    return(G_dir)