# Background

Clustering high dimensional data points in challenging! The regular techniques such as k-means, DBSCAN, HDBSCAN, Agglomerative Clustering all suffer from the non-intuitive properties of metrics in high dimension. However, experiments have shown that dimension reduction techniques applied before running a clustering algorithm can really improve on the results obtained. More precisely, using UMAP to reduce dimensionality followed by HDBSCAN to identify clusters perform reasonably well in identifying the underlying clusters. In general, it performs better than using PCA for dimensionality reduction. The following figure shows results from a study of the [Pendigits data set](http://odds.cs.stonybrook.edu/pendigits-dataset/). We show clustering accuracies of five clustering algorithms on the original high dimensional points and on low dimensional points obtained using PCA or UMAP. The accuracies are measured with the adjusted Rand index (ARI) and the adjusted mutual information (AMI). 

![](https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/dim_red_4_clustering.png)

This same experiment has been run on five high dimensional data sets, all of which are images. The two data sets COIL and Buildings are obtained via pictures of objects at different angles. Hence, the distance between images of the same object/same cluster in opposite direction (180 degrees) can be quite large, but the distance between the same object at similar angles is small. It is important to note that the Buildings data is more challenging then all of the other data sets. 

| Name     | Sample | Clustering Accuracy (Blue: None, Orange: PCA, Green: UMAP) | 
| --- | :-: | --- | 
| Pendigits   |  <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/examples_Digits_6_0.png" width="200" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/dimred_pendigits.png" width="400" />  |
| Coil | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/coil_bw.jpg" width="300" /> | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/dimred_coil.png" width="400" />  |
| MNIST | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/320px-MnistExamples.png" width="200" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/dimred_mnist.png" width="400" />  |
| USPS  | <img src="https://production-media.paperswithcode.com/datasets/USPS-0000001055-6cd416b0_D96Rryg.jpg" width="200" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/dimred_usps.png" width="400" />  |
| Buildings | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/buildings.png" width="300" /> | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/dimred_buildings.png" width="400" />  |

As mentioned earlier, comparisons were run on all of the five data sets and the results are displayed in the table above. The saturation of the colour indicates the proportion of points that were actually clustered and not identified as outliers. One problem we encounter with the original data or data reduced via PCA, is that the density-based clustering algorithms, DBSCAN and HDBSCAN, identify a large proportion of the points as noise points. UMAP doesn't suffer from this behaviour as much. Moreover, the UMAP representation yields overall better results with any of the clustering algorithms, but generally performs better with HDBSCAN. 

Even though this problem is hard, in practice the pipeline UMAP-HDBSCAN has proven to be very successful. But for this particular task, is the function UMAP aims at optimizing the right one? UMAP uses a gradient descent to optimize the cross entropy of some weighted version of the nearest neighbour graphs in the two spaces: before and after reduction. Is all of this work required to succeed in the clustering task? Would it be possible to reduce the complexity without hurting the performance? Even better, could we improve on the performance? Here we document the empirical study whose goal is to guide us in answering these questions. We use UMAP-HDBSCAN as our baseline model.

# UMAP's gradient descent

UMAP reduces the points to a space whose dimension is specified by the user. It uses a gradient descent to move points around in order to optimize the cross entropy between weighted nearest neighbour graphs obtained in the original space and in the reduced space, see [the paper](https://arxiv.org/pdf/1802.03426.pdf) for complete details. The initial position of the points can either be random or obtained by the spectral decomposition of this weighted nearest neighbor graph (is this right?). The first question we can ask ourselves: would the initial position of the points be sufficient to identify the clusters? If not, how many epochs of the gradient descent are required to converge to stable clusters?
|   | Impact of UMAP's nepoch parameter  |   |
| :-: | :-: | :-: | 
| Pendigits   |  USPS | Coil  |
| <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/nepochs_pendigits.png" width="400" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/nepochs_usps.png" width="400" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/nepochs_coil.png" width="400" />  |
| MNIST  | Buildings |
| <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/nepochs_mnist.png" width="400" /> | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/nepochs_buildings.png" width="400" /> |

In this experiment, all the initializations were obtained via a spectral decomposition. As we can see, the cluster qualities do not improve so much after 50-100 epochs. It also shows that performing HDBSCAN on a spectral decomposition of the UMAP graph does not recover the partitions well. So the gradient descent does improve the results of the clustering task, but does not need too many steps. Could we design an early stopping criteria for the clustering task? Is all of the information relevant for clustering captured in the UMAP graph? Could we turn this clustering algorithm into a graph clustering algorithm?

# Clustering neighbour graphs

The UMAP + HDBSCAN pipeline does not use the high dimensional points' positions per se: it entirely relies on pairwise distances and, more specifically, on a **weighted nearest neighbour graph representation** of the data. The weights are computed on the directed nearest neighbour graph first and then symmetrized. The asymetric weights take into account the distance between the edge endpoints, the distance to the closest neighbour and a normalization factor that controls the distribution of the distances of neighbours (and makes those distributions much more similar to what we see in low dimensional spaces). Our question: can this graph be used directly for cluster identification?

As a first experiment, we evaluate the performance of standard graph clustering algorithms on this weighted graph. We actually performed clustering on two weighted graphs derived from the original data and one derived from the points reduced by UMAP in the lower dimensional space. From the high dimensional space, we use a "fuzzy union" graph, which is equivalent to making the directed nearest neighbour undirected and a "fuzzy intersection" graph, which limits the edge set to reciprocal directed edges (a parameter in UMAP allows to make this decision, even in a smooth fashion).  We run a set of standard graph clustering algorithms on these graphs: Louvain, Leiden, Label Propagation and ECG. The algorithms Louvain and Leiden were both run on the weighted and unweighted versions of the graph. They both had very similar accuracies, we omitted Louvain in the following figures.

|   | Graph partition accuracies  |   |
| :-: | :-: | :-: | 
| Pendigits   |  USPS | Coil  |
| <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_pendigits.png" width="400" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_usps.png" width="400" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_coil.png" width="400" />  |
| MNIST  | Buildings | Legend |
| <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_mnist.png" width="400" /> | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_buildings.png" width="400" /> | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_legend.png" width="150" /> |

Apart from the Buildings data set, **all** the graph clustering algorithms we have tested do not perform as well as our baseline model and yield better results with the fuzzy union graph. The Buildings data set is known to be more challenging than the others, It is promising to see that the graph clustering does improve results on this task, but results are still not very good (with ARI less than 0.5). 

### Relation between high and low dimension
Interestingly, clustering the nearest neighbour graph of the low dimensional points obtained with UMAP, points on which HDBSCAN is able to recover the clusters, do not yield good results at all. We have studied how similar the nearest neighbour graphs are in high and low dimension, before and after UMAP. Neighborhood preservation is commonly studied for evaluating dimensionality reduction techniques. It is done by looking at the distribution of neighborhood Jaccard - or Kendall Tau - similarities over all points. For all of these data sets (but the Coil data set), the distributions are centered around a 0.6 value, indicating that, on average, only 60% of neighbours are preserved when reducing dimension (the Coil is better). 60% does not seem to be a lot, so how come is this reduction technique still very successful in retrieving the ground truth clusters?

### Noise points
HDBSCAN does classify some points as being noise points. In our baseline evaluation, these points are discarded whereas they are not in the graph clustering evaluation. This can make the comparison unfair, favouring our baseline. In order to run a fair comparison, we have retrieved all noise points identified by HDBSCAN from the graph before performing the clustering. This experiment showed a very marginal improvement for the graph clustering algorithms. Nothing important enough to explain its poor performance.

## Graph partition properties
Performing graph clustering algorithms on the UMAP graph does not recover the high dimensional clusters. Why is this the case? If we look into the the ground truth partition and the ones obtained by the graph clustering algorithms, say Leiden, how do they compare? We will compare their sizes, the number of _external_, i.e. edges that are connect the parts and their modularity. 


| | | | | | | | | | | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
|   <th colspan=3> Size </th><th colspan=3> External edges </th><th colspan=3> Modularity </th>| 
| Name    | G. truth | Leiden | Baseline| G. truth | Leiden | Baseline|G. truth | Leiden | Baseline|
| Pendigits   | 10 | 12| 10| 5.7  |  4.7 | 4.88 | 0.727 | 0.896 | 0.854 |
| Coil  | 21 | 27 | 20 | 5.5 | 2.3 | 1.6 |  0.894 | 0.932 | 0.91 | 
| MNIST |  10 | 14 | 12  | 7.14 | 5.88 | 6.21 | 0.828 | 0.858 | 0.85 | 
| USPS |  11 | 12  | 9 | 6.33 | 5.34  | 3.83 | 0.828 | 0.861 | 0.84 |
| Buildings | 41 | 57 | 100 | 24.2 | 7.2 | 14.3 | 0.727 | 0.898 | 0.882 |

In these five experiments, the partitions produced by Leiden all have higher modularity than the UMAP-HDBSCAN or the ground truth partitions. The Leiden partitions are also the ones with the smallest number of external edges, i.e. edges connecting the different parts. This is exactly what graph clustering algorithms are aiming for, but yet, the partitions found do not correspond to the high dimensional clusters. Modularity is clearly not the desired objective function for this problem as it focuses on edges. Could it be that the preservation by partitions of higher order cliques - structures that correspond to higher order simplices of the [Vietoris-Rips complex](https://en.wikipedia.org/wiki/Vietoris%E2%80%93Rips_complex) - is what should be optimized? The empirical study of the number of cliques perserved by the ground truth vs. Leiden partitions show that this is not what we should be optimizing either. Surprisingly, the ground truth partition breaks up a larger number of cliques of size 3 (triangles), 4 or 5 than Leiden in all five cases. If the Leiden partitions tend to have a very large cluster with a number of smaller ones, that could explain an increase in the number of cliques preserved. However, this is not what we observe in practice. Leiden do produce a larger number of clusters, some of which are of smaller sizes than the ground truth parts, leading towards a partition refinement which should break up more cliques - not preserve more cliques. It does not have any giant part as shown in the barplots below.

|   | Cluster sizes  |   |
| :-: | :-: | :-: | 
| Pendigits   |  USPS | Coil  |
|  <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_pendigits_sizes.png" width="400" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_usps_sizes.png" width="400" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_coil_sizes.png" width="400" />  |
| MNIST  | Buildings | Legend |
| <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_mnist_sizes.png" width="400" /> | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/graphClust_buildings_sizes.png" width="400" /> |  |
 

# Discriminant properties

By limiting our objective function to counting edges or higher order structures that are preserved by the final partition is not enough. However, we are dealing here with metric graphs meaning that each vertex in our graphs correspond to a (high dimensional) point in a metric space. We can therefore very naturally define functions from edges to the set of reals: distance in original space, rank as a neighbour (for directed version), UMAP weight, average/sum of distances to closest neighbour of the edge's endpoints (rho in UMAP documentation), average/sum of normalizing factor of the edge's endpoints (sigma in UMAP documentation), etc. These functions can even be pushed to higher order structures (triangles, ...) by aggregating the edges' values. Moreover, other edge properties based on the graph topology can also be included: number of distinct triangles containing this edge, edge betweenness, etc. Can we identify edge properties that would allow to distinguish between internal and external edges? Could the vertex partition problem be transformed into a edge pruning problem yielding the desired connected components?

Here, we study the two distributions (internal vs. external edges) of natural edge values: high dimensional distance, UMAP edge weight, sum of rho's, sum of sigma's, number of triangles and edge betweenness.

|   |  Edge property distributions |   |
| :-: | :-: | :-: | 
| Pendigits   |  USPS | Coil  |
| <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/edge_properties_pendigits.png" width="400" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/edge_properties_usps.png" width="400" />  | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/edge_properties_coil.png" width="400" />  |
| MNIST  | Buildings | Legend |
| <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/edge_properties_mnist.png" width="400" /> | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/edge_properties_buildings.png" width="400" /> | <img src="https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/edge_properties_legend.PNG" width="400" /> |

Note that all the histograms above are in a log scale for their y-axes. As can be seen from these plots, some of the edge properties have different distributions when conditioned on internal (blue) vs. external (red) edge status. Recall that the internal edges form the dominant set containing over 92% of the edges in all cases. Apart from the sum of sigmas, all of the conditional distributions have smaller supports for the external edges rather than internal edges. The internal edge distributions have their leftmost or rightmost tails outside the intersection of supports, making these edges separable from the rest. However, if we restrict the domains to the intersection of the supports, the external edges' distributions attain their peak, but, at the same time, the internal edges still dominate in terms of frequencies - direct consequence of the unbalanced nature of the internal/external distribution. This phenomenon is not as pronounced when we look at the distributions of the low dimensional distances. The low dimensional distances do separate better the two edge types as we can see in the plot below.


![](https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/edge_lowdim_dist.png)

We have also explored properties of the directed edges of the k-NN graphs. We have looked at the two distributions (internal vs. external) of the "neighbour rank" of each directed edge. The conclusion is similar if not worst: the supports of the two distributions are identical in all cases  

The next question is could we still use these edge properties - or a combination of those - to prune edges from the graph in order to recover the true high dimensional clusters? Obviously the internal vs. external edge property distributions overlap. However, graph connected components or other graph clustering algorithms can be quite robust to pruning internal edges if their is enough redundancy.

## Directed graph edge properties



## Single linkage using edge properties

## Edge classifier from edge properties

## Triangle properties