# [Notebook 01](./01-DimRed_comparison.ipynb) : High dimensional clustering comparison

This notebook has been stollen from [here](https://gist.github.com/lmcinnes/24ed5c22c80125be5133811d677eae7b). Note that you can see how to download all datasets from this notebook.

The goal of notebook 01 is to provide a basic template walkthrough of obtaining and preparing a number of (simple) high dimensional datasets that can reasonably used to clustering evaluation. The datasets chosen have associated class labels that *should* be meaningful in terms of how the data clusters, and thus we can use label based clustering evaluation such as ARI and AMI to determine how well different clustering approaches are performing.

The datasets:
* Pendigits is the smallest, with only 1797 samples, and is only 64 dimensional: 8x8 images of digits.
* MNIST provides a good basic scaling test with 70,000 samples of handwritten digits in 784 dimensions.
* COIL-20 was collected by the Center for Research on Intelligent Systems at the Department of Computer Science, Columbia University. The database contains grayscale images of 20 objects. The objects were placed on a motorized turntable against a black background and images were taken at pose internals of 5 degrees. So, 72 images of the same object from different angles for each class.
* The USPS dataset refers to numeric data obtained from the scanning of handwritten digits from envelopes by the U.S. Postal Service. The original scanned digits are binary and of different sizes and orientations; the images here have been deslanted and size normalized, resulting in 16 x 16 grayscale images (Le Cun et al., 1990). Almost 10,000 samples and 256 dimensions.
* Buildings are images of  41 buildings under rotatins. A total of 4178 images.

# [Notebook 02](./02-GraphClustering_on_UMAP_graph.ipynb) : High dimensional clustering via graph clustering

The goal of this notebook is to compare different graph clustering algorithms on the UMAP graphs (fuzzy intersection, fuzzy union, with and without edge weights). Fuzzy intersection limits the k-NN graph to bi-directional edges whereas the fuzzy union considers the directed graph yield by k-NN graph as undirected. The weights used are the UMAP weights.

We have run standard graph clustering algorithms: Louvain, Leiden, ECG and label propagation.
Results show that Leiden and Louvain are the two clustering techniques that perform the best for this task. But it is generally not as good as applying HDBSCAN on the low dimension vectors obtained from UMAP.

We have also tested if the difference in performance between graph clustering algorithms and HDBSCAN could come from the fact that HDBSCAN has a "noise" category, which reduces the data set size used for evaluating the technique. Removing noise, either from evaluation or from the dataset, does improve the performance of all algorithms, but it does not change the how the algorithms compare to one another.

# [Notebook 03](./03-UMAP_graph_study.ipynb) : Characterizing edges

The goal of this notebook is to characterize edges of the UMAP graph with respect with their position (internal/external) in the groundtruth partition. It is also to see if the edges misclassified by Leiden/HDBSCAN partitions have particularities. The edge features studied include their endpoints' rho and sigma values, the number of supporting triangles, etc. 

## [Notebook 06](./06-KNN_graph_study.ipynb) : k-NN graph : Ranks of edges

The k-NN graphs we get from low dimensional vs. high dimensional data representation differ. Here we look at how much they differ. To measure the similarity between a point's neiborhood before and after dimensionality reduction (UMAP), we use the truncated-Kendall Tau and the Jaccard similarity. We use a different neighborhood size for each dataset: we use the $k$ value that was used in the initial clustering experiment (yielding good UMAP+HDBSCAN partition).

The distributions of the neighborhood similarities over the set of vertices of these two measures are really similar. However, the distributions's shape differ quite significantly from one dataset to the other. Consequence of the neighborhood sizes and of the dataset properties? 

Would the ranks of the edges incident to a vertex - ranks in terms of different measures (high dim distance, low dim distance and UMAP weight) - be indicative of the status (internal/external) of the edge in the ground-truth clustering?
The high dimensional distance ranks and the UMAP weights ranks are both indicative of the edge status proportions in the ground truth partition. However, the low dimensional distance ranks are a lot more indicative.

# [Notebook 04](./04-cliques_preservation.ipynb) : Higher dimensional simplices preservation

Here, we explore how the higher dimensional simplices are preserved within parts of a partition. We compare the number of simplices preserved in the ground truth partition with the number preserved via Leiden.

Surprisingly, although Leiden generally yields a larger partition size - so a finer partition - than the ground truth partition, it tends to preserve a larger number of simplices. This implies that Leiden's result is not a refinement of the ground truth: it groups points differently and preserves more of the simplices structure than the ground truth.

Moreover, the Leiden partition (obtained on the kNN in high dimension) also preserves more of the low-dimensional simplices than the ground-truth partition does.

## [Notebook 05](./05-weighted_clique_preservation.ipynb) : Higher dimensional simplices : with weights?

Here we look at the minimum umap weights of the simplices. We want to see if simplices that are internal to ground truth parts have higher minimum weights than simplices that are external (split by ground truth partition).

Edges that are part of internal simplices (internal w.r.t. the ground truth partition) have higher minimal UMAP weights. However, the distributions of internal vs. external simplex weights are not easily separable: both are skewed towards small values.

# [Notebook 07](./07-HDBSCAN_on_UMAP_graph.ipynb) : Running HDBSCAN on partial distance matrices?

Here, trying to be lazy and get as much as we can out of HDBSCAN without understanding how it behaves on sparse matrices really. What can we get from HDBSCAN on a sparse distance matrix having non-zero entries only between high dimensional k-NN data pairs? 
* First we cheat: the distance values are the low dimensional distances (after dimension reduction with UMAP). If we use a small number of NNs (same as for the UMAP projection say), running HDBSCAN on the sparse matrix does not provide good clusters. The larger the number of NNs - the larger the number of non-zero values - the closer we get to HDBSCAN result on UMAP reduction. I am still surprised that by *cheating* a lot and providing a kNN graph (with large k value) of the low dim distances, we do not get similar results as HDBSCAN on the low dim points.
* from UMAP weighted adjacency matrix with unknown values encoded as infinite values. - I actually don't know how to turn the UMAP weight into a reasonable distance. I just used the inverse. (doesn't work!)


# [Notebook 08](./08-UMAP-nepochs.ipynb) : What precision does UMAP need for good clustering?

We study the impact of UMAP's parameter n_epochs on the HDBSCAN clustering task. We wish to see what portion of the work performed by UMAP is needed for enabling cluster identification.
