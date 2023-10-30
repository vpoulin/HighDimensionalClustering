High Dimensional Clustering
==============================
_Author: vpoulin_

Clustering high dimensional data points in challenging! The regular techniques such as k-means, DBSCAN, HDBSCAN, Agglomerative Clustering all suffer from the non-intuitive properties of metrics in high dimension. However, experiments have shown that dimension reduction techniques applied before running a clustering algorithm can really improve on the results obtained. More precisely, using UMAP to reduce dimensionality followed by HDBSCAN to identify clusters perform reasonably well in identifying the underlying clusters. In general, it performs better than using PCA for dimensionality reduction. The following figure shows results from a study of the [Pendigits data set](http://odds.cs.stonybrook.edu/pendigits-dataset/). We show clustering accuracies of five clustering algorithms on the original high dimensional points and on low dimensional points obtained using PCA or UMAP. The accuracies are measured with the adjusted Rand index (ARI) and the adjusted mutual information (AMI).

![](https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/dim_red_4_clustering.png)

We explore and try to make sense of these results via some initial experiments in the following notebooks.

## [Notebook 01](01-DimRed_comparison.ipynb) : High dimensional clustering comparison

This notebook has been copied from [here](https://gist.github.com/lmcinnes/24ed5c22c80125be5133811d677eae7b). Note that you can see how to download all datasets from this notebook.

The goal of notebook 01 is to provide a basic template walkthrough of obtaining and preparing a number of (simple) high dimensional datasets that can reasonably used to clustering evaluation. The datasets chosen have associated class labels that *should* be meaningful in terms of how the data clusters, and thus we can use label based clustering evaluation such as ARI and AMI to determine how well different clustering approaches are performing.

The datasets:
* Pendigits is the smallest, with only 1797 samples, and is only 64 dimensional: 8x8 images of digits.
* MNIST provides a good basic scaling test with 70,000 samples of handwritten digits in 784 dimensions.
* COIL-20 was collected by the Center for Research on Intelligent Systems at the Department of Computer Science, Columbia University. The database contains grayscale images of 20 objects. The objects were placed on a motorized turntable against a black background and images were taken at pose internals of 5 degrees. So, 72 images of the same object from different angles for each class.
* The USPS dataset refers to numeric data obtained from the scanning of handwritten digits from envelopes by the U.S. Postal Service. The original scanned digits are binary and of different sizes and orientations; the images here have been deslanted and size normalized, resulting in 16 x 16 grayscale images (Le Cun et al., 1990). Almost 10,000 samples and 256 dimensions.
* Buildings are images of  41 buildings under rotatins. A total of 4178 images.

## [Notebook 02](02-GraphClustering_on_UMAP_graph.ipynb) : High dimensional clustering via graph clustering

The goal of this notebook is to compare different graph clustering algorithms on the UMAP graphs (fuzzy intersection, fuzzy union, with and without edge weights). Fuzzy intersection limits the k-NN graph to bi-directional edges whereas the fuzzy union considers the directed graph yield by k-NN graph as undirected. The weights used are the UMAP weights.

We have run standard graph clustering algorithms: Louvain, Leiden, ECG and label propagation.
Results show that Leiden and Louvain are the two clustering techniques that perform the best for this task. But it is generally not as good as applying HDBSCAN on the low dimension vectors obtained from UMAP.

We have also tested if the difference in performance between graph clustering algorithms and HDBSCAN could come from the fact that HDBSCAN has a "noise" category, which reduces the data set size used for evaluating the technique. Removing noise, either from evaluation or from the dataset, does improve the performance of all algorithms, but it does not change the how the algorithms compare to one another.

## [Notebook 03](03-UMAP_graph_study.ipynb) : Characterizing edges

The goal of this notebook is to characterize edges of the UMAP graph with respect with their position (internal/external) in the groundtruth partition. It is also to see if the edges misclassified by Leiden/HDBSCAN partitions have particularities. The edge features studied include their endpoints' rho and sigma values, the number of supporting triangles, etc.

## [Notebook 06](06-KNN_graph_study.ipynb) : k-NN graph : Ranks of edges

The k-NN graphs we get from low dimensional vs. high dimensional data representation differ. Here we look at how much they differ. To measure the similarity between a point's neiborhood before and after dimensionality reduction (UMAP), we use the truncated-Kendall Tau and the Jaccard similarity. We use a different neighborhood size for each dataset: we use the $k$ value that was used in the initial clustering experiment (yielding good UMAP+HDBSCAN partition).

The distributions of the neighborhood similarities over the set of vertices of these two measures are really similar. However, the distributions's shape differ quite significantly from one dataset to the other. Consequence of the neighborhood sizes and of the dataset properties?

Would the ranks of the edges incident to a vertex - ranks in terms of different measures (high dim distance, low dim distance and UMAP weight) - be indicative of the status (internal/external) of the edge in the ground-truth clustering?
The high dimensional distance ranks and the UMAP weights ranks are both indicative of the edge status proportions in the ground truth partition. However, the low dimensional distance ranks are a lot more indicative.

## [Notebook 04](04-cliques_preservation.ipynb) : Higher dimensional simplices preservation

Here, we explore how the higher dimensional simplices are preserved within parts of a partition. We compare the number of simplices preserved in the ground truth partition with the number preserved via Leiden.

Surprisingly, although Leiden generally yields a larger partition size - so a finer partition - than the ground truth partition, it tends to preserve a larger number of simplices. This implies that Leiden's result is not a refinement of the ground truth: it groups points differently and preserves more of the simplices structure than the ground truth.

Moreover, the Leiden partition (obtained on the kNN in high dimension) also preserves more of the low-dimensional simplices than the ground-truth partition does.

## [Notebook 05](05-weighted_clique_preservation.ipynb) : Higher dimensional simplices : with weights?

Here we look at the minimum umap weights of the simplices. We want to see if simplices that are internal to ground truth parts have higher minimum weights than simplices that are external (split by ground truth partition).

Edges that are part of internal simplices (internal w.r.t. the ground truth partition) have higher minimal UMAP weights. However, the distributions of internal vs. external simplex weights are not easily separable: both are skewed towards small values.

## [Notebook 07](07-HDBSCAN_on_UMAP_graph.ipynb) : Running HDBSCAN on partial distance matrices?

Here, trying to be lazy and get as much as we can out of HDBSCAN without understanding how it behaves on sparse matrices really. What can we get from HDBSCAN on a sparse distance matrix having non-zero entries only between high dimensional k-NN data pairs?
* First we cheat: the distance values are the low dimensional distances (after dimension reduction with UMAP). If we use a small number of NNs (same as for the UMAP projection say), running HDBSCAN on the sparse matrix does not provide good clusters. The larger the number of NNs - the larger the number of non-zero values - the closer we get to HDBSCAN result on UMAP reduction. I am still surprised that by *cheating* a lot and providing a kNN graph (with large k value) of the low dim distances, we do not get similar results as HDBSCAN on the low dim points.
* from UMAP weighted adjacency matrix with unknown values encoded as infinite values. - I actually don't know how to turn the UMAP weight into a reasonable distance. I just used the inverse. (doesn't work!)

## [Notebook 07A](07A-HDBSCAN_on_UMAP_graph.ipynb) : Running HDBSCAN on partial distance matrices?

In this notebook, we use a decent code for clustering sparse distance matrices. We try it on a number of matrix transformation, namely:
* Neighborhood amplification (Romeo and Juliet) : each edge weight is replaced by the average of weights (including 0-weight) between the two endpoints' neighborhoods. If a node appears in both neiborhoods, we "clone" it and add an edge of weight 1 between the node and itself.
* Egonet distance : count number of 1, 3 and 4 cycles an edge is involved in.
* Triangle distance: Count number of triangles in which each edge is involved
* Average neighbor distance : replace high dimensional points by the centroid of their neighbors
Results vary from not good to very bad.


## [Notebook 08](08-UMAP-nepochs.ipynb) : What precision does UMAP need for good clustering?

We study the impact of UMAP's parameter n_epochs on the HDBSCAN clustering task. We wish to see what portion of the work performed by UMAP is needed for enabling cluster identification.

## [Notebook 09](09-plot_slides.ipynb) : Plots for slides

Notebook used to generate plots for presentations.

## [Notebook 10](10-boundary_nodes.ipynb) : Are boundary nodes identifiable?

TODO (not much has been tried here.) The intention here is to study features of nodes and see if we can disciminate boundary nodes from non-boundary nodes. A boundary node is a node that is connected to at least one node that is part of a different ground-truth community. The set of boundary nodes depends a lot on $k$, the number of neighbors, with a very large $k$, every node is a boundary node. 

## [Notebook 10A](10A-threshold_edge_weights.ipynb) : What if we filter edges based on weights?

Here, we have looked at running Leiden as a graph clustering algorithm on different filtered version of the UMAP graph. The filtering helps or makes it worst depending on the dataset. We have found nothing that systematically helps.

## [Notebook 11](11-edge_classifier.ipynb)  and  [Notebook 12](12-edge_clustering.ipynb): Can we identify external edges from features?

The goal of this notebook is to characterize edges of the UMAP graph with respect with their position (internal/external) in the groundtruth partition. To do so, we train an edge classifier based on some edge characteristics to predict if the edge is internal or external with respect to the ground truth partition. The edge features we consider are:
* high dimensional distance between the edge's endpoints
* UMAP weight
* Sum of endpoints' $\sigma$ and $\rho$ values - as described above
* Number of triangles in the graph containing this specific edge
We do this with all edges of all datasets at once. Our intent is to find discriminant features for edges so we train a decision tree.

TODO: This analysis is not complete I have first struggle with the imbalanced aspect of the problem. The problem is imbalanced in two ways : some datasets have a lot more edges than others, and the proportion of exteral edges is very small compared to the proportion of internal edges. From what we see on the 2-d plot of the edge features, the largest dataset (MNIST) explains most of the variabilty observed.

## [Notebook 13](13-connectedComp_truncated.ipynb) : Iterate Romeo and Juliet weights.

This is link to one of the method tried in notebook 07A.

## [Notebook 14](14-Vectorize_neighborhoods.ipynb) : Iterate Romeo and Juliet weights.

The intention here was to use the adjacency matrix as a vector representation, re-weighted by the information gain and run HDBSCAN on this high dimensional representation (don't see why it wouldn't suffer from high dimensional issues?) Did not pursue because HDBSCAN does not handle reading a sparse matrix as entry.

## [Notebook 15](15-Curvature.ipynb) : Ricci Flow Clustering

There is a recent paper that does dimension reduction using Ricci flows, and it seems very promision. They start by constructing the weighted UMAP graph, and then they modify the weights using a notion of curvature. They compare their results against UMAP and what they get is almost too good to be true. Leland suggested to look into it in order to see if any ideas there could be apply to high dimensional clustering. It turns out that the paper they based their work off is a graph clustering paper that uses Ricci flows on graphs. Those results were published in Nature and their code is available.

Here, I try to run their graph clustering code on the UMAP graph we get and the results I get are juste terrible. François did the same on graphs very easy to cluster, and same results for him: nothing works.

## [Notebook 16](16-UMAP-on-graphs-1-complete-graph.ipynb) : Exploring UMAP on a complete graph

Starter question: When everything is equally far apart, how does it pick $k$ neighbours when $n$ is bigger than $k$? I'm thinking about whether it breaks it down in a balanced way or not. How does it tear apart something it can't? What happens? An $n$-simplex lives in $n-1$ dimensions. Does it do what's expected when $k=n$ (assuming by taking $k$ neighbours I'm count myself as a neighbour)?

**Summary of Takeaways**:
* (With precomputed distances) UMAP uses sort + truncate to determine the $k$ nearest neighbours. This leads to a biased and hence extremely unbalanced selection of $k$-nearest neighbours and an extremely unbalanced graph for $U(X)$. This is definitely **NOT** the behaviour we want. Here are some other options:
    * With random sampling of the $k$ nearest neighbours this can be fixed and make $U(X)$ seem reasonable and balanced. This is likely the "right" way to select the $k$-nearest neighbours when there is a tie for the $k^{th}$ neighbour, assuming we want to stick to exactly $k$ neighbours (which seems like a practical assumption). 
    * Another option is to keep all of the neighbours from the tie. In the case of an $n$-simplex, then $U(X)$ is the $n$-complete graph. This likely adds too much computational complexity in the extreme cases (like this one) and keeps too much information as we're not doing anything to select only local information. This is probably reasonable and fine in cases with very few ties, but unreasonable when there are highly connected objects like an $n$-simplex in the data (where $k$ is smaller than $n$).
    * Another option is to only go up to the neighbour that we're sure about. In the case of an $n$-simplex, then $U(X)$ is a completely disconnected graph on $n$-nodes. This retains no connection information from a highly connected object, so is also likely not what we want in cases where there are a lot of ties for $k^{th}$ nearest neighbour. As with the previous case, this is likely ok and maybe even what we want when there are very few ties, but will be problematic when there exists an $n$-simplex where $n$ is bigger than $k$. 
* When dealing with equidistant points in a highly connected configuration, fuzzy union behaves closer to what we'd expect and want from it (maintaining the connections that are all equally important) whereas fuzzy intersection completely disconnects the $n$-simplex. Both are extremes of "all of this information is equal to me". One preserves it, one destroys it. In the case of the $n$-simplex, I'd want the conenction information to be preserved. 
* When $k=n-1$, $X \rightarrow U(X)$ is the identity, as expected (there's really no other choice). I'd expect, in theory, that $U(X)$ for any graph $X$ on $n$ nodes should give the same graph with different weights perhaps, as all other nodes are in the $k$-nearest neighbours. This doesn't quite bear out like this. See below.
* **Technical note:** the graph layout step $V(X) \rightarrow Y$ refuses to place and layout singletons in the `metric='precomputed'` code path. This means that the fuzzy intersection fails to embed all of the $n$-simplex and morally only embeds birdirectional edges coming from $U(X)$. With random sampling this means that there is some counting argument (that I'm not going to work out here) that means that we can work out exactly how many point and edges we'd expect to preserve here. Morally the number is based on how likely 2 points are to choose each other as neighbours. In other words, the bigger $n$ is relative to $k$, the fewer points (proportional to $n$) get embedded. In this case I'd prefer laying out the singletons to not embedding them at all. I'm not sure yet what UMAP does to lay out disconnected components (in the case with a precomputed metric and random initialization for the graph layout) and what the issue is with randomly placing the singletons, but I'm assuming there's a reason this is bad....need to find out more.

## [Notebook 17](17-UMAP-on-graphs-2.ipynb) : Exploring UMAP on a graph - Part 2 - Adjacency as Similarity

Moving on from the complete graph example, let's more generally consider what happens when we consider two nodes similar if they are adjacent aka. give distance 1 when there is an edge and otherwise distance is $\infty$ (often using 1000 instead of $\infty$ since the precomputed code path doesn't take sparse matrices).

**Questions**:
* What structures are preserved from $X$ to $V(X)$ when using adjacency as similarity? What is not preserved? Explore cliques for this.
* It seems like the $k$ relative to the degree distribution should matter...quantify this? Compare with randomly removing edges.

## [Notebook 18](18-UMAP-on-graphs-3-football-dataset.ipynb) 

See Notebook 17 for explanation.

## [Notebook 19](19-Concensus_Clustering_Random_graphs.ipynb) and [Notebook 19A](19A-Concensus_Clustering_Diffused_graphs.ipynb) Concensus clustering on Random Graphs

The graph obtained via UMAP has weights that can be interpreted as the probability of the edge’s existence. These weights can be used to generate several unweighted random graphs obtained by keeping edges with a probability proportional to their weights. We are exploring the idea of using a consensus clustering by clustering each random graph separately and see if this turns out to be a good way to obtain good clusters of the high dimensional points.

I used an ensemble of 20 members. I used different weak learners - Leiden or connected components on a graph obtained by randomly dropping edges of the original UMAP graph (the probability that an edge is kept is proportional to its UMAP weight). I combined the clusters obtained on the random graphs in a final graph where each edge is the proportion of times the two endpoints were put together in partitions of random graphs. As a final partition strategy we either use Leiden or HDBSCAN.

| Dataset | Straight Leiden | Leiden (final), Leiden (random) | HDBSCAN (final), CC (random) | HDBSCAN (final), Leiden (random) | UMAP+HDBSCAN | 
| --- | ----------- | ----------- | ----------- |----------- |----------- | 
| pendigits | 0.89/0.91 | 0.87/0.91 | 0.05/0.24 | 0.87/0.91 | 0.92/0.93 |
| coil | 0.75/0.88 | 0.76/0.89 | 0.60/0.89 | 0.81/0.95 | 0.79/0.94 |
| mnist | 0.80/0.84 | 0.76/0.83 | 0.00/0.00 | 0.57/0.70 | 0.92/0.90 |
| usps | 0.80/0.87 | 0.79/0.87 | 0.00/0.01 | 0.58/0.76 | 0.88/0.90 |
| buildings | 0.29/0.61 | 0.29/0.63 | 0.00/0.08 | 0.21/0.60 | 0.34/0.68 |

The results show that some of these combinations work well for a given example or another. But overall, straight Leiden is a better option for all graphs.

## [Notebook 20](20-HDBSCAN_on_diffusion_and_pruning.ipynb) and [Notebook 20A](20A-HDBSCAN_diffusion_prune_with_quantiles.ipynb) How to prune off external edges?

In this notebook, we try diffusing edge weights via the directed k-nn graph, and pruning all edges that have a resulting value under a certain threshold. This method does indeed increase the density of internal edges and reduces the density of external edges, but not everywhere. Some external edges survive that are part of a very large number of triangles. Therefore, running HDBSCAN or a graph clustering on the resulting graph does not yield the right clusters.

## [Notebook 21](21-hierarchical-clustering-by-weight-on-umap-graph-COIL.ipynb) Hierarchical clustering on the weighted UMAP graph

This is an attempt to use hierarchical clustering on the weighted UMAP graph directly.

## [Notebook 22](22-normal_probabilities.ipynb) Not the UMAP graph... a different graph!

The goal is to contruct a graph with edge weights given by estimating the probability of being the nearest neighbor. How can we estimate such a probability? Surely the sample is either the nearest neighbor or it isn't? We assume sampling has been random and somewhat noisy; but that the distribution of samples is locally uniform. In other words we assume that in a local region there is a distribution of distances to the nearest neighbor. This distribution is asymptotically a Gamma distribution; since we are in high dimensions this can be well approximated by a normal distribution (which is much cheaper to model, and to compute probabilities for). Thus for each sample we consider it's local neighborhood and fit a model of the distance to the nearest neighbor for samples in that neighborhood. Given such a model we can then compute the probability that the nearest neighbor of the sample is at least as far away as any given sample, and thus create an edge with w weight given by the probability that this point would have been the nearest neighbor under our model. This provides a (directed!) graph with proabilities assigned to edges.

This graph and the rest of the clustering algorithm in this notebook is very promising. For this reason, we will fork right here and start a new branch of the project. 

ABOUT EASYDATA
--------------
This git repository is build from the [Easydata](https://github.com/hackalog/easydata) framework, which aims to make your data science workflow reproducible.

EASYDATA REQUIREMENTS
------------
* Make
* conda >= 4.8 (via Anaconda or Miniconda)
* Git

GETTING STARTED
---------------
### Initial Git Configuration and Checking Out the Repo

If you haven't yet done so, we recommend following the instructions in [Setting up git and Checking Out the Repo](reference/easydata/git-configuration.md) in order to check-out the code and set-up your remote branches

### Setting up your environment

* Make note of the path to your conda binary:
```
   $ which conda
   ~/miniconda3/bin/conda
```
* ensure your `CONDA_EXE` environment variable is set to this value (or edit `Makefile.include` directly)
```
    export CONDA_EXE=~/miniconda3/bin/conda
```
* Create and switch to the virtual environment:
```
cd HighDimensionalClustering
make create_environment
conda activate HighDimensionalClustering
```

Now you're ready to run `jupyter notebook` (or jupyterlab) and explore the notebooks in the `notebooks` directory.

For more instructions on setting up and maintaining your environment (including how to point your environment at your custom forks and work in progress) see [Setting up and Maintaining your Conda Environment Reproducibly](reference/easydata/conda-environments.md).


Project Organization
------------
* `LICENSE`
* `Makefile`
    * Top-level makefile. Type `make` for a list of valid commands.
* `Makefile.include`
    * Global includes for makefile routines. Included by `Makefile`.
* `Makefile.env`
    * Command for maintaining reproducible conda environment. Included by `Makefile`.
* `README.md`
    * this file
* `catalog`
  * Data catalog. This is where config information such as data sources
    and data transformations are saved.
  * `catalog/config.ini`
     * Local Data Store. This configuration file is for local data only, and is never checked into the repo.
* `data`
    * Data directory. Often symlinked to a filesystem with lots of space.
    * `data/raw`
        * Raw (immutable) hash-verified downloads.
    * `data/interim`
        * Extracted and interim data representations.
    * `data/interim/cache`
        * Dataset cache
    * `data/processed`
        * The final, canonical data sets ready for analysis.
* `docs`
    * Sphinx-format documentation files for this project.
    * `docs/Makefile`: Makefile for generating HTML/Latex/other formats from Sphinx-format documentation.
* `notebooks`
    *  Jupyter notebooks. Naming convention is a number (for ordering),
    the creator's initials, and a short `-` delimited description,
    e.g. `1.0-jqp-initial-data-exploration`.
* `reference`
    * Data dictionaries, documentation, manuals, scripts, papers, or other explanatory materials.
    * `reference/easydata`: Easydata framework and workflow documentation.
    * `reference/templates`: Templates and code snippets for Jupyter
    * `reference/dataset`: resources related to datasets; e.g. dataset creation notebooks and scripts
* `reports`
    * Generated analysis as HTML, PDF, LaTeX, etc.
    * `reports/figures`
        * Generated graphics and figures to be used in reporting.
* `environment.yml`
    * The user-readable YAML file for reproducing the conda/pip environment.
* `environment.(platform).lock.yml`
    * resolved versions, result of processing `environment.yml`
* `setup.py`
    * Turns contents of `src` into a
    pip-installable python module  (`pip install -e .`) so it can be
    imported in python code.
* `src`
    * Source code for use in this project.
    * `src/__init__.py`
        * Makes `src` a Python module.
    * `src/data`
        * Scripts to fetch or generate data.
    * `src/analysis`
        * Scripts to turn datasets into output products.

--------

<p><small>This project was built using <a target="_blank" href="https://github.com/hackalog/easydata">Easydata</a>, a python framework aimed at making your data science workflow reproducible.</small></p>
