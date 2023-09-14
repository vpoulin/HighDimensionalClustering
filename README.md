High Dimensional Clustering
==============================

Clustering high dimensional data points in challenging! The regular techniques such as k-means, DBSCAN, HDBSCAN, Agglomerative Clustering all suffer from the non-intuitive properties of metrics in high dimension. However, experiments have shown that dimension reduction techniques applied before running a clustering algorithm can really improve on the results obtained. More precisely, using UMAP to reduce dimensionality followed by HDBSCAN to identify clusters perform reasonably well in identifying the underlying clusters. In general, it performs better than using PCA for dimensionality reduction. The following figure shows results from a study of the [Pendigits data set](http://odds.cs.stonybrook.edu/pendigits-dataset/). We show clustering accuracies of five clustering algorithms on the original high dimensional points and on low dimensional points obtained using PCA or UMAP. The accuracies are measured with the adjusted Rand index (ARI) and the adjusted mutual information (AMI).

![](https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/figures/dim_red_4_clustering.png)

Previous exploration work can be found in the [archive folder](https://github.com/vpoulin/HighDimensionalClustering/blob/master/notebooks/archive/README.md)

The new algorithm we are currently testing works as follow. It first builds a graph. The goal is to contruct a graph with edge weights given by estimating the probability of being the nearest neighbor. This provides a (directed!) graph with proabiulities assigned to edges. We can run single linkage clustering on the resulting graph, and then use HDBSCAN style condensed tree approaches to simplify and get some clusters out. Next we need single linkage clustering of the graph. That is most easily done by computing a minimum spanning forest, and then processing that into a forest of merge trees. Lastly we condense then extract clusters. We could use any technique, but went with the leaf extraction because it seemed to work better. The clustering of the graph works well in that it picks out most of the clusters, but it leaves a great deal of data as noise. We can fix that by running a label propagation through the graph. We do still want to keep the ability to label points as noise, and it would be good to keep the propagation soft, so we can do a Bayesian label propagation of probability vectors over possible labels, including a noise label, with a Bayesian update of the distribution at each propagation round. We can then iterate until relative convergence.

See [Notebook 00]('notebooks/00-HighDimClusterer.ipynb') for the code.

## Predict cluster assignment for new data

One very nice aspect of the proposed algorithm is that it can naturally be adapted to deal with new data: unseen points. So given a data partition, perhaps with noise, and a set of new points, we can predict to which of the existing clusters the new data points should be assigned, including a noise attribution. We currently have two versions of this *predict* function, but none of them has been tested extensively so far. (The code is part of the 00 notebook).

## Stability Issues

When investigating the predict function, we have encountered stability issues. That is, under a random sample that contains 90% of the 70,000 MNIST data points, the adjusted rand index we get for our clusterings vary from 0.85 to 0.92. The scores are clustered into two groups: the clusterings that yielded 10 clusters (same as ground truth) and the ones that yielded 11 clusters. More has to be learned from this experiment. We have started to [run experiments](notebooks/StabilitySamplingExperiments_HighDClustering.ipynb) and analyse results [here](notebooks/StabitilySamplingResults_0.9_MNIST.ipynb) and [here](notebooks/StabilitySamplingResults_decisionBoundary.ipynb).

## Parameter exploration

The algorithm proposed runs under a large number of parameters. These parameters are not always intuitive to set and are not at all independent. Can the parameter space be transformed into a simplified space, more intuitive? We have started to [run experiments](notebooks/HyperParamsExperiments.ipynb) and analyse results [here](notebooks/HyperParamsResults_MNIST_boxplot.ipynb) and [here](notebooks/HyperParamsResults_MNIST_parallel_coord.ipynb).

### Issues with min_cluster_size parameter
The min-cluster-size parameter is used to build the condensed tree that is in turn use to seed the label propagation step of the algorithm. Because of the label propagation step, the minimum cluster size given in the parameters can be much smaller than the minimum cluster size. With the Japanese character Kuzushiji-MNIST Dataset, we observe the following problem. If the min-cluster-size is not small enough, some of the smallish clusters are just labelled as noise. If we make the min-cluster-size small enough, we fix the problem but we introduce a new one: larger clusters get split into smaller ones. 



## Theoretical insights into k-NN graphs

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
