import numpy as np
import pandas as pd
from sklearn import cluster
from src import paths

data_folder = paths['data_path']
data_set_list = ['pendigits', 'coil', 'mnist', 'usps', 'buildings', 'clusterable']

def get_dataset_name(dataset_id):
    return(data_set_list[dataset_id])

def read_pendigits(data_folder = data_folder, images=False):
    from sklearn.datasets import load_digits
    digits = load_digits()
    raw_data = np.asarray(digits.data.astype(np.float32))
    labels = np.asarray(digits.target) 
    if(images):
        return(raw_data, labels, [16-x for x in digits['images']])
    else:
        return(raw_data, labels)

def read_coil(data_folder = data_folder, images=False):
    import re
    import zipfile
    import imageio.v2 as imageio
    import os
    
    if not os.path.isfile(data_folder / 'coil20.zip'):
        results = requests.get('http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip')
        with open(data_folder / 'coil20.zip', "wb") as code:
            code.write(results.content)
        
    images_zip = zipfile.ZipFile(data_folder / f'coil20.zip')
    mylist = images_zip.namelist()
    r = re.compile(".*\.png$")
    filelist = list(filter(r.match, mylist))
    images_zip.extractall(data_folder / '.')
    
    coil_feature_vectors = []
    image_files = []
    for filename in filelist:
        im = imageio.imread(data_folder / filename)
        coil_feature_vectors.append(im.flatten())
        image_files.append(im)
    coil_20_data = np.asarray(coil_feature_vectors)
    coil_20_target = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False).values.astype(np.int32)
    
    raw_coil = coil_20_data.astype(np.float32)
    if(images):
        return(raw_coil, coil_20_target, image_files)
    else:
        return(raw_coil, coil_20_target)

def read_mnist(data_folder = data_folder, images=False):
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("MNIST_784")
    raw_mnist = np.asarray(mnist.data.astype(np.float32))
    targets = np.array(mnist.target.astype('int'))
    if(images):
        return(raw_mnist[:35000], targets[:35000], 
               [256-np.resize(x, (28,28)) for x in raw_mnist[:35000]])
    else:
        return(raw_mnist[:35000], targets[:35000])

def read_usps(data_folder = data_folder, images=False):
    from sklearn.datasets import fetch_openml
    usps = fetch_openml("USPS", version=2)
    raw_usps = np.asarray(usps.data.astype(np.float32))
    targets = np.array(usps.target.astype('int'))
    if(images):
        return(raw_usps, targets, [-1*np.resize(x, (16,16)) for x in np.array(usps['data'])])
    else:
        return(raw_usps, targets)

def read_buildings(data_folder = data_folder, images=False):
    from glob import glob
    from PIL import Image
    import os
    
    if not os.path.isfile(data_folder / 'buildings.rar'):
        import requests
        results = requests.get('http://eprints.lincoln.ac.uk/id/eprint/16079/1/dataset.rar')
        with open(data_folder / 'buildings.rar', "wb") as code:
            code.write(results.content)
            
    if not os.path.isfile(data_folder / 'sheffield_buildings/Dataset/Dataset/1/S1-01.jpeg'):
        import rarfile
        rf = rarfile.RarFile(f'{data_folder}/buildings.rar')
        rf.extractall(f'{data_folder}/sheffield_buildings')
            
    buildings_data = []
    buildings_target = []
    buildings_images = []
    for i in range(1, 41):
        directory = data_folder / f"sheffield_buildings/Dataset/Dataset/{i}"
        buildings_images_tmp = []
        images_tmp = []
        for filename in glob(f"{directory}/*"):
            temp =Image.open(filename)
            keep = temp.copy()
            buildings_images_tmp.append(keep)
            images_tmp.append(np.asarray(keep.resize((96, 96))).flatten())
            temp.close()
        images_ = np.vstack(images_tmp)
        # buildings_images_tmp = [Image.open(filename) for filename in glob(f"{directory}/*")]
        # images_ = np.vstack([np.asarray(Image.open(filename).resize((96, 96))).flatten() for filename in glob(f"{directory}/*")])
        labels = np.full(len(glob(f"{directory}/*")), i, dtype=np.int32)
        buildings_data.append(images_)
        buildings_target.append(labels)
        buildings_images = buildings_images + buildings_images_tmp
    buildings_data = np.vstack(buildings_data)
    buildings_target = np.hstack(buildings_target)
    if(images):
        return(buildings_data, buildings_target, buildings_images)
    else:
        return(buildings_data, buildings_target)

def read_clusterable_data(data_folder = data_folder, images=False):
    import hdbscan
    import os
    
    if not os.path.isfile(data_folder / 'clusterable_data.npy'):
        import urllib.request
        git_repo_url = 'https://github.com/scikit-learn-contrib/hdbscan/blob/master/notebooks/clusterable_data.npy?raw=true'
        urllib.request.urlretrieve(git_repo_url, filename=f"{data_folder}/clusterable_data.npy")

    data= np.load(data_folder / 'clusterable_data.npy')
    target = hdbscan.HDBSCAN(min_cluster_size=15).fit_predict(data)
    if(images):
        return(data, target, None)
    else:
        return(data, target)

def map_id_name(dataset_id=-1, dataset_name=None):
    n = len(data_set_list)
    if(dataset_name is None and dataset_id == -1):
        raise ValueError('Need to define dataset_name or dataset_id')
    if(dataset_name is None):
        if(dataset_id not in list(range(n))):
            raise ValueError(f'dataset_id must an integer be between 0 and {n-1}')
        else:
            dataset_name = data_set_list[dataset_id]
    if(dataset_id == -1 and dataset_name not in data_set_list):
        raise ValueError(f'dataset_name must be in {data_set_list}')
    else:
        dataset_id = data_set_list.index(dataset_name)
    return(dataset_id, dataset_name)

def read(dataset_id, data_folder = data_folder, return_images=False):
    if(dataset_id == 0):
        res = read_pendigits(data_folder=data_folder, images=return_images)

    if(dataset_id==1):
        res = read_coil(data_folder=data_folder, images=return_images)

    if(dataset_id==2):
        res = read_mnist(data_folder=data_folder, images=return_images)

    if(dataset_id==3):
        res = read_usps(data_folder=data_folder, images=return_images)

    if(dataset_id==4):
        res = read_buildings(data_folder=data_folder, images=return_images)
        
    if(dataset_id==5):
        res = read_clusterable_data(data_folder, images=return_images)
        
    raw_data = res[0]
    labels = res[1]
    if(return_images):
        images = res[2]
        return(raw_data, labels, images)
    else:
        return(raw_data, labels)
    

def get_dataset(dataset_id=-1, dataset_name=None, top_n=None, data_folder = data_folder, return_images=False):
    dataset_id, dataset_name = map_id_name(dataset_id, dataset_name)      
    res = read(dataset_id, data_folder=data_folder, return_images=return_images)
    raw_data = res[0]
    labels = res[1]
    if(raw_data.shape[0] != len(labels)): 
        raise ValueError(f'data and labels of different lengths {raw_data.shape[0]} and {len(labels)}')
    if(top_n is not None and top_n < len(labels)):
        raw_data = raw_data[:top_n]
        labels = labels[:top_n]
    if(return_images):
        images = res[2]
        return(raw_data, labels, dataset_name, images)
    else:
        return(raw_data, labels, dataset_name)

    
# The goal of this function is to store the default parameters for every dataset (as we are calling this frequently)
# Could set set_op_mix_ratio=0.0 to get a pure fuzzy intersection - similar to the exploration of bi-directional edges in graphs
# graph_type is 'nx' or 'ig' to designate Networkx or iGraph respectively.
dataset_params ={
    0:{'min_samples':5, 'min_cluster_size':100, 'eps':0.5, 'n_neighbors':15, 
       'min_dist':1e-8, 'random_state':0, 'n_epochs':100, 'n_components':4},
    1:{'min_samples':3, 'min_cluster_size':20, 'eps':0.3, 'n_neighbors':5, 
       'min_dist':1e-8, 'random_state':0, 'n_epochs':100, 'n_components':4},
    2:{'min_samples':10, 'min_cluster_size':100, 'eps':0.1, 'n_neighbors':10, 
       'min_dist':1e-8, 'random_state':42, 'n_epochs':100, 'n_components':4},
    3:{'min_samples':10, 'min_cluster_size':100, 'eps':0.15, 'n_neighbors':10, 
       'min_dist':1e-8, 'random_state':42, 'n_epochs':100, 'n_components':4},
    4:{'min_samples':3, 'min_cluster_size':20, 'eps':0.25, 'n_neighbors':8, 
       'min_dist':1e-8, 'random_state':0, 'n_epochs':100, 'n_components':4},
    5:{'min_samples':3, 'min_cluster_size':15, 'eps':0.0, 'n_neighbors':15, 
       'min_dist':0.33, 'random_state':0, 'n_epochs':100, 'n_components':2},
}

dataset_clustering_params = {'None':
                             {
    0:{'hdbscan_min_samples':5, 'hdbscan_min_cluster_size':100, 'sl_k':160, 'dbscan_eps':20.0},
    1:{'hdbscan_min_samples':3, 'hdbscan_min_cluster_size':20, 'sl_k':80, 'dbscan_eps':5000.0},
    2:{'hdbscan_min_samples':10, 'hdbscan_min_cluster_size':100, 'sl_k':80, 'dbscan_eps':1000.0},
    3:{'hdbscan_min_samples':10, 'hdbscan_min_cluster_size':100, 'sl_k':80, 'dbscan_eps':3.5},
    4:{'hdbscan_min_samples':3, 'hdbscan_min_cluster_size':20, 'sl_k':120, 'dbscan_eps':6000.0},
},
                             'PCA':
                             {
    0:{'hdbscan_min_samples':5, 'hdbscan_min_cluster_size':100, 'sl_k':160, 'dbscan_eps':15.0, 'pca_n_components':16},
    1:{'hdbscan_min_samples':3, 'hdbscan_min_cluster_size':20, 'sl_k':80, 'dbscan_eps':4000.0, 'pca_n_components':64},
    2:{'hdbscan_min_samples':10, 'hdbscan_min_cluster_size':100, 'sl_k':80, 'dbscan_eps':600.0, 'pca_n_components':32},
    3:{'hdbscan_min_samples':10, 'hdbscan_min_cluster_size':100, 'sl_k':80, 'dbscan_eps':2.0, 'pca_n_components':32},
    4:{'hdbscan_min_samples':3, 'hdbscan_min_cluster_size':20, 'sl_k':120, 'dbscan_eps':2000.0, 'pca_n_components':32},
},
                             'UMAP':
                             {
    0:{'hdbscan_min_samples':5, 'hdbscan_min_cluster_size':100, 'sl_k':20, 'dbscan_eps':0.5},
    1:{'hdbscan_min_samples':3, 'hdbscan_min_cluster_size':20, 'sl_k':80, 'dbscan_eps':0.3},
    2:{'hdbscan_min_samples':10, 'hdbscan_min_cluster_size':100, 'sl_k':80, 'dbscan_eps':0.1},
    3:{'hdbscan_min_samples':10, 'hdbscan_min_cluster_size':100, 'sl_k':80, 'dbscan_eps':0.15},
    4:{'hdbscan_min_samples':3, 'hdbscan_min_cluster_size':20, 'sl_k':120, 'dbscan_eps':0.25},
}}

def get_dataset_params(dataset_id, param=None):
    if(param is not None):
        res = dataset_params[dataset_id][param]
    else:
        res = dataset_params[dataset_id]
    return(res)

def get_umap_graph(raw_data=None, dataset_id=-1, dataset_name=None, return_all=False, return_only_matrix=False, set_op_mix_ratio=1.0, graph_type='ig', data_folder = data_folder, params=None):
    import umap
    dataset_id, dataset_name = map_id_name(dataset_id, dataset_name)
    if(raw_data is None):
        raw_data, labels, dataset_name = get_dataset(dataset_id=dataset_id, 
                                        dataset_name=dataset_name, 
                                        data_folder = data_folder)
        
    if(params is None):
        params = get_dataset_params(dataset_id)

    A, sigmas, rhos, dists = umap.umap_.fuzzy_simplicial_set(X=raw_data, 
                                                     n_neighbors=params['n_neighbors'], 
                                                     random_state=params['random_state'], 
                                                     metric='euclidean', 
                                                     return_dists=True,
                                                    set_op_mix_ratio=set_op_mix_ratio)
    if(return_only_matrix):
        return(A, sigmas, rhos, dists)
    if(graph_type=='nx'):
        G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
    else:
        G = ig.Graph.Weighted_Adjacency(A, 'undirected')
        G.vs['sigmas'] = sigmas
        G.vs['rhos'] = rhos
    if(return_all):
        return(G, A, dists)
    else:
        return(G)
    
# The goal of this function is to store the default parameters for every dataset (as we are calling this frequently)    
def get_umap_vectors(dataset_id=-1, dataset_name=None, raw_data=None, data_folder = data_folder, n_components=None, return_vectors=True, params=None):
    import umap
    dataset_id, dataset_name = map_id_name(dataset_id, dataset_name)
    if(raw_data is None):
        raw_data, labels, dataset_name = get_dataset(dataset_id=dataset_id, 
                                        dataset_name=dataset_name, 
                                        data_folder = data_folder)

    if (params is None):
        params = get_dataset_params(dataset_id)
    if(n_components is None):
        n_components = params['n_components']

    umap_rep = umap.UMAP(n_neighbors=params['n_neighbors'],
                         n_components=n_components, 
                         min_dist=params['min_dist'], 
                         random_state=params['random_state']).fit(raw_data)
    if(return_vectors):
        umap_rep = umap_rep.embedding_
    
    return(umap_rep)

def h_dbscan(data, which_algo, dataset_id=-1):
    import hdbscan
    if which_algo not in ['hdbscan', 'dbscan']:
        raise ValueError(f'{which_algo} needs to be hdbscan or dbscan')
        
    params = get_dataset_params(dataset_id)
    if(which_algo == 'dbscan'):
        labels = cluster.DBSCAN(eps=params['eps']).fit_predict(data)

    if(which_algo == 'hdbscan'):
        labels = hdbscan.HDBSCAN(min_samples=params['min_samples'], 
                                     min_cluster_size=params['min_cluster_size']).fit_predict(data)
            
    return(labels)

    
    



