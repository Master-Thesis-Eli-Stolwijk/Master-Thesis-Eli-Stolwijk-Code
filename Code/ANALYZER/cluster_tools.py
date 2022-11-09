from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from itertools import chain
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
import sys

sys.path.insert(0, '/Fridge/users/eli/Code/UTIL')
import Util

def convert_label_cluster_to_data_cluster(representations, label_clusters):
    
    output = []
    
    for cluster in label_clusters:
       
       data_cluster = []
       
       for item in cluster:
           for item_b in representations:
               if item == item_b.label:
                   data_cluster.append(item_b.data)
                  
                    
       output.append(data_cluster)
    
    return output

def get_AC_clusters(model_mode, representations, distance_thresh):
    
    data = []
    labels = []
    
    for item in representations:
        if model_mode == 'LSTM' or model_mode == 'GRU':
            data.append(item.data)
        if model_mode == '3DCNN':
            data.append(item.data[0])
        labels.append(item.label)
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_thresh)
    
    clustering.fit(data)
    
    cluster_labels = clustering.labels_
    score = silhouette_score(data, cluster_labels)
    
    num_clusters = clustering.n_clusters_
    
    clusters = [[] for _ in range(num_clusters)]
    
    for u in range(0, len(labels)):
        index = cluster_labels[u]
        label = labels[u]
        clusters[index].append(label)
    
    return num_clusters, clusters, score

def get_kmedoids_clusters(model_mode, representations, num_clusters):
    
    data = []
    labels = []
    
    for item in representations:
        if model_mode == 'LSTM' or model_mode == 'GRU':
            data.append(item.data)
        if model_mode == '3DCNN':
            data.append(item.data[0])
        labels.append(item.label)
    
    Kmedoids = KMedoids(n_clusters=num_clusters).fit(data)
    
    Kmedoids_labels = Kmedoids.labels_
    score = silhouette_score(data, Kmedoids_labels)
    
    clusters = [[] for _ in range(num_clusters)]
    
    for u in range(0, len(Kmedoids_labels)):
        index = Kmedoids_labels[u]
        label = labels[u]
        clusters[index].append(label)
    
    medoid_indices = Kmedoids.medoid_indices_
    
    medoids = []
    for index in medoid_indices:
        medoids.append(representations[index].label)
    
    
    return clusters, medoids, score

def get_kmeans_clusters(model_mode, representations, num_clusters):
    
    data = []
    labels = []
    
    for item in representations:
        if model_mode == 'LSTM' or model_mode == 'GRU':
            data.append(item.data)
        if model_mode == '3DCNN':
            data.append(item.data[0])
        labels.append(item.label)
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    kmeans = KMeans(init="random", n_clusters=num_clusters, n_init=10, max_iter=300, random_state=42)

    kmeans.fit(scaled_data)
    
    kmeans_labels = kmeans.labels_
    
    score = silhouette_score(data, kmeans_labels)
    
    clusters = [[] for _ in range(num_clusters)]
    
    for u in range(0, len(kmeans_labels)):
        index = kmeans_labels[u]
        label = labels[u]
        clusters[index].append(label)
    
    cluster_centres = kmeans.cluster_centers_
    
    representatives = ["" for i in range(0, len(cluster_centres))]
    
    for u in range(0, len(cluster_centres)):
        
        representative = Util.closest_point(np.array([cluster_centres[u]]), np.array(data))
        
        for item in representations:
            if np.array_equal(np.array([representative]), item.data):
                representatives[u] = item.label
                
          
    return clusters, representatives, score

def get_AF_clusters(model_mode, representations):
    
    data = []
    labels = []
    
    for item in representations:
        if model_mode == 'LSTM' or model_mode == 'GRU':
            data.append(item.data)
        if model_mode == '3DCNN':
            data.append(item.data[0])
        labels.append(item.label)
    
    AF = AffinityPropagation()

    AF.fit(data)
    
    AF_labels = AF.labels_
    
    score = silhouette_score(data, AF_labels)
    
    num_clusters = len(AF.cluster_centers_)
    
    clusters = [[] for _ in range(num_clusters)]
    
    for u in range(0, len(AF_labels)):
        index = AF_labels[u]
        label = labels[u]
        clusters[index].append(label)
    
    cluster_centres = AF.cluster_centers_indices_
    
    representatives = ["" for i in range(0, len(cluster_centres))]
    
    for u in range(0, len(cluster_centres)):
        
        representative = representations[cluster_centres[u]].label
        representatives[u] = representative
                
          
    return clusters, representatives, score

def tsne_visualization(data, labels, dim, colors=[], m=[], representatives=[], rep_colors=[], rep_markers=[]):
    
    """
    Visualizes data through t-nse
    
    - dim: can be either 2 or 3
    - colors (optional, random color assignment otherwise): a list of colors for each data point
    - m (optional, but should be present if colors is present): a list of markers for each data point 
    - representatives (optional, but does not generate a legend without): list of representatives corresponding to each color. 
    - rep_colors (optional, but should be present if representatives is present): list of colors that correspond to the representatives
    -
    
    """
    
    print("Started T-SNE...")
    
    data = np.array(data)
    
    if dim == 2:
    
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(np.array(data))
        
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
            
        plt.figure(figsize=(16, 16)) 
    
    
        if len(colors) == 0:
            for i in range(len(x)):
                plt.scatter(x[i],y[i], s=5)
                plt.annotate(labels[i],
                             xy=(x[i], y[i]),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom')
        else:
            for i in range(0, len(x)):
                scatter = plt.scatter(x[i],y[i], c=np.array([colors[i]]), marker=m[i], s = 5)
            
            
            
            if len(representatives) > 0:
                
                handles = get_legend_handles(representatives, rep_colors, rep_markers)
            
                plt.legend(handles=handles)
                
                for u in range(len(representatives)):
                    for i in range(len(x)):
                        if labels[i] == representatives[u]:
                            plt.annotate(labels[i],
                             xy=(x[i], y[i]),
                             xytext=(5, 2),
                             textcoords='offset points',
                             ha='right',
                             va='bottom')
                            
    if dim == 3:
                                                                              
        tsne_model = TSNE(perplexity=40, n_components=3, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(np.array(data))
        
        x = []
        y = []
        z = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
            z.append(value[2])
            
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(projection='3d')
        
        if len(colors) == 0:
            for i in range(len(x)):
                
                ax.scatter(x[i],y[i],z[i], s=5) 
                ax.text(x[i],y[i],z[i], labels[i]) 
                
        else:
            
            for i in range(0, len(x)):
                ax.scatter(x[i],y[i],z[i], c=np.array([colors[i]]), marker=m[i], s = 5)
            
            if len(representatives) > 0:
                
                handles = get_legend_handles(representatives, rep_colors, rep_markers)
            
                ax.legend(handles=handles)
                for u in range(len(representatives)):
                    for i in range(len(x)):
                        if labels[i] == representatives[u]:
                            ax.text(x[i],y[i],z[i], labels[i])
        
        
    plt.show()
    
def get_legend_handles(representatives, rep_colors, rep_markers):
    
    """
    Returns the appropriate shapes and colors for the legend
    """
    
    handles = []
    
    for i in range(0, len(representatives)):
        
        patch = Line2D([0], [0], linestyle='none', mfc=rep_colors[i], mec=rep_colors[i], 
                           marker=rep_markers[i], label=representatives[i]) 
          
        handles.append(patch)
    
    return handles