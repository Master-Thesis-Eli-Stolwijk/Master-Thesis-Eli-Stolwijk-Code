from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import sys

sys.path.insert(0, '/Fridge/users/eli/Code/UTIL')
import Util


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