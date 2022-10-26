from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler



def get_kmedoids_clusters(model_mode, representations, num_clusters):
    
    data = []
    labels = []
    
    for item in representations:
        if model_mode == 'LSTM' or model_mode == 'GRU':
            data.append(item.data)
        if model_mode == '3DCNN':
            data.append(item.data[0])
        labels.append(item.label)
        
    #scaler = StandardScaler()
    #scaled_data = scaler.fit_transform(data)
    
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
    
    kmeans_labels = kmeans.labels_.tolist()
    
    clusters = [[] for _ in range(num_clusters)]
    
    for u in range(0, len(kmeans_labels)):
        index = kmeans_labels[u]
        label = labels[u]
        clusters[index].append(label)
    
    cluster_representatives = kmeans.cluster_centers_
          
    return clusters, cluster_representatives