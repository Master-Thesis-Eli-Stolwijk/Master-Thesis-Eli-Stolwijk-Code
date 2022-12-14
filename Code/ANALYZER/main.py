import cluster_tools
from Condensed_Word import Condensed_word
from Condenser import Condenser
import analysis_tools as Analyzer
import pickle
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import pyphen
import matplotlib

sys.path.insert(0, '/Fridge/users/eli/Code/UTIL')
import Util

participant = 'F1'


def get_eval_loader(model_mode, participant):
    """
    Returns the appropraite data loader given the type of model
    """

    if model_mode == 'GRU':
        path = "/Fridge/users/eli/Code/LSTM/files/pickled_files/models/LSTM_np_words_cut_[47, 47]_" + participant
    if model_mode == 'LSTM':
        path = "/Fridge/users/eli/Code/LSTM/files/pickled_files/models/LSTM_np_words_cut_[47, 47]_" + participant
    if model_mode == '3DCNN':
        path = "/Fridge/users/eli/Code/3D_CNN/files/pickled_files/np_words_cut_geq_5_leq_20_" + participant

    if model_mode == 'LSTM' or model_mode == 'GRU':
        sys.path.insert(0, '/Fridge/users/eli/Code/LSTM')
        import data_loaders
        data_loader = data_loaders.LSTM_loader(
            participant, 1, 0.9, [47, 47], 15)
        _, _, _, _, eval_loader = data_loader.get_loaders(path)
    if model_mode == '3DCNN':
        sys.path.insert(0, '/Fridge/users/eli/Code/3D_CNN')
        import data_loaders
        data_loader = data_loaders.julia_loader(participant, 1, 5, 20, 0.9)
        _, _, _, _, eval_loader = data_loader.get_loaders(path, 20, 15, 13)

    return eval_loader


def visualize_data(model_mode, alg, dim, clusters=[], representatives=[], representations=[], per_cluster=False):
    """
        Visualizes the data in a 2 or 3 dimensional space dimensional
        - dim: can only be 2 or 3, specifies the dimensionality to reduce towards.
    """

    if len(representations) == 0:

        eval_loader = get_eval_loader(model_mode, participant)

        model = Util.load_model_from_file()

        representations = get_representations(model, model_mode, eval_loader)

    data, labels = Util.split_condensed_words(model_mode, representations)
    

    if len(clusters) == 0:

        if alg == 'tsne':

            cluster_tools.tsne_visualization(data, labels, dim)

    else:

        colors = []

        markers = []

        color_map = []
        marker_map = []
        
        if per_cluster == False:
            cmap = matplotlib.cm.get_cmap('nipy_spectral')
        else:
            cmap = matplotlib.cm.get_cmap('gist_rainbow')

        counter = 0

        for u in range(0, len(clusters)):

            counter += 1

            color = cmap((1 / len(clusters)) * u)

            color_map.append(color)

            if counter == 1:
                marker_map.append("o")
            if counter == 2:
                marker_map.append("^")
            if counter == 3:
                counter = 0
                marker_map.append("s")

        for u in range(0, len(labels)):

            for v in range(0, len(clusters)):

                if labels[u] in clusters[v]:

                    colors.append(color_map[v])
                    markers.append(marker_map[v])
                    break

        if len(representatives) == 0:

            cluster_tools.tsne_visualization(
                data, labels, dim, colors=np.array(colors), m=markers)

        else:

            new_values = cluster_tools.tsne_visualization(data, labels, dim, colors=np.array(
                colors), m=markers, representatives=representatives, rep_colors=color_map, rep_markers=marker_map)
    
    if per_cluster == True:
        
        color_map = []
        
        for u in range(0, len(clusters)):
    
                color = cmap((1 / len(clusters)) * u)
    
                color_map.append(color)
        
        cluster_tools.tsne_visualization_per_cluster(data, labels, clusters, representatives, color_map, new_values = new_values)


def get_representations(model, model_mode, loader):
    """
    Returns model embeddings for all data points
    """

    con = Condenser()

    representations = con.condense(loader, model_mode, model)

    representations = Util.switch_representations(representations)

    representations.sort(key=lambda x: x.label, reverse=True)

    representations = Util.switch_representations(representations)

    return representations


def get_ema_correlations(model_mode, verbose):
    """
    Returns the correlations between the model embeddings and the EMA data
    """

    eval_loader = get_eval_loader(model_mode, participant)

    model = Util.load_model_from_file()

    representations = get_representations(model, model_mode, eval_loader)

    _, MRI_matrices = Analyzer.get_all_correlations(
        model, model_mode, representations, False, ['pl'])

    to_concat = []

    for u in range(0, len(MRI_matrices)):

        path = "/Fridge/users/julia/projects_STUDENTS/EvanKemmer/results/trans_mfa_normalize_false_pad_sides/f1_similarity_mat_nsyl" + \
            str(u+1) + ".csv"

        to_concat.append(Analyzer.get_EMA_correlation(
            MRI_matrices[u], path, verbose))

    result = pd.concat(to_concat)

    print(result)
    return result


def get_model_correlations(model_mode, verbose, comparisons):
    """
    Returns the correlations of the model embeddings with the given comparison proxies.
    comparisons argument should be a list of strings. Valid string: 'pl', 'mp', 'l'
    -'l': levenshtein distance
    -'pl': phonemic levenshtein distance
    -'mp': mouth position distance

    """

    eval_loader = get_eval_loader(model_mode, participant)

    model = Util.load_model_from_file()

    representations = get_representations(model, model_mode, eval_loader)

    Analyzer.get_all_correlations(
        model, model_mode, representations, verbose, comparisons)


def correlate_two_model_representations(model_mode):
    """
    Method that returns the correlations between two model embeddings per syllable count
    """

    eval_loader = get_eval_loader(model_mode, participant)

    model_a = Util.load_model_from_file()
    model_b = Util.load_model_from_file()

    representations_a = get_representations(model_a, model_mode, eval_loader)
    representations_b = get_representations(model_b, model_mode, eval_loader)

    _, MRI_matrices_a = Analyzer.get_all_correlations(
        model_a, model_mode, representations_a, False, ['mp'])
    _, MRI_matrices_b = Analyzer.get_all_correlations(
        model_b, model_mode, representations_b, False, ['mp'])

    results = Analyzer.get_MRI_Matrices_correlation(
        MRI_matrices_a, MRI_matrices_b, True)

    print(results)
    return results


def get_clusters(model_mode, alg, verbose, num_clusters=None):
    """
    Returns clusters over all data points
    """

    model = Util.load_model_from_file()
    eval_loader = get_eval_loader(model_mode, participant)

    representations = get_representations(model, model_mode, eval_loader)

    if alg == 'KMeans':

        if num_clusters == None:

            best_score = 0
            best_nc = 0

            for num_clusters in range(2, 30):

                _, _, score = cluster_tools.get_kmeans_clusters(
                    model_mode, representations, num_clusters)

                if score > best_score:

                    best_score = score
                    best_nc = num_clusters

            clusters, representatives, score = cluster_tools.get_kmeans_clusters(
                model_mode, representations, best_nc)

        else:
            clusters, representatives, score = cluster_tools.get_kmeans_clusters(
                model_mode, representations, num_clusters)

    if alg == 'KMedoids':

        if num_clusters == None:

            best_score = 0
            best_nc = 0

            for num_clusters in range(2, 30):

                _, _, score = cluster_tools.get_kmedoids_clusters(
                    model_mode, representations, num_clusters)

                if score > best_score:

                    best_score = score
                    best_nc = num_clusters

            clusters, representatives, score = cluster_tools.get_kmedoids_clusters(
                model_mode, representations, best_nc)

        else:

            clusters, representatives, score = cluster_tools.get_kmedoids_clusters(
                model_mode, representations, num_clusters)

    if alg == 'AC':

        best_score = 0
        best_nc = 0
        
        if num_clusters == None:
        
            for num_clusters in range(2, 30):
    
                _, _, score = cluster_tools.get_AC_clusters(
                    model_mode, representations, num_clusters)
    
                if score > best_score:
    
                    best_score = score
                    best_nc = num_clusters
    
            num_clusters, clusters, score = cluster_tools.get_AC_clusters(
                model_mode, representations, best_nc)
        else:
            num_clusters, clusters, score = cluster_tools.get_AC_clusters(
                model_mode, representations, num_clusters)
        representatives = [None for i in range(0, len(clusters))]

    if alg == 'AF':

        clusters, representatives, score = cluster_tools.get_AF_clusters(
            model_mode, representations)
        num_clusters = len(clusters)

    Util.log_clusters(model, alg, clusters, representatives, [score], False)

    if verbose == True:

        for u in range(0, len(clusters)):
            if len(clusters[u]) > 0:
                print("These are the cluster representatives:")
                print(representatives[u])
        print("Silhouette score")
        print(score)

    return num_clusters, clusters, representatives, score, representations


def get_clusters_per_syllables(model_mode, alg, verbose, num_clusters=None):
    """
    Returns the clusters generated by the given clustering algorithm clustered only per syllable count
    """

    model = Util.load_model_from_file()
    eval_loader = get_eval_loader(model_mode, participant)

    representations = get_representations(model, model_mode, eval_loader)

    syllable_groups = [[] for _ in range(0, 20)]
    syllable_group_representatives = [[] for _ in range(0, 20)]
    syllable_group_clusters = [[] for _ in range(0, 20)]
    silhouette_scores = []

    for item in representations:
        syllable_count = Util.count_syllables(Util.strip_string(item.label))
        syllable_groups[syllable_count].append(item)

    if alg == 'KMeans':

        if num_clusters == None:

            best_score = 0
            best_nc = 0

            for num_clusters in range(2, 30):

                silhouette_scores = []

                for u in range(0, len(syllable_groups)):

                    if len(syllable_groups[u]) > 0:

                        try:

                            _, _, score = cluster_tools.get_kmeans_clusters(
                                model_mode, syllable_groups[u], num_clusters)

                        except:

                            print("Clustering unsuccesfull; " +
                                  str(num_clusters) + " was penalized")
                            score = -1000

                        silhouette_scores.append(score)

                avg_score = sum(silhouette_scores) / len(silhouette_scores)

                if avg_score > best_score:

                    best_score = avg_score
                    best_nc = num_clusters

            silhouette_scores = []

            for u in range(0, len(syllable_groups)):

                if len(syllable_groups[u]) > 0:

                    clusters, representatives, score = cluster_tools.get_kmeans_clusters(
                        model_mode, syllable_groups[u], num_clusters)
                    syllable_group_representatives[u] = representatives
                    syllable_group_clusters[u] = clusters
                    silhouette_scores.append(score)

        else:

            for u in range(0, len(syllable_groups)):
                if len(syllable_groups[u]) > 0:

                    clusters, representatives, score = cluster_tools.get_kmeans_clusters(
                        model_mode, syllable_groups[u], num_clusters)
                    syllable_group_representatives[u] = representatives
                    syllable_group_clusters[u] = clusters
                    silhouette_scores.append(score)

    if alg == 'KMedoids':

        if num_clusters == None:

            best_score = 0
            best_nc = 0

            for num_clusters in range(2, 30):

                silhouette_scores = []

                for u in range(0, len(syllable_groups)):

                    if len(syllable_groups[u]) > 0:

                        try:

                            _, _, score = cluster_tools.get_kmedoids_clusters(
                                model_mode, syllable_groups[u], num_clusters)

                        except:

                            print("Clustering unsuccesfull; " +
                                  str(num_clusters) + " was penalized")
                            score = -1000

                        silhouette_scores.append(score)

                avg_score = sum(silhouette_scores) / len(silhouette_scores)

                if avg_score > best_score:

                    best_score = avg_score
                    best_nc = num_clusters

            silhouette_scores = []

            for u in range(0, len(syllable_groups)):

                if len(syllable_groups[u]) > 0:

                    clusters, medoids, score = cluster_tools.get_kmedoids_clusters(
                        model_mode, syllable_groups[u], num_clusters)
                    syllable_group_representatives[u] = medoids
                    syllable_group_clusters[u] = clusters
                    silhouette_scores.append(score)

        else:

            for u in range(0, len(syllable_groups)):
                if len(syllable_groups[u]) > 0:

                    clusters, medoids, score = cluster_tools.get_kmedoids_clusters(
                        model_mode, syllable_groups[u], num_clusters)
                    syllable_group_representatives[u] = medoids
                    syllable_group_clusters[u] = clusters
                    silhouette_scores.append(score)

    if alg == 'AC':

        best_score = 0
        best_nc = 0
        
        if num_clusters == None:
            
            for num_clusters in range(2, 30):
                silhouette_scores = []
    
                for u in range(0, len(syllable_groups)):
    
                    if len(syllable_groups[u]) > 0:
    
                        try:
    
                            _, _, score = cluster_tools.get_AC_clusters(
                                model_mode, syllable_groups[u], num_clusters)
    
                        except:
    
                            print("Clustering unsuccesfull; " +
                                  str(num_clusters) + " was penalized")
                            score = -1000
    
                        silhouette_scores.append(score)
    
                avg_score = sum(silhouette_scores) / len(silhouette_scores)
    
                if avg_score > best_score:
    
                    best_score = avg_score
                    best_nc = num_clusters
        else:
            best_nc = num_clusters
            
        silhouette_scores = []

        for u in range(0, len(syllable_groups)):

            if len(syllable_groups[u]) > 0:

                num_clusters, clusters, score = cluster_tools.get_AC_clusters(
                    model_mode, syllable_groups[u], best_nc)

                syllable_group_representatives[u] = [
                    None for i in range(0, len(clusters))]
                syllable_group_clusters[u] = clusters
                silhouette_scores.append(score)

    if alg == 'AF':

        for u in range(0, len(syllable_groups)):

            if len(syllable_groups[u]) > 0:

                clusters, representatives, score = cluster_tools.get_AF_clusters(
                    model_mode, syllable_groups[u])
                syllable_group_representatives[u] = representatives
                syllable_group_clusters[u] = clusters
                silhouette_scores.append(score)

    Util.log_clusters(model, alg, syllable_group_clusters,
                      syllable_group_representatives, silhouette_scores, True)

    if verbose == True:

        for u in range(0, len(syllable_group_representatives)):
            if len(syllable_group_representatives[u]) > 0:
                print(
                    "These are the cluster representatives for the words with " + str(u) + " syllables:")
                print(syllable_group_representatives[u])
        print("Silhouette scores per syllable group")
        for score in silhouette_scores:
            print(item)

    return syllable_group_clusters, syllable_group_representatives, score, representations


def cluster_and_visualize(model_mode, clus_alg, vis_alg, dim, per_syl=False, num_clusters=None, per_clus=True):
    """
    First clusters based on selected model and clustering algorithm, then visualizes based on the selected visualization algorithm

    -per_syl (optional): if false, clusters all datapoint. If true, only clusters words with same number of syllables
    -num_clusters (optional): specifies the number of clusters for the algorithms that allow specification of number 
                  of clusters, performs cluster number optimization based on silhouette scores if not provided
    """

    if per_syl == False:
        num_clusters, clusters, representatives, score, representations = get_clusters(
            model_mode, clus_alg, False, num_clusters)
        visualize_data(model_mode, vis_alg, dim, clusters,
                       representatives, representations, per_cluster=per_clus)
    else:

        syllable_clusters, representatives, score, representations = get_clusters_per_syllables(
            model_mode, clus_alg, False, num_clusters)

        for u in range(0, len(syllable_clusters)):

            if len(syllable_clusters[u]) > 0:

                clustered_data = []

                flat_clusters = [item for sublist in syllable_clusters[u]
                                 for item in sublist]

                for label in flat_clusters:

                    con_word = Condensed_word(label, Util.get_data_from_label(label, representations))
                    clustered_data.append(con_word)

                visualize_data(model_mode, vis_alg, dim, syllable_clusters[u], representatives[u], clustered_data, per_cluster=per_clus)


#get_model_correlations('3DCNN', True, ['pl'])
#get_clusters_per_syllables('3DCNN', 'KMeans', True)
#get_ema_correlations('GRU', False)
#visualize_data('3DCNN', 'tsne')

cluster_and_visualize('3DCNN', 'AC', 'tnse', 2, num_clusters=20, per_syl=False, per_clus=True)

#num_clusters, clusters, representatives, score, representations = get_clusters('3DCNN', 'KMeans', True, 20)
#visualize_data('3DCNN', 'tsne', 2, clusters, representatives)
