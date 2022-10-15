import numpy as np
from Condenser import Condenser
import pandas as pd
from scipy.spatial import distance
from Levenshtein import distance as lev_distance
import nltk
import scipy.stats as stats
import pyphen
from jellyfish import soundex
import py_stringmatching as sm
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

sys.path.insert(0, '/Fridge/users/eli/Code/UTIL')
import Util


def get_phonemic_levenshtein_matrix(group, show=False, title=None):
    
    group_y = []
    group_labels = []
    
    for item in group:
        if isinstance(item, (str)):
            group_labels.append(Util.strip_string(item, True))
            group_y.append(Util.strip_string(item, False))
        else:
            group_labels.append(Util.strip_string(item.label, True))
            group_y.append(Util.strip_string(item.label, False))
    
    matrix = [ [0]* len(group_y) for i in range(len(group_y))]
    
    
    arpabet = nltk.corpus.cmudict.dict()
    
    for u in range(0, len(group_y)):
        for v in range(0, len(group_y)):

            
            word_a = arpabet[Util.strip_string(group_y[u]).lower()][0]
            
            word_b = arpabet[Util.strip_string(group_y[v]).lower()][0]
            
            pld = Util.get_pld(word_a, word_b)
            
            matrix[u][v] = pld
    
    diff = np.array(matrix)
    df = pd.DataFrame(diff)
    df.columns, df.index = group_labels, group_labels
    correlations = df.corr()
    correlations_array = np.asarray(df.corr())
    
    
    if show == True:   
        
        show_heatmap(correlations_array, group_labels, "Phonemic Levenshtein distance; " + title)
        
    return correlations_array

def get_MPD_matrix(group, show=False, title=None):
    
    group_y = []
    group_labels = []
    
    for item in group:
        if isinstance(item, (str)):
            group_labels.append(Util.strip_string(item, True))
            group_y.append(Util.strip_string(item, False))
        else:
            group_labels.append(Util.strip_string(item.label, True))
            group_y.append(Util.strip_string(item.label, False))
    
    matrix = [ [0]* len(group_y) for i in range(len(group_y))]
    
    
    arpabet = nltk.corpus.cmudict.dict()
    
    for u in range(0, len(group_y)):
        for v in range(0, len(group_y)):
            
            word_a = arpabet[Util.strip_string(group_y[u]).lower()][0]
        
            word_b = arpabet[Util.strip_string(group_y[v]).lower()][0]
            
            pld = Util.get_MPD(word_a, word_b)
                
            matrix[u][v] = pld
    
    diff = np.array(matrix)
    df = pd.DataFrame(diff)
    df.columns, df.index = group_labels, group_labels
    correlations = df.corr()
    correlations_array = np.asarray(df.corr())
    
    
    if show == True:   
        
        show_heatmap(correlations_array, group_labels, "Mouth Position distance; " + title)
        
    return correlations_array

def get_levenshtein_matrix(group, show=False, title=None):
    
    group_y = []
    group_labels = []
    
    for item in group:
        if isinstance(item, (str)):
            group_labels.append(Util.strip_string(item, True))
            group_y.append(Util.strip_string(item, False))
        else:
            group_labels.append(Util.strip_string(item.label, True))
            group_y.append(Util.strip_string(item.label, False))
    
    matrix = [ [0]* len(group_y) for i in range(len(group_y))]
    
    for u in range(0, len(group_y)):
        for v in range(0, len(group_y)):
            matrix[u][v] = lev_distance(Util.strip_string(group_y[u], False), Util.strip_string(group_y[v], False))
    
    diff = np.array(matrix)
    df = pd.DataFrame(diff)
    df.columns, df.index = group_labels, group_labels
    correlations = df.corr()
    correlations_array = np.asarray(df.corr())
    
    
    if show == True:   
        
        show_heatmap(correlations_array, group_labels, "Vanilla Levenshtein distance; " + title)
        
    return correlations_array

def get_soundex_matrix(group, show=False, title=None):
    
    group_y = []
    group_labels = []
    
    for item in group:
        if isinstance(item, (str)):
            group_labels.append(Util.strip_string(item, True))
            group_y.append(soundex(Util.strip_string(item, False)))
        else:
            group_labels.append(Util.strip_string(item.label, True))
            group_y.append(soundex(Util.strip_string(item.label, False)))
    
    matrix = [ [0]* len(group_y) for i in range(len(group_y))]
    
    for u in range(0, len(group_y)):
        for v in range(0, len(group_y)):
            matrix[u][v] = lev_distance(group_y[u], group_y[v])
    
    diff = np.array(matrix)
    df = pd.DataFrame(diff)
    df.columns, df.index = group_labels, group_labels
    correlations = df.corr()
    correlations_array = np.asarray(df.corr())
    
    
    if show == True:   
        
        show_heatmap(correlations_array, group_labels, "Soundex distance; " + title)
    
    return correlations_array

def get_editex_matrix(group, show=False, title=None):
    
    group_y = []
    group_labels = []
    
    for item in group:
        if isinstance(item, (str)):
            group_labels.append(Util.strip_string(item, True))
            group_y.append(Util.strip_string(item, False))
        else:
            group_labels.append(Util.strip_string(item.label, True))
            group_y.append(Util.strip_string(item.label, False))
    
    matrix = [ [0]* len(group_y) for i in range(len(group_y))]
    
    ed = sm.similarity_measure.mra.MRA()
    
    for u in range(0, len(group_y)):
        for v in range(0, len(group_y)):
            matrix[u][v] = ed.get_raw_score(group_y[u], group_y[v])
            
    diff = np.array(matrix)
    df = pd.DataFrame(diff)
    df.columns, df.index = group_labels, group_labels
    correlations = df.corr()
    correlations_array = np.asarray(df.corr())
    
    
    if show == True:   
        
        show_heatmap(correlations_array, group_labels, "Editex distance; " + title)
    
    return correlations_array

def get_distance_matrix(group, show=False, title=None):
    
    group_x = []
    group_y = []
    group_labels = []
    
    for item in group:
        if isinstance(item, (str)):
            group_labels.append(Util.strip_string(item, True))
            group_y.append(Util.strip_string(item, False))
        else:
            group_labels.append(Util.strip_string(item.label, True))
            group_y.append(Util.strip_string(item.label, False))
    
    for item in group:
        group_x.append(item.data)
        
    
    matrix = [ [0]* len(group_x) for i in range(len(group_x))]
    
    for u in range(0, len(group_x)):
        for v in range(0, len(group_x)):
            matrix[u][v] = distance.euclidean(group_x[u], group_x[v])
            
    diff = np.array(matrix)
    df = pd.DataFrame(diff)
    df.columns, df.index = group_labels, group_labels
    correlations = df.corr()
    correlations_array = np.asarray(df.corr())
    
    
    if show == True:   
        
        show_heatmap(correlations_array, group_labels, "Model generated distance matrix; " + title)
    
    return correlations

def get_phonemic_levenshtein_matrix_from_tensor(group_y):
    
    matrix = [ [0]* len(group_y) for i in range(len(group_y))]
    
    for u in range(0, len(group_y)):
        for v in range(0, len(group_y)):

            
            word_a = arpabet[Util.strip_string(group_y[u]).lower()][0]
            
            word_b = arpabet[Util.strip_string(group_y[v]).lower()][0]
            
            pld = Util.get_pld(word_a, word_b)
            
            matrix[u][v] = pld
            
    return matrix

def get_mpd_matrix_from_tensor(group_y):
    
    matrix = [ [0]* len(group_y) for i in range(len(group_y))]
    
    for u in range(0, len(group_y)):
        for v in range(0, len(group_y)):

            
            word_a = arpabet[Util.strip_string(group_y[u]).lower()][0]
            
            word_b = arpabet[Util.strip_string(group_y[v]).lower()][0]
            
            pld = Util.get_MPD(word_a, word_b)
            
            matrix[u][v] = pld
            
    return matrix

def show_heatmap(data, labels, title):
    
    ax = sns.heatmap(data, xticklabels=labels, yticklabels=labels, cmap = 'vlag')
    ax.set_title(title)
    plt.figure()

def switch_representations(to_switch):
    
    switched = []
    
    for item in to_switch:
        item.label = item.label[::-1]
        switched.append(item)

    return switched


def count_syllables(word):
    return len(
        re.findall('(?!e$)[aeiouy]+', word, re.I) +
        re.findall('^[^aeiouy]*e$', word, re.I)
    )

def split_per_syllable(representations):
    
    syllable_split_words = [[] for x in range(10)]
    dic = pyphen.Pyphen(lang='en')
    
    for item in representations:
        
        if len(item.label) > 13:
            syl_debug = item.label
        
        syllabled= dic.inserted(Util.strip_string(item.label))
        
        syllables_count = 1
        
        for char in syllabled:
            if char == '-':
                syllables_count += 1
                
        syllable_split_words[syllables_count].append(item)
        
    return syllable_split_words

def select_upper(matrix):
    
    result = []
    
    for u in range(0, len(matrix)): #columns
        for v in range(0, len(matrix)): #rows
            if v > u:
                result.append(matrix[u][v])
    return result

def get_duplicate_columns(df):
 
    duplicateColumnNames = set()
    for x in range(df.shape[1]):
 
        col = df.iloc[:, x]
        for y in range(x + 1, df.shape[1]):
            otherCol = df.iloc[:, y]
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)

def number_duplicate_columns(df):

    renamer = defaultdict()
     
    debug = df.index.duplicated().tolist()
    debug2 = df.columns[debug].tolist()
    for column_name in debug2:
        size = len(column_name)
        if column_name[:size - 2] not in renamer:
            renamer[column_name[:size - 2]] = [column_name[:size - 2]]
            renamer[column_name[:size - 2]].append(column_name[:size - 2]+'(1)')
        else:
            renamer[column_name[:size - 2]].append(column_name[:size - 2] +'('+str(len(renamer[column_name[:size - 2]]))+')')

    column_names = df.columns.tolist()
    for column_name in column_names:
        if Util.strip_string(column_name) in renamer:
            index = df.columns.get_loc(column_name)
            new_name = renamer[Util.strip_string(column_name)].pop(0)
            df = df.rename(columns={column_name:new_name})
    
    df.index = df.index + df.groupby(level=0).cumcount().astype(str).replace('0','')
    
    return df


def get_MRI_Matrices_correlation(matrices_a, matrices_b, verbose):
    
    to_concat = []
    
    for u in range(len(matrices_a)):
        
        if u < len(matrices_b):
        
            matrix_a = matrices_a[u]
            matrix_b = matrices_b[u]
            
            header_a = matrix_a.columns.values.tolist()
            header_b = matrix_b.columns.values.tolist()
            
            new_headers = []
    
            for word_a in header_a:
                for word_b in header_b:
                    if word_a.upper() == word_b.upper():
                        new_headers.append(word_b.upper())
                        break
            
            len_header = len(new_headers)
    
            new_a_df = pd.DataFrame(np.zeros((len_header, len_header)), columns = new_headers, index = new_headers)
            new_b_df = pd.DataFrame(np.zeros((len_header, len_header)), columns = new_headers, index = new_headers)




            for word_a in new_headers:
                for word_b in new_headers:
                    val_a = matrix_a.at[word_a.upper(), word_b.upper()]
                    val_b = matrix_b.at[word_a.upper(), word_b.upper()]
                    
                    new_a_df.at[word_a.upper(), word_b.upper()] = val_a
                    new_b_df.at[word_a.upper(), word_b.upper()] = val_b
           
            
            np_a = new_a_df.to_numpy()
            np_b = new_b_df.to_numpy()
            
            
            if verbose == True:
    
                show_heatmap(np_a, new_headers, "MRI heatmap")
                show_heatmap(np_b, new_headers, "EMA heatmap")
      
            scores = []
            
            score = [u + 1,
                         stats.spearmanr(select_upper(np_a), select_upper(np_b))[0],
                         stats.spearmanr(select_upper(np_a), select_upper(np_b))[1]]
            scores.append(score)
            
            df = pd.DataFrame(scores)
            df.columns = ['num syllables', 'pearson r', 'p'] 
            df.reset_index(drop=True, inplace=True)
            
            
            to_concat.append(df)
    
    df_results_concat = pd.concat(to_concat)
    return df_results_concat

def get_EMA_correlation(MRI_matrix, path, verbose):
    
    EMA_matrix = pd.read_csv(path, index_col=[0])
    
    EMA_matrix = number_duplicate_columns(EMA_matrix)
    
    EMA_header = EMA_matrix.columns.values.tolist()
    MRI_header = MRI_matrix.columns.values.tolist()
    
    
    
    new_headers = []
    
    for word_a in MRI_header:
        for word_b in EMA_header:
            if word_a.upper() == word_b.upper():
                new_headers.append(word_b.upper())
                break
    
    len_header = len(new_headers)
    
    new_MRI_df = pd.DataFrame(np.zeros((len_header, len_header)), columns = new_headers, index = new_headers)
    new_EMA_df = pd.DataFrame(np.zeros((len_header, len_header)), columns = new_headers, index = new_headers)
    
    for word_a in new_headers:
        for word_b in new_headers:
            MRI_val = MRI_matrix.at[word_a.upper(), word_b.upper()]
            EMA_val = EMA_matrix.at[Util.format_string(word_a).lower(), word_b.lower()]
            
            if isinstance(EMA_val, float) == False:
                debug = 0
            else:
                new_MRI_df.at[word_a.upper(), word_b.upper()] = MRI_val
                new_EMA_df.at[word_a.upper(), word_b.upper()] = EMA_val
           
            
    np_MRI = new_MRI_df.to_numpy()
    np_EMA = new_EMA_df.to_numpy()
    
    if verbose == True:
    
        
        get_MPD_matrix(new_headers, True, "MPD matrix")
        show_heatmap(np_MRI, new_headers, "Model trained with PLD custom loss (weight = 1), generated distance matrix")
        show_heatmap(np_EMA, new_headers, "EMA generated distance matrix")
      
    print("Correlation:")
        
    scores = []
    
    score = [
                 stats.spearmanr(select_upper(np_MRI), select_upper(np_EMA))[0],
                 stats.spearmanr(select_upper(np_MRI), select_upper(np_EMA))[1]]
    scores.append(score)
    
    df = pd.DataFrame(scores)
    df.columns = ['pearson r', 'p'] 
    df.reset_index(drop=True, inplace=True)
    
    
    print(df)
    return df

    
def get_representation_correlations(representations, show, phonemic=False):
    
    levenshtein_matrices = []
    euclidean_matrices = []
    pld_matrices = []
    
    syllable_split_words = split_per_syllable(representations)
    
    counter = 0
    
    for group in syllable_split_words:
        counter += 1
        if len(group) > 2:
            if counter > 1:
                lev_matrix = get_levenshtein_matrix(group, False, str(counter - 1) + " syllables")
                dis_matrix = get_distance_matrix(group, show, str(counter - 1) + " syllables")
                pld_matrix = get_phonemic_levenshtein_matrix(group, show, str(counter - 1) + " syllables")
            else:
                lev_matrix = get_levenshtein_matrix(group, False)
                dis_matrix = get_distance_matrix(group, False)
                pld_matrix = get_phonemic_levenshtein_matrix(group, False)
    
            
            levenshtein_matrices.append(lev_matrix)
            euclidean_matrices.append(dis_matrix)
            pld_matrices.append(pld_matrix)
    
    if phonemic == False:
    
        print("Correlations with classic Levenshtein matrix")
        
        scores = pd.DataFrame({'number of syl': []}, {'pearson r': []}, {'p': []})
        scores = []
        
        for i in range(len(levenshtein_matrices)):
            
            score = [i + 1,
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(levenshtein_matrices[i]))[0],
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(levenshtein_matrices[i]))[1]]
            scores.append(score)
        
        df = pd.DataFrame(scores)
        df.columns = ['number of syllables', 'pearson r', 'p'] 
        df.reset_index(drop=True, inplace=True)
        
        print(df)
        return df
    else:
        print("Correlations with phonemic Levenshtein matrix")
        
        scores = []
        
        for i in range(len(pld_matrices)):
            
            score = [i + 1,
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(pld_matrices[i]))[0],
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(pld_matrices[i]))[1]]
            scores.append(score)
        
        df = pd.DataFrame(scores)
        df.columns = ['number of syllables', 'pearson r', 'p'] 
        df.reset_index(drop=True, inplace=True)
        
        print(df)
        return df
    

def get_all_correlations(model, model_mode, loader, show, modes):
    
    #choice of modes: 'pl' (phonemic levenshtein), 'l' (vanilla levenshtein), 'mp' (mout position)
    
    con = Condenser()
    
    representations = con.condense(loader, model_mode, model)
    
    representations = switch_representations(representations)
    
    representations.sort(key=lambda x: x.label, reverse=True)
    
    representations = switch_representations(representations)
    
    levenshtein_matrices = []
    euclidean_matrices = []
    pld_matrices = []
    mpd_matrices = []
    to_concat = []
    to_return = []
    
    syllable_split_words = split_per_syllable(representations)
     
    counter = 0
    
    for group in syllable_split_words:
        counter += 1
        if len(group) > 2:
            
            if counter > 4:
                if 'l' in modes: lev_matrix = get_levenshtein_matrix(group, False, str(counter - 1) + " syllables")
                dis_matrix = get_distance_matrix(group, show, str(counter - 1) + " syllables")
                if 'pl' in modes: pld_matrix = get_phonemic_levenshtein_matrix(group, show, str(counter - 1) + " syllables")
                if 'mp' in modes: mpd_matrix = get_MPD_matrix(group, show, str(counter - 1) + " syllables")
                
            else:
                
                if 'l' in modes: lev_matrix = get_levenshtein_matrix(group, False)
                dis_matrix = get_distance_matrix(group, False)
                if 'pl' in modes: pld_matrix = get_phonemic_levenshtein_matrix(group, False)
                if 'mp' in modes: mpd_matrix = get_MPD_matrix(group, False)
                
            if 'l' in modes: levenshtein_matrices.append(lev_matrix)
            euclidean_matrices.append(dis_matrix)
            if 'pl' in modes: pld_matrices.append(pld_matrix)
            if 'mp' in modes: mpd_matrices.append(mpd_matrix)
            to_return.append(dis_matrix)
        
    if 'l' in modes:
    
        print("Correlations with classic Levenshtein matrix")
        
        scores = []
        
        for i in range(len(levenshtein_matrices)):
            
            score = [i + 1,
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(levenshtein_matrices[i]))[0],
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(levenshtein_matrices[i]))[1]]
            scores.append(score)
        
        l_df = pd.DataFrame(scores)
        l_df.columns = ['number of syllables', 'pearson r', 'p'] 
        l_df.reset_index(drop=True, inplace=True)
        
        print(l_df)
        to_concat.append(l_df)
    
    
    if 'pl' in modes:
        print("Correlations with phonemic Levenshtein matrix")
        
        scores = []
        
        for i in range(len(pld_matrices)):
            
            score = [i + 1,
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(pld_matrices[i]))[0],
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(pld_matrices[i]))[1]]
            scores.append(score)
        
        pl_df = pd.DataFrame(scores)
        pl_df.columns = ['number of syllables', 'pearson r', 'p'] 
        pl_df.reset_index(drop=True, inplace=True)
        
        print(pl_df)
        to_concat.append(pl_df)
    
    if 'mp' in modes:
        print("Correlations with Mouth Position distance matrix")
        
        scores = []
        
        for i in range(len(mpd_matrices)):
            
            score = [i + 1,
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(mpd_matrices[i]))[0],
                     stats.spearmanr(select_upper(euclidean_matrices[i].to_numpy()), select_upper(mpd_matrices[i]))[1]]
            scores.append(score)
        
        mp_df = pd.DataFrame(scores)
        mp_df.columns = ['number of syllables', 'pearson r', 'p'] 
        mp_df.reset_index(drop=True, inplace=True)
        
        print(mp_df)
        to_concat.append(mp_df)
    
    
    df_results_concat = pd.concat(to_concat)
    
    return df_results_concat, to_return
    
arpabet = nltk.corpus.cmudict.dict()