import analysis_tools as Analyzer
import pickle
import sys
 
# adding Folder_2 to the system path
sys.path.insert(0, '/Fridge/users/eli/Code/UTIL')
import Util


participant = 'F1'

def get_eval_loader(model_mode, participant):
    
    if model_mode == 'LSTM':
        path = "/Fridge/users/eli/Code/LSTM/files/pickled_files/models/LSTM_np_words_cut_[47, 47]_" + participant
    if model_mode == '3DCNN':
        path = "/Fridge/users/eli/Code/3D_CNN/files/pickled_files/np_words_cut_geq_5_leq_20_" + participant
    
    if model_mode == 'LSTM':
        sys.path.insert(0, '/Fridge/users/eli/Code/LSTM')
        import data_loaders
        data_loader = data_loaders.LSTM_loader(participant, 1, 0.9, [47,47])
        _,_,_,_, eval_loader = data_loader.get_loaders(path)
    if model_mode == '3DCNN':
        sys.path.insert(0, '/Fridge/users/eli/Code/3D_CNN')
        import data_loaders
        data_loader = data_loaders.julia_loader(participant, 1, 5, 20, 0.9)
        _,_,_,_, eval_loader = data_loader.get_loaders(path, 20, 15, 13)
    
    
    
    return eval_loader

def get_data(path):
    with open(path, "rb") as fp:   # Unpickling
        words = pickle.load(fp)
    return words

def get_all_model_correlations(model_mode, verbose):
    
    eval_loader = get_eval_loader(model_mode, participant)
    
    model = Util.load_model_from_file()
    
    Analyzer.get_all_correlations(model, model_mode, eval_loader, verbose, ['pl', 'mp'])

def correlate_two_model_representations(model_mode):
    
    eval_loader = get_eval_loader(model_mode, participant)
    
    model_a = Util.load_model_from_file()
    model_b = Util.load_model_from_file()
    
    _, MRI_matrices_a = Analyzer.get_all_correlations(model_a, model_mode, eval_loader, False, ['mp'])
    _, MRI_matrices_b = Analyzer.get_all_correlations(model_b, model_mode, eval_loader, False, ['mp'])
    
    results = Analyzer.get_MRI_Matrices_correlation(MRI_matrices_a, MRI_matrices_b, True)
    print(results)
    return results

get_all_model_correlations('LSTM', True)