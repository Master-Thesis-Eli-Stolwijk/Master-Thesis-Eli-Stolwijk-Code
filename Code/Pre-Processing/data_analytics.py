import os
import shutil
import time
import math
import cv2
import numpy as np
import pandas as pd
from Util import shave_filename_extras

import pickle
from Util import shave_filename_extras
from Util import shave_avi

from matplotlib import pyplot as plt
import matplotlib.patches as patches

class data_object:
    
    label = ""
    frame_count = 0
    mean_grid = None
    std_grid = None
    var_grid = None
    total_grid = None
    character_count = None
    place_in_sentence = None
    
    
    def __init__(self, la, fc, mg, sg, vg, tg, ng):
        self.label = la
        self.frame_count = fc
        self.mean_grid = mg
        self.std_grid = sg
        self.var_grid = vg
        self.total_grid = tg
        self.np_grid = ng
        
    def set_extra_values(self, cc, ps):
        self.character_count = cc
        self.place_in_sentence = ps
    
    def print_properties(self):
        print("Label: " + self.label)
        print("Frame count: " + str(self.frame_count))
        
    def print_pixel_properties(self, x, y):
        print("Label: " + self.label)
        print("Frame count: " + str(self.frame_count))
        print("Pixel: (" + str(x), "," + str(y) + ")")
        print("Total: " + self.total_grid[x][y])
        print("Mean: " + self.mean_grid[x][y])
        print("Variation: " + self.var_grid[x][y])
        print("Standard deviation: " + self.std_grid[x][y])
        
        
def create_numpy_file():
    participant = "F2"
    
    parent_dir = "D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT"
    parent_dir = os.path.join(parent_dir, participant)
    
    video_path = os.path.join(parent_dir, participant + "_video")
    audio_path = os.path.join(parent_dir, participant + "_audio")
    combi_path = os.path.join(parent_dir, participant + "_combi")
    
    list_of_videos = []
    count = 0
    
    list_of_data_objects = []
    
    for filename in os.listdir(video_path):
        count +=1
        if count % 100 == 0: print(count)
        file_path = os.path.join(video_path,filename)
        
        vidcap = cv2.VideoCapture(file_path)
        success,image = vidcap.read()
    
        
        video = []
        
        while success:    
            success,image = vidcap.read()
            if success:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                video.append(gray)
            
        if len(video) > 0: 
            list_of_videos.append(video)
            mean = np.mean(video)
            std = np.std(video)
            
            grid = [[[0]*len(video)]*68]*68
            np_grid = np.array(grid)
            
            frame_count = 0
            for frame in video:
                for y in range(0,68):
                    for x in range(0,68):                   
                        pixel = frame[x][y]
                        np_grid[x][y][frame_count] = pixel
                       
                frame_count += 1
                
                
                
            mean_pixel_grid = np.array([[0.1]*68]*68)
            var_pixel_grid = np.array([[0.1]*68]*68)
            std_pixel_grid = np.array([[0.1]*68]*68)
            total_pixel_grid = np.array([[0]*68]*68)
            
            for y in range(0,68):
                for x in range(0,68):
                    pixel_array = np_grid[x][y]
                    
                    mean = np.mean(pixel_array)
                    std = np.std(pixel_array)
                    var = np.var(pixel_array)
                    total = np.sum(pixel_array)
                    
                    mean_pixel_grid[x][y] = mean
                    std_pixel_grid[x][y] = std
                    var_pixel_grid[x][y] = var
                    total_pixel_grid[x][y] = total
            
            list_of_data_objects.append(data_object(filename, len(video), mean_pixel_grid, std_pixel_grid, var_pixel_grid, total_pixel_grid, np_grid))
    
    import pickle
    with open("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/data_analysis_data_" + str(participant), "wb") as fp:   #Pickling
        pickle.dump(list_of_data_objects, fp)

def create_numpy_file_without_path(videos, participant):
    
    count = 0
    print("Creating numpy file...")
    list_of_videos = []
    
    list_of_data_objects = []
    
    for word in videos:
        count += 1
        if count % 25 == 0: print("Processed " + str(count) + " words out of " + str(len(videos)))
        character_count = len(shave_filename_extras(word.label))
        
        video = np.array(word.video_data)        
        
        if len(video) > 0: 
            list_of_videos.append(video)
            mean = np.mean(video)
            std = np.std(video)
            
            grid = [[[0]*len(video)]*68]*68
            np_grid = np.array(grid)
         #   print("video")
         #   print(video.shape)
            frame_count = 0
            for frame in video:
             #   print("frame")
             #   print(frame.shape)
                for y in range(0,68):
                    for x in range(0,68):                   
                        pixel = frame[x][y]
                       # print("pixel")
                       # print(pixel.shape)
                        np_grid[x][y][frame_count] = pixel
                       
                frame_count += 1
                
            mean_pixel_grid = np.array([[0.1]*68]*68)
            var_pixel_grid = np.array([[0.1]*68]*68)
            std_pixel_grid = np.array([[0.1]*68]*68)
            total_pixel_grid = np.array([[0]*68]*68)
            
            for y in range(0,68):
                for x in range(0,68):
                    pixel_array = np_grid[x][y]
                    
                    mean = np.mean(pixel_array)
                    std = np.std(pixel_array)
                    var = np.var(pixel_array)
                    total = np.sum(pixel_array)
                    
                    mean_pixel_grid[x][y] = mean
                    std_pixel_grid[x][y] = std
                    var_pixel_grid[x][y] = var
                    total_pixel_grid[x][y] = total
            
            to_add = data_object(word.label, len(video), mean_pixel_grid, std_pixel_grid, var_pixel_grid, total_pixel_grid, np_grid)
            to_add.set_extra_values(character_count, word.place_in_sentence)
            list_of_data_objects.append(to_add)
    
    import pickle
    with open("extended_data_analysis_data_" + str(participant), "wb") as fp:   #Pickling
        pickle.dump(list_of_data_objects, fp)
    return("extended_data_analysis_data_" + str(participant))   
    
def analyze_numpy_file(path, participant, Excell_write):
    
        with open(path, "rb") as fp:   # Unpickling
            data = pickle.load(fp)

        frame_count_groups = [[] for x in range(200)]

        for item in data:
            frame_count_groups[item.frame_count].append(item)

        list_of_labels_1 = []
        list_of_labels_2 = []


        list_of_variances = [0.0]
        list_of_variances_avg = [0.0]
        list_of_std =[0.0]
        list_of_std_avg = [0.0]
        list_of_sums = [0.0]
        list_of_sums_avg = [0.0]
        total_pixel_grid = np.array([[0]*68]*68)
        total_var_grid = np.array([[0]*68]*68)

        list_of_firsts = []
        list_of_lasts = []

        list_of_double_vars = []
        list_of_avg_vars = []


        qrt_1 = []
        qrt_2 = []
        qrt_3 = []
        qrt_4 = []

        for word in data:
            if word.place_in_sentence <= 0.25:
                qrt_1.append(word)
            else:
                if word.place_in_sentence <= 0.5:
                    qrt_2.append(word)
                else:
                    if word.place_in_sentence <= 0.75:
                        qrt_3.append(word)
                    else:
                        qrt_4.append(word)
            if word.place_in_sentence <= 0.2:
                list_of_firsts.append(word)
            if word.place_in_sentence == 1:
                list_of_lasts.append(word)

        frame_count_1 = 0
        frame_count_2 = 0
        frame_count_3 = 0
        frame_count_4 = 0
        frame_count_first = 0
        frame_count_last = 0

        character_count_1 = 0
        character_count_2 = 0
        character_count_3 = 0
        character_count_4 = 0
        character_count_first = 0
        character_count_last = 0

        variances_1 = 0
        variances_2 = 0
        variances_3 = 0
        variances_4 = 0
        variances_first = 0
        variances_last = 0

        for word in qrt_1:
            frame_count_1 += word.frame_count
            character_count_1 += len(shave_filename_extras(word.label))
            variances_1 += np.sum(word.var_grid)
        for word in qrt_2:
            frame_count_2 += word.frame_count
            character_count_2 += len(shave_filename_extras(word.label))
            variances_2 += np.sum(word.var_grid)
        for word in qrt_3:
            frame_count_3 += word.frame_count
            character_count_3 += len(shave_filename_extras(word.label))
            variances_3 += np.sum(word.var_grid)
        for word in qrt_4:
            frame_count_4 += word.frame_count
            character_count_4 += len(shave_filename_extras(word.label))
            variances_4 += np.sum(word.var_grid)

        for word in list_of_firsts:
            frame_count_first += word.frame_count
            character_count_first += len(shave_filename_extras(word.label))
            variances_first += np.sum(word.var_grid)

        for word in list_of_lasts:
            frame_count_last += word.frame_count
            character_count_last += len(shave_filename_extras(word.label))
            variances_last += np.sum(word.var_grid)

        avg_frame_count_1 = frame_count_1 / len(qrt_1)
        avg_frame_count_2 = frame_count_2 / len(qrt_2)
        avg_frame_count_3 = frame_count_3 / len(qrt_3)
        avg_frame_count_4 = frame_count_4 / len(qrt_4)
        avg_frame_count_first = frame_count_first / len(list_of_firsts)
        avg_frame_count_last = frame_count_last / len(list_of_lasts)

        avg_character_count_1 = character_count_1 / len(qrt_1)
        avg_character_count_2 = character_count_2 / len(qrt_2)
        avg_character_count_3 = character_count_3 / len(qrt_3)
        avg_character_count_4 = character_count_4 / len(qrt_4)
        avg_character_count_first = character_count_first / len(list_of_firsts)
        avg_character_count_last = character_count_last / len(list_of_lasts)

        avg_variances_1 = variances_1 / len(qrt_1)
        avg_variances_2 = variances_2 / len(qrt_2)
        avg_variances_3 = variances_3 / len(qrt_3)
        avg_variances_4 = variances_4 / len(qrt_4)
        avg_variances_first = variances_first / len(list_of_firsts)
        avg_variances_last = variances_last / len(list_of_lasts)


        for frame_group in frame_count_groups:
            
            if len(frame_group) > 0:     
                
                var_pixel_grid = np.array([[0]*68]*68)
                std_pixel_grid = np.array([[0]*68]*68)
                sum_pixel_grid = np.array([[0]*68]*68)
                
                for individual in frame_group:
                    
                    total_var_grid = total_var_grid + np.absolute(individual.var_grid)
                    total_pixel_grid = total_pixel_grid + individual.total_grid
                    var_pixel_grid = var_pixel_grid + individual.var_grid
                    std_pixel_grid = std_pixel_grid + individual.std_grid
                    sum_pixel_grid = sum_pixel_grid + individual.total_grid
                    
                
                list_of_variances.append(np.sum(var_pixel_grid))
                list_of_std.append(np.sum(std_pixel_grid))
                list_of_sums.append(np.sum(sum_pixel_grid))
                
                ### Average block ###
                divide_grid = np.array([[len(frame_group)]*68]*68)
                average_var_per_pixel_per_frame = np.divide(var_pixel_grid, divide_grid)
                list_of_variances_avg.append(np.sum(average_var_per_pixel_per_frame))
                average_std_per_pixel_per_frame = np.divide(std_pixel_grid, divide_grid)
                list_of_std_avg.append(np.sum(average_std_per_pixel_per_frame))
                average_sum_pixel_grid_per_frame = np.divide(sum_pixel_grid, divide_grid)
                list_of_sums_avg.append(np.sum(average_sum_pixel_grid_per_frame))


        label_list = ["Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4", "First words", "Last words"]
        frame_count_list = [avg_frame_count_1, avg_frame_count_2, avg_frame_count_3, avg_frame_count_4, avg_frame_count_first, avg_frame_count_last]
        character_count_list = [avg_character_count_1, avg_character_count_2, avg_character_count_3, avg_character_count_4, avg_character_count_first, avg_character_count_last]
        variances = [avg_variances_1, avg_variances_2, avg_variances_3, avg_variances_4, avg_variances_first, avg_variances_last]
        member_count = [len(qrt_1), len(qrt_2), len(qrt_3), len(qrt_4), len(list_of_firsts), len(list_of_lasts)]

        identical_variance_df = pd.DataFrame(
            {'Label': label_list,
             'Average frame count': frame_count_list,
             'Average character count': character_count_list,
             'Average variance': variances,
             'Nr members': member_count
            })

        var_sum = 0
        counter = 0
        for row in total_var_grid:
            for number in row:
                counter += 1
                var_sum = var_sum + number

        average = var_sum / counter

        print("Average variation per pixel was determined to be: " + str(average))

        from matplotlib import pyplot as plt
        import matplotlib.patches as patches
        plt.imshow(total_var_grid, interpolation='nearest')
        plt.show()

        dupe_total_var_grid = total_var_grid.copy()

        for u in range(0, len(dupe_total_var_grid)):
            for v in range(0, len(dupe_total_var_grid)):
                if dupe_total_var_grid[u][v] < average:
                    dupe_total_var_grid[u][v] = 0

        right_border = 0
        left_border = 0
        upper_border = 0
        lower_border = 0



        for u in range(0, len(dupe_total_var_grid)):
            for v in range(0, len(dupe_total_var_grid)):
                if dupe_total_var_grid[u][v] < average:
                    dupe_total_var_grid[u][v] = 0

        row_counter = 0
        for row in dupe_total_var_grid:
            row_counter += 1
            if row.sum() > 1000:
                upper_border = row_counter
                break
            
        row_counter = 0
        for row in np.flip(dupe_total_var_grid, 0):
            row_counter += 1
            if row.sum() > 1000:
                lower_border = len(dupe_total_var_grid) - row_counter
                break

        row_counter = 0
        for row in np.rot90(dupe_total_var_grid, 1, (0,1)):
            row_counter += 1
            if row.sum() > 1000:
                right_border = len(dupe_total_var_grid) - row_counter
                break

        row_counter = 0
        for row in np.rot90(dupe_total_var_grid, 3, (0,1)):
            row_counter += 1
            if row.sum() > 1000:
                left_border = row_counter
                break

        print("Right border of the triangle")
        print(right_border)
        print("Left border of the triangle")
        print(left_border)
        print("Upper border of the triangle")
        print(upper_border)
        print("Lower border of the triangle")
        print(lower_border)


                    
        heigth = upper_border - lower_border
        width = left_border - right_border         
        
        pixels = heigth * width
        print("Pixels within border")
        print(pixels)        
                    
        plt.imshow(total_var_grid, interpolation='nearest')
        ax = plt.gca()

        rect = patches.Rectangle((right_border, lower_border), width, heigth, linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)

        plt.show()

        if Excell_write == True:

            writer = pd.ExcelWriter("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/" + participant +"_word_placement_analysis.xlsx", engine='xlsxwriter')
            identical_variance_df.to_excel(writer, sheet_name='Word placement analysis' + participant, index=False)
            writer.save()   
            
            df = pd.DataFrame(total_var_grid)
            writer = pd.ExcelWriter("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/" + participant +"_total_variance.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Sum of variances per pixel' + participant, index=False)
            writer.save()    
            
            df = pd.DataFrame(total_pixel_grid)
            writer = pd.ExcelWriter("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/" + participant +"_total_pixel_activation.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Sum of values per pixel' + participant, index=False)
            writer.save()
                    
            df = pd.DataFrame(list_of_variances_avg)
            writer = pd.ExcelWriter("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/" + participant +"_average_variance_per_frame_count.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Average variance per framecount', index=False)
            writer.save()
                      
            df = pd.DataFrame(list_of_std_avg)
            writer = pd.ExcelWriter("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/" + participant +"_average_std_per_frame_count.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Average std per framecount', index=False)
            writer.save()
            
            df = pd.DataFrame(list_of_sums_avg)
            writer = pd.ExcelWriter("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/" + participant +"_average_sums_per_frame_count.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Average sums per framecount', index=False)
            writer.save()
            
            df = pd.DataFrame(list_of_variances)
            writer = pd.ExcelWriter("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/" + participant +"_variance_per_frame_count.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name='variance per framecount', index=False)
            writer.save()
                      
            df = pd.DataFrame(list_of_std)
            writer = pd.ExcelWriter("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/" + participant +"_std_per_frame_count.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name='std per framecount', index=False)
            writer.save()
            
            df = pd.DataFrame(list_of_sums)
            writer = pd.ExcelWriter("D:/School/Master/Scriptie/BCI/Data_sets/Processed_TIMIT/analysis/" + participant +"_sums_per_frame_count.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name='sums per framecount', index=False)
            writer.save()
