#-------------------------------------------------------------------------------
# @author: Kun Gao
# @created: 2021/03/10
# @updated: 2021/03/10
#
#
# @desc: differentiable LFIT accuracy on relational dataset
#-------------------------------------------------------------------------------
import sys
import numpy as np
import time
from source.train_fol import meta_info_learner
from source.interpretor_fol import interpretor_learner

#deifine global variable
# Change incomplete list for a specific task
incomplete_list = [10,20,30,40,50,60,70,80,90,100]
#incomplete_list = [100]
# Change mislabelled list for a specific task
fuzzy_list = [5,10,15,20,25,30,35,40,45,50]
#fuzzy_list = [35,50]
repeate_times = 2

def step_one_meta_info(file_name, target_feature_name, incomplete = 0 , fuzzy = 0, current_index_in_list = 0,repeated_time = 0):

    # search meta information
    total_feature_number = 0.0
    feature_number = []
    mse  = []
    max_mse = 0

    #When test on incomplete datasets
    if incomplete == 1 and fuzzy == 0:
        all_folds = 5 # Reviese the number of folds into 5
        with open("../data/"+file_name+"/incomplete/"+str(current_index_in_list)+'/'+str(repeated_time)+"/meta_info.txt",'w') as meta_file:
            for j in range(1,all_folds+1):
                #** Learning single meta information and MSE information 
                single_meta, single_mse = meta_info_learner(file_name, str(j), target_feature_name, incomplete, fuzzy, current_index_in_list, repeated_time)
                #** to compute mean value 
                feature_number.append(single_meta)
                total_feature_number = np.mean(np.array(feature_number), axis=0, dtype=np.int32)
                mse.append(single_mse)
                average_mse = np.mean(np.array(mse), axis = 0, dtype=np.float32)
                print(feature_number, mse)
                if single_mse >= max_mse:
                    max_mse = single_mse
                print("Incom. Mean all possible feature number of logic program is:", (total_feature_number), "\nTemperal result is", feature_number, file= meta_file, flush= True)
                print("Incom. Mean MSE is:", (average_mse), "\n MSE list is ", mse, file= meta_file, flush= True)
        meta_file.close()

    elif fuzzy == 1 and incomplete == 0:
        print("Fuzzy Step 2.")
        #time.sleep(2)
        all_folds = 5 # Reviese the number of folds into 5
        with open("../data/"+file_name+"/fuzzy/"+str(current_index_in_list)+'/'+str(repeated_time)+"/meta_info.txt",'w') as meta_file:
            for j in range(1,all_folds+1):
                single_meta, single_mse = meta_info_learner(file_name, str(j), target_feature_name, incomplete, fuzzy, current_index_in_list, repeated_time)
                feature_number.append(single_meta)
                total_feature_number = np.mean(np.array(feature_number), axis=0, dtype=np.int32)
                mse.append(single_mse)
                average_mse = np.mean(np.array(mse), axis = 0, dtype=np.float32)  
                print(feature_number, mse)
                if single_mse >= max_mse:
                    max_mse = single_mse
                print("Fuzzy. Mean all possible feature number of logic program is:", (total_feature_number), "\nTemperal result is", feature_number, file= meta_file, flush= True)
                print("Fuzzy. Mean MSE is:", (average_mse), "\n MSE list is ", mse, file= meta_file, flush= True)
        meta_file.close()   

    else:
        all_folds = 10
        with open("../data/"+ file_name + "/meta_info.txt",'w') as meta_file:
            for i in range(1,all_folds+1):
                single_meta, single_mse = meta_info_learner(file_name, str(i), target_feature_name, incomplete, fuzzy, current_index_in_list, repeated_time)  # call meta info learner
                #single_meta = int(single_meta[:-1])
                feature_number.append(single_meta)
                print("Feature number is",feature_number)
                total_feature_number = np.mean(np.array(feature_number),axis = 0, dtype=np.int32)
                mse.append(single_mse)
                average_mse = np.mean(np.array(mse), axis = 0, dtype=np.float32) 
                print("Mean all possible feature number of logic program is:", (total_feature_number), 
                "\nTemperal result is", feature_number, file= meta_file, flush= True)
                print("Mean MSE is:", (average_mse), "\n MSE list is ", mse, file= meta_file, flush= True)
            meta_file.close()

    mean_total_feature_number = total_feature_number
    return mean_total_feature_number, max_mse

def step_two_interpretor(file_name, target_feature_name, mean_total_feature_number, incomplete= 0, fuzzy=0, current_index_in_list=0, repeated_time=0):
    #set dataset index
    all_folds = 10

    accuracy_rule = []
    accuracy_NN = []
    fidelity = []
    mean_accuracy_rule = 0.0
    mean_accuracy_nn = 0.0
    mean_fidelity = 0.0
    max_acc = 0

    if incomplete == 1 and fuzzy == 0:  #ADD REPEATED TIME AND INCOMPLETE Rate
        all_folds = 5
        with open("../data/"+file_name+"/incomplete/"+str(current_index_in_list)+'/'+str(repeated_time)+"/accuracy.txt",'w') as file_open:
            for i in range(1,all_folds+1):
                
                #run meta-info learner
                #meta_info = meta_info_learner(file_name, i)
                meta_info = mean_total_feature_number
                #run interpretator learner and check the accuracy
                accu_per_time, fidelituy_pre_time, acc_nn_per_time = interpretor_learner(file_name, meta_info, str(i), target_feature_name, incomplete, fuzzy, current_index_in_list, repeated_time) # call extracting logic program function
                accuracy_rule.append(accu_per_time)
                accuracy_NN.append(acc_nn_per_time)
                fidelity.append(fidelituy_pre_time)
                mean_accuracy_rule += accu_per_time / all_folds
                mean_accuracy_nn += acc_nn_per_time / all_folds
                mean_fidelity += fidelituy_pre_time / all_folds
                if accu_per_time >= max_acc:
                    max_acc = accu_per_time
                print("Mean rule accuracy through",all_folds,"fold(s) is",mean_accuracy_rule, "\n", accuracy_rule, file = file_open, flush = True)
                print("Mean neural network accuracy through",all_folds,"fold(s) is",mean_accuracy_nn, "\n", accuracy_NN, file = file_open, flush = True)  
                print("Mean fidelity through",all_folds,"fold(s) is",mean_fidelity, "\n", fidelity, file = file_open, flush = True) 
                print("----------------------------------")
            file_open.close()
            return max_acc
    elif fuzzy == 1 and incomplete == 0:
        print("Fuzzy step 3.")
        #time.sleep(2)
        all_folds = 5
        with open("../data/"+file_name+"/fuzzy/"+str(current_index_in_list)+'/'+str(repeated_time)+"/accuracy.txt",'w') as file_open:
            for i in range(1,all_folds+1):
                
                #run meta-info learner
                #meta_info = meta_info_learner(file_name, i)
                meta_info = mean_total_feature_number
                #run interpretator learner and check the accuracy
                accu_per_time, fidelituy_pre_time, acc_nn_per_time = interpretor_learner(file_name, meta_info, str(i), target_feature_name, incomplete, fuzzy, current_index_in_list, repeated_time) # call extracting logic program function
                accuracy_rule.append(accu_per_time)
                accuracy_NN.append(acc_nn_per_time)
                fidelity.append(fidelituy_pre_time)
                mean_accuracy_rule += accu_per_time / all_folds
                mean_accuracy_nn += acc_nn_per_time / all_folds
                mean_fidelity += fidelituy_pre_time / all_folds
                if accu_per_time >= max_acc:
                    max_acc = accu_per_time
                print("Mean rule accuracy through",all_folds,"fold(s) is",mean_accuracy_rule, "\n", accuracy_rule, file = file_open, flush = True)
                print("Mean neural network accuracy through",all_folds,"fold(s) is",mean_accuracy_nn, "\n", accuracy_NN, file = file_open, flush = True)  
                print("Mean fidelity through",all_folds,"fold(s) is",mean_fidelity, "\n", fidelity, file = file_open, flush = True) 
                print("----------------------------------")
            file_open.close()
            return max_acc
    else:
        with open('../data/'+file_name+"/accuracy.txt",'w') as file_open:
            for i in range(1,all_folds+1):
                #run meta-info learner
                #meta_info = meta_info_learner(file_name, i)
                meta_info = mean_total_feature_number
                #run interpretator learner and check the accuracy
                accu_per_time, fidelituy_pre_time, acc_nn_per_time = interpretor_learner(file_name, meta_info, str(i), target_feature_name, incomplete, fuzzy, current_index_in_list, repeated_time, all_folds) # call extracting logic program function
                accuracy_rule.append(accu_per_time)
                accuracy_NN.append(acc_nn_per_time)
                fidelity.append(fidelituy_pre_time)
                mean_accuracy_rule += accu_per_time / all_folds
                mean_accuracy_nn += acc_nn_per_time / all_folds
                mean_fidelity += fidelituy_pre_time / all_folds
                print("Mean rule accuracy through",all_folds,"fold(s) is",mean_accuracy_rule, "\n", accuracy_rule, file = file_open, flush = True)
                print("Mean neural network accuracy through",all_folds,"fold(s) is",mean_accuracy_nn, "\n", accuracy_NN, file = file_open, flush = True)  
                print("Mean fidelity through",all_folds,"fold(s) is",mean_fidelity, "\n", fidelity, file = file_open, flush = True) 
                print("----------------------------------")
            file_open.close()


def extract_fol(file_name, target_feature_name, incomplete, fuzzy, repeate_times ):
    print("Begin to do execution:")
    if incomplete == 0 and fuzzy == 0 :
        mean_total_feature_number = step_one_meta_info(file_name = file_name, target_feature_name = target_feature_name)
        step_two_interpretor(file_name=file_name, target_feature_name = target_feature_name, mean_total_feature_number = mean_total_feature_number)
    elif incomplete == 1 and fuzzy ==0 : # incomplete
        repeate_times = 2
        file_all = open("../data/"+file_name+"/incomplete/all_acc.txt", 'w')
        target_list = []
        
        for i in incomplete_list:
            one_round_max = 0
            for j in range(repeate_times):
                mean_total_feature_number, single_acc = step_one_meta_info(file_name = file_name, target_feature_name = target_feature_name, incomplete = 1, fuzzy = 0, current_index_in_list = i, repeated_time = j)
                #mean_total_feature_number = np.array([]]) 
                single_acc = step_two_interpretor(file_name = file_name, target_feature_name = target_feature_name, mean_total_feature_number = mean_total_feature_number, incomplete = 1, fuzzy = 0, current_index_in_list = i, repeated_time = j)
                if single_acc >= one_round_max:
                    one_round_max = single_acc
            target_list.append(one_round_max)
            print(target_list,file = file_all,flush = True)
        file_all.close()
    elif incomplete == 0 and fuzzy == 1: # fuzzy 
        print("Fuzzy Step 1")
        #time.sleep(2)
        repeate_times = 2 #standrand 2
        file_all = open("../data/"+file_name+"/fuzzy/all_acc.txt", 'w')
        target_list = []
        
        for i in fuzzy_list:
            one_round_max = 0
            for j in range(repeate_times):
                mean_total_feature_number, single_acc = step_one_meta_info(file_name = file_name, target_feature_name = target_feature_name, incomplete = 0, fuzzy = 1, current_index_in_list = i, repeated_time = j)   
                #print("Meta info:", mean_total_feature_number)
                #mean_total_feature_number = np.array([]) 
                single_acc = step_two_interpretor(file_name = file_name, target_feature_name = target_feature_name, mean_total_feature_number = mean_total_feature_number, incomplete = 0, fuzzy = 1, current_index_in_list = i, repeated_time = j)
                if single_acc >= one_round_max:
                    one_round_max = single_acc
            target_list.append(one_round_max)
            print(target_list,file = file_all, flush = True)
    print("The end")

file_name = sys.argv[1] # name of model
target_feature_name = sys.argv[2] # name of target predicate
incomplete = sys.argv[3]
fuzzy = sys.argv[4]
print("Dataset is", file_name,"Incomplete?" ,incomplete,"Fuzzy?", fuzzy)
#time.sleep(4)
extract_fol(file_name = file_name,target_feature_name = target_feature_name, incomplete = int(incomplete),fuzzy = int(fuzzy), repeate_times = 5)