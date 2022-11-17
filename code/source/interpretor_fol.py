#-------------------------------------------------------------------------------
# @author: Kun Gao
# @created: 2020/08/01
# @updated: 2019/09/01
#
#
# @desc: interpretor
#-------------------------------------------------------------------------------
import tensorflow.compat.v1 as tf
# from source.model_interpretor import InterILP
import sys
from source.model_interpretor_fol import InterILP
import os
import codecs
import numpy as np
import pickle
import random
import time 

def interpretor_learner(file_name = '', meta_info = [], index = 0, target_feature_name = '', incomplete = 0, fuzzy= 0, current_index_in_list = 0, repeated_time = 0 , all_folds = 10):
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_integer('batch_size_inter', 16, 'num_head_variables_logic_programber of seqs in one batch')
    tf.flags.DEFINE_float('learning_rate_inter', 0.05, 'learning_rate_inter') 
    # when dataset is krk, set accuracy rate = 0.005
    # When dataset is mugenesis or amine, set accuracy rate = 0.005
    tf.flags.DEFINE_float('threshold', 0.9, 'a threshold for get final result')
    # when dataset is krk, set accuracy rate = 0.9
    # When dataset is mugenesis or amine, set accuracy rate = 0.8
    tf.flags.DEFINE_integer('n_epoch_inter', 100, 'n_epoch_inter') #1300
    tf.flags.DEFINE_integer('max_range', 1, 'max number ')

    # Returns factorial of n
    def fact(n):
        res = 1
        for i in range(2, n + 1):
            res = res * i
        return res

    # Compute the Combination
    def nCr(n, r):
        return (fact(n) / (fact(r)  * fact(n - r)))


    # meta_info = meta_info.replace(' ','')
    # print(meta_info)
    # meta_li = meta_info.split(',')
    # print(meta_li)
    meta_list = meta_info.tolist()
    # meta_list = []
    # for i in meta_li:
    #     if i != '':
    #         meta_list.append(int(i))
    print(meta_list)

    max_number_rules = []
    for i in meta_list:
        max_number_rules.append(int(nCr(i,i//2)))
    print(max_number_rules) # maximum possible nnumber of rules in the corresponding head

    num_head_variables_logic_program = len(meta_list)


    varlist = []
    if incomplete == 1 and  fuzzy == 0:
        feature_file_string = "../data/"+file_name+"/in_format/incomplete/"+str(current_index_in_list)+ "/"+str(repeated_time)+'/' +file_name+"_"+index+".feature"
    elif incomplete == 0 and fuzzy == 1:
        feature_file_string = "../data/"+file_name+"/in_format/fuzzy/"+str(current_index_in_list)+ "/"+str(repeated_time)+'/' +file_name+"_"+index+".feature"
    else:
        feature_file_string = "../data/"+file_name+"/in_format/"+file_name+"_"+index+".feature"
    
    with open(feature_file_string, "rb") as feature_f:
        varlist = pickle.load(feature_f)
        feature_f.close()
    print("Feature list is", varlist)

    # with open("../../data/"+file_name+"/in_format/illegal"+index+"Training.data", "rb") as fp:
    #     train = pickle.load(fp)
    #     num_data = len(train)
    #     #print(train)
    #     num_x = len(train[0][0])
    #     n_num_variable = int(len(train[0][0]) / 2)
    #     print("num_head_variables_logic_programber of instance in the training dataset:")
    #     print(num_data)
    #     print("double of variables")
    #     print(num_x)
    #     print("num_head_variables_logic_programber of variables:")
    #     print(n_num_variable)

    # * open train dataset
    if incomplete == 1 and fuzzy == 0:
        file_string_train = "../data/"+file_name +"/in_format/incomplete/"+str(current_index_in_list)+"/"+str(repeated_time)+'/'+target_feature_name+index+"Training.data"
        file_string_test = "../data/"+file_name +"/in_format/incomplete/"+str(current_index_in_list)+"/"+str(repeated_time)+'/'+target_feature_name+index+"Testing.data"
        single_weight_infomation = '../data/'+file_name+'/incomplete/'+str(current_index_in_list)+"/"+str(repeated_time)+'/single_weights_dt/'+index+'_list_info.dt'
        path_updated_list_dt = '../data/'+file_name+'/incomplete/'+str(current_index_in_list)+"/"+str(repeated_time)+'/single_weights_dt/'+index+'_list_info.dt'
        path_updated_list_txt = '../data/'+file_name+'/incomplete/'+str(current_index_in_list)+"/"+str(repeated_time)+'/single_weights_dt/'+index+'_list_info.txt'
    elif fuzzy == 1 and incomplete == 0:
        file_string_train = "../data/"+file_name +"/in_format/fuzzy/"+str(current_index_in_list)+"/"+str(repeated_time)+'/'+target_feature_name+index+"Training.data"
        file_string_test = "../data/"+file_name +"/in_format/fuzzy/"+str(current_index_in_list)+"/"+str(repeated_time)+'/'+target_feature_name+index+"Testing.data"
        single_weight_infomation = '../data/'+file_name+'/fuzzy/'+str(current_index_in_list)+"/"+str(repeated_time)+'/single_weights_dt/'+index+'_list_info.dt'
        path_updated_list_dt = '../data/'+file_name+'/fuzzy/'+str(current_index_in_list)+"/"+str(repeated_time)+'/single_weights_dt/'+index+'_list_info.dt'
        path_updated_list_txt = '../data/'+file_name+'/fuzzy/'+str(current_index_in_list)+"/"+str(repeated_time)+'/single_weights_dt/'+index+'_list_info.txt'
    else:
        file_string_train = "../data/"+file_name +"/in_format/"+target_feature_name+index+"Training.data"    
        file_string_test = "../data/"+file_name +"/in_format/"+target_feature_name+index+"Testing.data"
        single_weight_infomation = '../data/'+file_name+'/single_weights_dt/'+index+'_list_info.dt'
        path_updated_list_dt = '../data/'+file_name+'/single_weights_dt/'+index+'_list_info.dt'
        path_updated_list_txt = '../data/'+file_name+'/single_weights_dt/'+index+'_list_info.txt'
        
    with open(file_string_train, "rb") as ftrain:
        train = pickle.load(ftrain)
        train_num_data = len(train)
        print("Training data",type(train))
        #print(train)
        print(random.shuffle(train))   # randomly shuffle dataset for the first time
        #print(train)
        # input = np.array(train)[:,0]
        # output = np.array(train)[:,1]
        train_num_x = int(len(train[0][0]))
        train_num_y = 0
        for item in train[0][1]:
            if item != -2:
                train_num_y += 1
        print("training data: number of instance in the training dataset:")
        print(train_num_data)
        print("training data: number of features in clause body:")
        print(train_num_x)
        print("training data: number of features in clause head:")
        print(train_num_y)


    #open testing dataset
    with open(file_string_test, "rb") as ftest:
        test = pickle.load(ftest)
        test_num_data = len(test)
        print("Testing data",type(test))
        #print(train)
        print(random.shuffle(test))   # randomly shuffle dataset for the first time
        #print(train)
        # input = np.array(train)[:,0]
        # output = np.array(train)[:,1]
        test_num_x = int(len(test[0][0]))
        test_num_y = 0
        for item in test[0][1]:
            if item != -2:
                test_num_y += 1
        print("Testing data: number of instance in the Testing dataset:")
        print(test_num_data)
        print("Testing data: number of features in clause body:")
        print(test_num_x)
        print("Testing data: number of features in clause head:")
        print(test_num_y)

    # with open("../../data/"+file_name+"/in_format/illegal"+index+"Training.data", "rb") as fp:
    #     train = pickle.load(fp)
    #     num_data = len(train)
    #     #print(train)
    #     num_x = len(train[0][0])
    #     n_num_variable = int(len(train[0][0]) / 2)
    #     print("num_head_variables_logic_programber of instance in the training dataset:")
    #     print(num_data)
    #     print("double of variables")
    #     print(num_x)
    #     print("num_head_variables_logic_programber of variables:")
    #     print(n_num_variable)
    
    

    # def batch_generator(arr, batch_size, n_epoch):
    #     print("running")
    #     arr = copy.copy(arr)
    #     n_batches = int(len(arr) / batch_size)
    #     arr = arr[:batch_size * n_batches]
    #     #print(arr)
    #     #print(len(arr))
    #     arr = np.array(arr)
    #     #print(arr)
    #     print(arr.shape)
    #     arr = arr.reshape((-1, batch_size, arr.shape[1],arr.shape[2]))
    #     print("arrx is")
    #     print(arr.shape)
    #     #print(arr)
    #     for i in range(n_epoch):
    #         print("this epoch is", i)
    #         np.random.shuffle(arr)
    #         for n in range(0, len(arr), 1):
    #             tmp = arr[n]
    #             #print("current data is:",tmp)
    #             x = tmp[:,0]
    #             y = tmp[:,1]
    #             x = list(x)
    #             x = np.array(x)
    #             y = list(y)
    #             y = np.array(y)[:,0:train_num_y]
    #             #print("x data",x)
    #             #print("y data",y)
    #             yield x, y, i, n, len(arr)


    #g = batch_generator(train, FLAGS.batch_size_inter, FLAGS.n_epoch_inter )
    
    try:
        f = open(single_weight_infomation,'rb')
        load_saved_weight = pickle.load(f)
        f.close()
        print("saved_rules_is",load_saved_weight)
        #time.sleep(10)
    except:
        print("Something went wrong when writing to the file")
        #time.sleep(10)
        load_saved_weight = [0]* train_num_y
        #load_saved_weight = [1, 1, 0, 0, 0, 1, 0, 1, 0, 0]

        
    
    while True:
        model = InterILP(
            n_num_features = train_num_x,
            n_num_variable = train_num_y,
            n_rule_logic = max_number_rules,
            learning_rate=FLAGS.learning_rate_inter,
            n_epoch = FLAGS.n_epoch_inter,
            batch_size=FLAGS.batch_size_inter,
            early_stop = False,
            num_train_case = train_num_data,
            max_range = FLAGS.max_range,
            file_name=file_name,
            threshold = FLAGS.threshold,
            variable = varlist,
            index = index,
            acc_keep_improved = 5,
            target_feature_name = target_feature_name,
            train_data = train,
            test_data = test,
            load_saved_weight = load_saved_weight,  # pass the saved information 
            current_index_in_list = current_index_in_list,
            repeated_time = repeated_time,
            incomplete_flag = incomplete,
            fuzzy_flag = fuzzy
        )

        accuracy_rule, fidelity, accuracy_NN, load_saved_weight_updated, terminal_flag = model.train()

        if terminal_flag == 0:
            break
        else:
            load_saved_weight = load_saved_weight_updated
            print("load_saved_weight_updated", load_saved_weight_updated)
            print("load_savved_weight", load_saved_weight)
            
            with open(path_updated_list_dt, 'wb') as fp:
                pickle.dump(load_saved_weight, fp)
                fp.close()
            with open(path_updated_list_txt,'w') as fp:
                print(load_saved_weight, file = fp)
                fp.close()


    # def final_check():
    # #** Compose all best rules information and do predication 
    #         ini_saved_weight_position = [0] * train_num_y
            
    #         #Settng file path
    #         if incomplete == 1 and fuzzy == 0:
    #             file_path_weight_1 = '../data/'+file_name+'/incomplete/'+str(current_index_in_list)+"/"+str(repeated_time)+'/single_weights_dt/'
    #             file_path_weight_2 = '_list_info.dt'   
    #         elif fuzzy == 1 and incomplete == 0:
    #             file_path_weight_1 = '../data/'+file_name+'/fuzzy/'+str(current_index_in_list)+"/"+str(repeated_time)+'/single_weights_dt/'
    #             file_path_weight_2 = '_list_info.dt'  
    #         else:
    #             file_path_weight_1 = '../data/'+file_name+'/single_weights_dt/'
    #             file_path_weight_2 = '_list_info.dt'

    #         # open each weight information stored in the folds
    #         for index_of_variable in range(1,all_folds+1):
    #             file_path = file_path_weight_1 + str(index_of_variable) + file_path_weight_2
    #             with open(file_path, 'rb') as f:
    #                 single_information = pickle.load(f)
    #                 f.close()
    #             current_index = 0
    #             for i in single_information:
    #                 if i == 1:
    #                     ini_saved_weight_position[current_index] = 1
                    
    #                 current_index += 1
            


    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()    
        keys_list = [keys for keys in flags_dict]    
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    del_all_flags(tf.flags.FLAGS)
    
    # print("Doing Final Predication......")
    # if index + 1 == all_folds or index == all_folds:
    #     final_check()

    return accuracy_rule, fidelity, accuracy_NN


#index = 1
#interpretor_learner("krk", '11,', str(index))