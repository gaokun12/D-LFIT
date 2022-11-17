#-------------------------------------------------------------------------------
# @author: Kun Gao
# @created: 2021/03/09
# @updated: 2021/03/09
# @desc: example of the use of LF1T algorithm
#-------------------------------------------------------------------------------

import sys
# sys.path.insert(0, 'src/')
# sys.path.insert(0, 'src/algorithms')
# sys.path.insert(0, 'src/objects')
import numpy as np
from ..src.utils import eprint
from ..src.objects.logicProgram import LogicProgram
from ..src.algorithms.lf1t import LF1T
import pickle
import time

# 1: Main
#------------
def generate_trans(file_name):
#if __name__ == '__main__':

    # 0) Example from text file representing a logic program
    #--------------------------------------------------------
    eprint("Example using logic program definition file:")
    eprint("----------------------------------------------")

    benchmark = LogicProgram.load_from_file("../data/benchmarks/logic_programs/"+file_name+".lp")

    eprint("Original logic program: \n", benchmark.logic_form())

    eprint("Generating transitions...")

    input = benchmark.generate_all_transitions()
    print(input)
    chunk_num = 10
    total_size = len(input)
    single_size = int(total_size/10)

    for i in range(chunk_num):
        with open("../data/"+file_name+"/in_format/train_only_positive.dt", "wb") as fp:
            pickle.dump(input, fp)
    print("Save successfully!")
    file_path = "../data/"+file_name+"/in_format/train_only_positive.dt"
    return file_path


def cal_acc(dataset_name):


    # 0) Example from text file representing a logic program
    #--------------------------------------------------------
    eprint("Example using logic program definition file:")
    eprint("----------------------------------------------")

    file_name = dataset_name
    benchmark =  LogicProgram.load_from_file("../../data/benchmarks/logic_programs/"+file_name+".lp")
    model_differential_LFIT = LogicProgram.load_from_file("../../data/" + file_name + "/test_lfit.txt")


    eprint("Generating transitions...")

    expected = benchmark.generate_all_transitions()
    my_res = model_differential_LFIT.generate_all_transitions()

    my_precision = LogicProgram.precision(expected, my_res) * 100


    eprint("D_LFIT accuracy: ", my_precision, "%")


    state = [1,0,1,1,1,1,1,1,1,1,1,1]
    next_label = benchmark.next(state)
    next_dilp = model_differential_LFIT.next(state)
    eprint("Next state of label ", state, " is ", next_label, " according to learned model")
    #eprint("Next state of MODEL ", state, " is ", next_model, " according to learned model")
    eprint("Next state of DILP ", state, " is ", next_dilp, " according to learned model")
    #eprint("Next state of NNLFIT ", state, " is ", next_NN_LFIT, " according to learned model")

    eprint("----------------------------------------------")

    return my_precision



def cal_acc_fol(dataset_name):

    # 0) Example from text file representing a logic program
    #--------------------------------------------------------
    eprint("Example using logic program definition file:")
    eprint("----------------------------------------------")

    file_name = dataset_name
    benchmark =  LogicProgram.load_from_file("../data/benchmarks/logic_programs/"+file_name+".lp")
    model_differential_LFIT = LogicProgram.load_from_file("../data/" + file_name + "/test_lfit.txt")


    eprint("Generating transitions...")

    expected = benchmark.generate_all_transitions()
    my_res = model_differential_LFIT.generate_all_transitions()

    my_precision = LogicProgram.precision(expected, my_res) * 100


    eprint("D_LFIT accuracy: ", my_precision, "%")
    eprint("----------------------------------------------")

    return my_precision

def rule_classification_accuracy(generated_file_name, standrad_file_name, states, class_number):
    '''
    Calculate classification accuracy of generated logic program. 
    Input: 
    - file_name. The path of benchmark.
    - states. List of state. Each state is a list. Last element in each state is label value of head feature.
    Output:
    The accuray of logic program. 
    '''
    #state = [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    length_states = len(states)    
    benchmark =  LogicProgram.load_from_file(generated_file_name) # change file name according the command
    correct = 0
    true_positive = [0] * class_number
    false_positive = [0] * class_number
    true_negative = [0] * class_number
    false_negative = [0] * class_number
    accuracy = [0] * class_number
    correct = [0] * class_number
    saved_weight_flag = [0] * class_number
    global_correct = 0


    for i in states:
        feature_list = benchmark.return_var_list()

        current_state = i[0]
        lengh_fetures = len(current_state)
        # print("Current State:")
        # for f_i in range(lengh_fetures):
        #     print(feature_list[f_i],current_state[f_i],end=', ')
        # print(' ')
        #print('current state is', current_state)
        next_state = benchmark.next(current_state)

        global_next_label = np.array(i[1][:class_number])
        global_next_predicate = np.array(next_state[-1 * (class_number):])
        #print("label next", global_next_label)
        #print("Predicate next", global_next_predicate)

        if (np.array_equal(global_next_predicate, global_next_label)):  # if predicate of rule = label 
            global_correct += 1
            #print("✅")
        #else:
            #print("❎")

        # print(np.argmax(global_next_label))
        # print(np.argmax(global_next_predicate))
        # print(np.count_nonzero(global_next_predicate == 1) == 1)
        # analysing the accuracy and AUC of each rule corresponding each class in the dataset
        for j in range(class_number):
            next_label = i[1][j]
            predicate_label = next_state[(-1 * class_number) + j]
            if next_label == predicate_label:
                correct[j] += 1
            if next_label == 1 and predicate_label == 1:
                true_positive[j] += 1
            if next_label == 0 and predicate_label == 1:
                false_positive[j] += 1
            if next_label == 1 and predicate_label == 0:
                false_negative[j] += 1
            if next_label == 0 and predicate_label == 0:
                true_negative[j] += 1
        #time.sleep(5)
        # current_state = i[0]
        # next_label = i[1][0]
        # next_state = benchmark.next(current_state)
        # predicate_label = next_state[-1]
        #print(next_state)
        #print(current_state)
        #print(next_label)
        #print(predicate_label)
        # if next_label == predicate_label:
        #     correct += 1
        
        # else:
            #print(current_state)
            # index = 1
            # for i in current_state:
            #    if i == 1:
            #        print(index,end=' ')
            #    index += 1
            #print(" ")
            #print(next_state)
        # if next_label == 1 and predicate_label == 1:
        #     true_positive += 1
        # if next_label == 0 and predicate_label == 1:
        #     false_positive += 1
        # if next_label == 1 and predicate_label == 0:
        #     false_negative += 1
        # if next_label == 0 and predicate_label == 0:
        #     true_negative += 1
        #print(i)
        #print(next_state)
    for i in range(class_number):
        accuracy[i] = correct[i] / length_states

    # accuracy = correct / length_states
    for i in range(class_number):
        print("----------------------------------")
        print("For number ", i, " class, correct/total:", correct[i],"/",length_states)
        print("Accuracy is", accuracy[i])
        print("True positive", true_positive[i], "False Positive", false_positive[i], "\nTrue negative", true_negative[i], "False Negative", false_negative[i])

    mean_accuracy = np.mean(np.array(accuracy))  # Calculate mean accuracy of rules
    global_accuracy = global_correct / length_states
    print("****************************")
    print("Mean accuracy is:", mean_accuracy)
    print("****************************")
    print("Global accuracy is:", global_accuracy)
    print("****************************")
    # print("correct/total: ",correct,"/",length_states)
    # print("Accuracy is", accuracy)
    # print("True positive", true_positive, "False Positive", false_positive, "\nTrue negative", true_negative, "False Negative", false_negative)

    for i  in range(class_number):
        if accuracy[i] == 1:
            saved_weight_flag[i] = 1

    return  mean_accuracy, saved_weight_flag

def next_state_file(generated_file_name, current_state, class_number):
    benchmark =  LogicProgram.load_from_file(generated_file_name) # change file name according the command
    next_state = benchmark.next(current_state)
    global_next_predicate = np.array(next_state[-1 * (class_number):])
    return global_next_predicate

