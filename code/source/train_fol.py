#-------------------------------------------------------------------------------
# @author: Kun Gao
# @created: 2020/08/01
# @updated: 2019/09/01
#
#
# @desc: meta-info learner
#-------------------------------------------------------------------------------
import tensorflow.compat.v1 as tf
from source.model import DifferILP # execute code on highest level
#from model import DifferILP #execute code in the same level
import numpy as np
import pickle

def meta_info_learner(file_name = '',index = 0, target_feature_name = '', incomplete=0, fuzzy=0, current_index_in_list = 0 , repeated_time = 0):

    FLAGS = tf.flags.FLAGS

    tf.flags.DEFINE_integer('batch_size', 16, 'number of seqs in one batch') # choose size for batch
    tf.flags.DEFINE_float('learning_rate', 1, 'learning_rate')  
    # 0.1 in origin version. Adjust learning rate according the number of trainable parameters. fission = 0.05
    # For example. For KRK, accuracy rate = 0.001. For Mutagenesis, learning rate = 0.1.  uwe   aine 
    tf.flags.DEFINE_integer('n_epoch', 300, 'n_epoch') #6000
    tf.flags.DEFINE_float('threshold', 0.9, 'n_rule_logic')
    # 0.6 in original version (without L1 loss and with minus one operation during inference process), 
    # 0.5 doing is ok becasue of L1 loss (No minus one and add L1)
    tf.flags.DEFINE_float('required_improvement', 200, 'a threshold of toleranting stoping improved loss')
    tf.flags.DEFINE_float('print_step_per_epoch', 8, 'each step to check matrix during one epoch')
    #tf.flags.DEFINE_string('file_name', 'fission', 'name of the output file')

    train_sizes = [1]
    run_step = 1

    file_string = ''
    #open train dataset
    if incomplete == 1 and fuzzy == 0:
        file_string_train = "../data/"+file_name +"/in_format/incomplete/"+str(current_index_in_list)+"/"+str(repeated_time)+'/'+target_feature_name+index+"Training.data"
        file_string_test = "../data/"+file_name +"/in_format/incomplete/"+str(current_index_in_list)+"/"+str(repeated_time)+'/'+target_feature_name+index+"Testing.data"

    elif fuzzy == 1 and incomplete == 0:
        file_string_train = "../data/"+file_name +"/in_format/fuzzy/"+str(current_index_in_list)+"/"+str(repeated_time)+'/'+target_feature_name+index+"Training.data"
        file_string_test = "../data/"+file_name +"/in_format/fuzzy/"+str(current_index_in_list)+"/"+str(repeated_time)+'/'+target_feature_name+index+"Testing.data"
    else:
        file_string_train = "../data/"+file_name +"/in_format/"+target_feature_name+index+"Training.data"    
        file_string_test = "../data/"+file_name +"/in_format/"+target_feature_name+index+"Testing.data"

    with open(file_string_train, "rb") as ftrain:
        train = pickle.load(ftrain)
        train_num_data = len(train)
        print("Training data",type(train))
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
    
    # open test dataset
    with open(file_string_test, "rb") as ftest:
        test = pickle.load(ftest)
        test_num_data = len(test)
        # input = np.array(test)[:,0]
        # output = np.array(test)[:,1]
        test_num_x = int(len(test[0][0]))
        test_num_y = 0
        for item in test[0][1]:
            if item != -2:
                test_num_y += 1
        print("testing data: number of instance in the training dataset:")
        print(test_num_data)
        print("testing data: number of features in clause body:")
        print(test_num_x)
        print("testing data: number of features in clause head:")
        print(test_num_y)

    # check equality 
    if train_num_x != test_num_x or train_num_y != test_num_y:
        print("Error")
        
    num_x = train_num_x
    num_y = train_num_y

    model = DifferILP(
        num_x = num_x,
        num_y = num_y, 
        learning_rate=FLAGS.learning_rate,
        index_sum = 1,
        index_or = 1,
        n_epoch = FLAGS.n_epoch,
        batch_size=FLAGS.batch_size,
        early_stop=True,
        threshold_res = FLAGS.threshold,
        file_name = file_name,
        require_improvement = FLAGS.required_improvement,
        index = index,
        print_step = FLAGS.print_step_per_epoch,
        incomplete_flag = incomplete,
        fuzzy_flag = fuzzy,
        current_index_in_parameter_list = current_index_in_list,
        repeated_time = repeated_time
        )


    train_index = 0
    final_graph = []
    meta_info = ''
    for train_index in range(len(train_sizes)):
        train_size = train_sizes[train_index]
        print("running on the train size:",train_size * train_num_data, "(", train_size, ")")
        test_mse = []
        point = []
        for run_time in range(run_step):

            train = np.array(train)
            #np.random.shuffle(train)
            #print("train",train)
            
            test = np.array(test)
            #np.random.shuffle(test)
            #print("test",test)

            n_test = test
            n_train = train

            print("test",n_test,np.shape(n_test))
            print("train",n_train,np.shape(n_train))

            info, single_mse = model.train(n_train,n_test)

            print("ini meta info from learner is",info)
            meta_info = info 
            # meta_info_list = info.tolist()
            # for index_meta in range(len(meta_info_list)):
            #     if index_meta == num_x - 1:
            #         meta_info+=str(meta_info_list[index_meta])
            #         break
            #     meta_info+=str(meta_info_list[index_meta])+','
            print(meta_info, single_mse)


    def del_all_flags(FLAGS):
        flags_dict = FLAGS._flags()    
        keys_list = [keys for keys in flags_dict]    
        for keys in keys_list:
            FLAGS.__delattr__(keys)

    del_all_flags(tf.flags.FLAGS)


    return meta_info, single_mse



#index = 1
#meta_info_learner("krk",str(index))