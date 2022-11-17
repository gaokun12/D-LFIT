#-------------------------------------------------------------------------------
# @author: Kun Gao
# @created: 2020/08/01
# @updated: 2021/03/08
#
#
# @desc: interpretor architecture
#-------------------------------------------------------------------------------
# coding: utf-8
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import time
import os
from datetime import timedelta
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
# import source.examples.example_lf1t as acc_calculator
import copy
import sys
sys.path.append("..")
print("path is",sys.path)
import check_accuracy as check_accuracy
import pickle


class InterILP:
    def __init__(self, batch_size=64,n_num_features = 1 ,n_num_variable = 1,learning_rate=0.001,
                n_epoch = 100, n_rule_logic = [], early_stop = False, index = 1, acc_keep_improved = 5, 
                num_train_case = 1, max_range = 2, file_name = '',threshold = 0.1, variable = [], target_feature_name = 'krk', train_data = [], test_data = [],
                load_saved_weight = [], current_index_in_list=0, repeated_time= 0, incomplete_flag= 0, fuzzy_flag = 0):

        self.batch_size = batch_size
        self.n_num_features = n_num_features # number of variable (features) in the body
        self.n_num_variable = n_num_variable # number of variable (features) in the head
        self.n_rule_logic = n_rule_logic # maximum possible nnumber of rules in the corresponding head, meta-info, e.g. [1,4,6]
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
        self.early_stop = early_stop
        self.num_train_case = num_train_case
        self.max_range = max_range
        self.file_name = file_name
        self.threshold = threshold
        self.variable_list= variable
        self.index = index # the number of sub dataset
        self.falg_acc_keep_improved = acc_keep_improved
        self.target_feature_name = target_feature_name
        self.train_data = train_data
        self.test_data = test_data
        self.load_saved_weight = load_saved_weight
        self.terminal_flag = 0
        self.current_index_in_list = current_index_in_list
        self.repeated_time = repeated_time
        self.incomplete_flag = incomplete_flag
        self.fuzzy_flag = fuzzy_flag
        # construct graph 
        tf.reset_default_graph()
        self.build_inputs()
        self.build_infer()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()



    def nor(self,z):
        nmax = np.max(z, axis=1)
        nmin = np.min(z, axis=1)
        nmax = nmax[:, np.newaxis]
        nmin = nmin[:, np.newaxis]
        return (z - nmin) / (nmax + 1e-7 - nmin)   # Add a perturbation


    def build_inputs(self):
        with tf.name_scope('inputs'):
            # self.bs = tf.placeholder(tf.int32, [], name="batch_size")
            self.x = tf.placeholder(tf.float32, [None, self.n_num_features], name='x')  # input node
            self.y_actual = tf.placeholder(tf.float32, shape=[None, self.n_num_variable ], name='y_actual')  # output node
            self.num_case = tf.placeholder(tf.float32)


    def build_infer(self):
        #self.W1 = tf.Variable(tf.random_uniform((1,2,4),0,1))  # first layer W
        self.W = []
        print("n_rule_logic is", self.n_rule_logic)
        add_weight_index = 0
        #if some single rule's accuracy reseaches the top, then read it directly from disk
        for i in self.n_rule_logic[:]:
            if self.load_saved_weight[add_weight_index] == 1:
                #open_weight_path = '../data/'+self.file_name+'/single_weights_dt/'+str(self.index)+'_num_'+str(add_weight_index)+'_head.dt'
                
                if self.incomplete_flag == 1 and self.fuzzy_flag == 0:            
                    open_weight_path = '../data/'+self.file_name+'/incomplete/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+'/single_weights_dt/'+str(self.index)+'_num_'+str(add_weight_index)+'_head.dt'
                    print("current_path_is", open_weight_path)
                elif self.incomplete_flag == 0 and self.fuzzy_flag == 1:
                    open_weight_path = '../data/'+self.file_name+'/fuzzy/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+'/single_weights_dt/'+str(self.index)+'_num_'+str(add_weight_index)+'_head.dt'
                else:
                    open_weight_path = '../data/'+self.file_name+'/single_weights_dt/'+str(self.index)+'_num_'+str(add_weight_index)+'_head.dt'                                    
                
                with open(open_weight_path, 'rb') as weight_file:
                    open_weight = pickle.load(weight_file)
                    weight_file.close()
                w_i = tf.Variable(open_weight, trainable = False, dtype = tf.float32)
            else:
                w_i = tf.Variable(tf.random.truncated_normal((i, 1, self.n_num_features),mean=0.5, stddev=0.25, dtype=tf.dtypes.float32), 
                constraint=lambda t: tf.clip_by_value(t, 0, self.max_range))
            self.W.append(w_i)
            add_weight_index += 1

        # adding background knowledge
        #self.W.append(tf.Variable( [[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]]] , trainable = True, dtype=tf.float32))
        # self.W.append(tf.Variable([[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]],trainable = False ,dtype = tf.float32))
        # for i in self.n_rule_logic[2:]:
        #     w_i = tf.Variable(tf.random.truncated_normal((i, 1, self.n_num_features),mean=0.5, stddev=0.25, dtype=tf.dtypes.float32), 
        #     constraint=lambda t: tf.clip_by_value(t, 0, self.max_range))
        #     self.W.append(w_i)

        # self.W1 = tf.Variable(tf.random.truncated_normal(
        #     (self.n_rule_logic, self.n_num_variable, self.n_num_variable  + self.n_num_variable),
        #     mean=0.5, stddev=0.25,dtype=tf.dtypes.float32),
        #     constraint=lambda t: tf.clip_by_value(t, 0, 1))
        # W2 = tf.Variable(tf.random.truncated_normal((n_rule_logic, n_num_variable, n_num_variable), mean=0.5, stddev=0.25, dtype=tf.dtypes.float32))
        # , constraint=lambda t: tf.clip_by_value(t, 0, 1))
        
        list_predict_or = []
        list_predict_sum = []
        forward_list_predict_or = []
        for num in range(len(self.n_rule_logic)): # go throutgh each logic rules
            one = tf.ones(shape=[self.num_case, 1], dtype=tf.dtypes.float32)
            self.list_combine = []
            self.forward_list_combine = []
            for i in range(0, self.n_rule_logic[num]):
                layer = self.W[num][i, :, :]
                self.multiple_tem_res = tf.matmul(self.x,tf.transpose(layer)) 
                multiple_tem_res = self.multiple_tem_res
                self.tem = tf.math.divide(tf.ones_like( multiple_tem_res) ,  ( tf.ones_like (multiple_tem_res ) + tf.math.exp ( -10 * (multiple_tem_res  -  one)  )))  
                # tem =  tf.math.sigmoid(tf.matmul(self.x,tf.transpose(layer)) ) 
                self.forward_tem = tf.where( multiple_tem_res >= 0.4, tf.ones_like (multiple_tem_res) , tf.zeros_like(multiple_tem_res)     )
                ## *******!!!Important!!!*********
                ## State inside the paper:
                ## Actually we do not need minus one because the sigmoid function is not exactly same with the ideal condition
                ## *******************************
                # standrad implementation
                #tem = step_funciton (tf.matmul(self.x, tf.transpose(layer)) - one)
                self.list_combine.append(self.tem)
                self.forward_list_combine.append(self.forward_tem)

            self.layer1 = tf.stack(self.list_combine, 1)
            self.forawrd_layer1 = tf.stack(self.forward_list_combine,1)

            # self.layer3 = tf.math.reduce_sum(self.layer1, 1)

            self.layer4 = tf.ones(shape=[self.num_case, self.n_rule_logic[num], 1], dtype=tf.dtypes.float32) - self.layer1
            self.forward_layer4 = tf.ones(shape=[self.num_case, self.n_rule_logic[num], 1], dtype=tf.dtypes.float32) - self.forawrd_layer1

            self.layer5 = tf.math.reduce_prod(self.layer4, 1)
            self.forward_layer5 = tf.math.reduce_prod(self.forward_layer4, 1)

            self.layer6 = tf.ones(shape=[self.num_case, 1], dtype=tf.dtypes.float32) - self.layer5
            self.forward_layer6 = tf.ones(shape=[self.num_case, 1], dtype=tf.dtypes.float32) - self.forward_layer5 

            list_predict_or.append(self.layer6)  # The or layer
            forward_list_predict_or.append(self.forward_layer6)

            # list_predict_sum.append(self.layer3) # The sum layer

        y_predict_or = tf.stack(list_predict_or, 1)
        forward_y_predicate_or = tf.stack(forward_list_predict_or, 1)
        self.y_predict_or = tf.reshape(y_predict_or, [self.num_case, self.n_num_variable])
        self.forward_y_predict_or = tf.reshape(forward_y_predicate_or, [self.num_case, self.n_num_variable])
        
        # predicate output for multi-class and single-label task
        #self.forward_y_predict_or = tf.one_hot( tf.math.argmax(self.y_predict_or,axis =1 ), depth = self.n_num_variable)
        
        #y_predict_sum = tf.stack(list_predict_sum, 1)
        #self.y_predict_sum = tf.reshape(y_predict_sum, [self.num_case, self.n_num_variable])

        #self.y_predicate_or_stand  = tf.math.sigmoid(self.y_predict_or)
        #self.neg_part_y_predicate =  1 - self.y_predicate_or_stand
        #self.y_final_predicate = tf.concat([self.y_predicate_or_stand,self.neg_part_y_predicate],1)



    def build_loss(self):
        with tf.name_scope('loss'):
            # set one loss value, penalization
            self.list_sum_one_loss = []

            for i in range(self.n_num_variable): #len(self.n_rule_logic))
                self.layer_sum_single_row = tf.math.reduce_sum(self.W[i], 2)
                self.sum_target = self.max_range * tf.ones(shape=[self.n_rule_logic[i], 1], dtype=tf.dtypes.float32)
                distance_w = self.sum_target - self.layer_sum_single_row
                self.list_sum_one_loss.append(tf.nn.l2_loss(distance_w))

            # set cross entropy as loss function
            self.loss_or = tf.reduce_mean(self.y_actual * -tf.log((1e-6+self.y_predict_or)) + 
            (1-self.y_actual) * -tf.log(1e-6+1-(self.y_predict_or)))  # cross entropy as the loss function, add a perturbation for compute nan because of mispredicatation
            #loss_sum = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_predict_sum,labels=self.y_actual))
            loss =   self.loss_or / 3  # set a coefficient
            for i in self.list_sum_one_loss:
                loss +=  i/3  # set a coefficient 0.2
            
            # Try L1 loss or L2 loss on trainable weight
            # self.l1_loss = tf.reduce_sum(tf.math.abs(tf.trainable_variables())) 
            # loss += 0.0 * self.l1_loss
            # self.l2_loss = tf.reduce_sum(tf.math.square(tf.trainable_variables()))
            # loss += 0.0 * self.l1_loss

            # Try L1 loss on predicate output for multi-class and single-label task
            self.l1_loss_res = tf.math.abs( tf.reduce_sum(tf.math.abs(self.y_predict_or)) - tf.ones_like(tf.reduce_sum(tf.math.abs(self.y_predict_or))))
            loss += self.l1_loss_res/3 # set a coefficient
            self.loss = loss

    def build_optimizer(self):
        ier =  self.num_train_case // self.batch_size
        global_step = tf.Variable(0, trainable=False)
        self.deacy_learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 5000 * ier , 0.5, staircase=True)
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.deacy_learning_rate).minimize(self.loss,global_step=global_step)  # Using gardient descent to training


    def softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]  # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]  # dito
        return e_x / div

    # time recording
    def get_time_dif(self, start_time):
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))
    
    def batch_generator(self, arr, batch_size, n_epoch):
        # print("running")
        arr = copy.copy(arr)
        n_batches = int(len(arr) / batch_size)  # Get number of batches 
        arr = arr[:batch_size * n_batches]
        #print(arr)
        #print(len(arr))
        arr = np.array(arr)
        #print(arr)
        # print(arr.shape)
        arr = arr.reshape((-1, batch_size, arr.shape[1],arr.shape[2]))  # reshape original data 
        #print("arrx is")
        #print(arr.shape)
        #print(arr)
        for i in range(n_epoch):
            # print("this epoch is", i)
            np.random.shuffle(arr)
            for n in range(0, len(arr), 1):
                tmp = arr[n]
                #print("current data is:",tmp)
                x = tmp[:,0]
                y = tmp[:,1]
                x = list(x)
                x = np.array(x)
                y = list(y)
                y = np.array(y)[:,0:self.n_num_variable]
                #print("x data",x)
                #print("y data",y)
                yield x, y, i, n, len(arr)
                
    def print_res_to_file(self, sess, final_res):
        # open file to write logic program and weight
        if self.incomplete_flag == 1 and self.fuzzy_flag == 0:
            weight_final_path = "../data/" + self.file_name + '/incomplete/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+"/weight_info_final/test_final"+str(self.index)+"final.txt"
            rules_final_path = "../data/" + self.file_name +'/incomplete/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+ '/logic_program_final/test_lfit'+str(self.index)+'final.txt'
            weight_tem_path = "../data/" + self.file_name +'/incomplete/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+ "/weight_info_tem/test_final"+str(self.index)+"temporary.txt"
            rules_tem_path = "../data/" + self.file_name + '/incomplete/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+'/logic_program_tem/test_lfit'+str(self.index)+'temporary.txt'
        elif self.incomplete_flag == 0 and self.fuzzy_flag == 1:
            weight_final_path = "../data/" + self.file_name + '/fuzzy/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+"/weight_info_final/test_final"+str(self.index)+"final.txt"
            rules_final_path = "../data/" + self.file_name +'/fuzzy/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+ '/logic_program_final/test_lfit'+str(self.index)+'final.txt'
            weight_tem_path = "../data/" + self.file_name +'/fuzzy/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+ "/weight_info_tem/test_final"+str(self.index)+"temporary.txt"
            rules_tem_path = "../data/" + self.file_name + '/fuzzy/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+'/logic_program_tem/test_lfit'+str(self.index)+'temporary.txt'
        else:
            weight_final_path = "../data/" + self.file_name + "/weight_info_final/test_final"+str(self.index)+"final.txt"
            rules_final_path = "../data/" + self.file_name + '/logic_program_final/test_lfit'+str(self.index)+'final.txt'
            weight_tem_path = "../data/" + self.file_name + "/weight_info_tem/test_final"+str(self.index)+"temporary.txt"
            rules_tem_path = "../data/" + self.file_name + '/logic_program_tem/test_lfit'+str(self.index)+'temporary.txt'
            
        if final_res == 1:
            weight_file = open(weight_final_path, 'w')
            lfit_file = open(rules_final_path, 'w')
        else:
            weight_file = open(weight_tem_path, 'w')
            lfit_file = open(rules_tem_path, 'w')
        
        for i in self.variable_list:
            print('VAR ' + i + " 0 1", file=lfit_file)
        indexw = 0        # indicate the number of head featur
        print("\n",file=lfit_file)
        for mw in self.W:
            w1 = mw.eval(sess)
            w1 = list(w1)
            # print("------" + str(indexw) + "-------")
            # print("The learning weight is", w1)
            soft_w = []
            for i in w1:
                soft_w.append(self.softmax(np.array(i)))
            # print("The weight after softmax:\n", soft_w)
            nor_w1 = []
            for i in w1:
                nor_w1.append(self.nor(np.array(i)))
            # print("The weight after normalization:\n", nor_w1)
            # print()
            # soft_w = []
            # for i in w1:
            #    soft_w.append(self.softmax(np.array(i)))
            # print(soft_w)
            print("--------" + str(indexw + 1) + "--------", file=weight_file)
            print("The weight:\n", w1, file=weight_file)
            print("The weight after softmax:\n", soft_w, file=weight_file)
            print("The weight after normalization:\n", nor_w1, file=weight_file)
            print("np array", np.array(nor_w1).reshape(self.n_rule_logic[indexw], -1), file=weight_file)
            res = np.where(np.array(nor_w1) >= self.threshold)  #######
            print("res", res, file=weight_file)
            info_1 = res[0]  # First dimention indicating target index
            info_2 = res[1]  # Second dimention indicating target index
            info_3 = res[2]  # Third dimention indicating target index

            if len(info_1) == 0:
                indexw += 1
                continue
        
            variable_index = 0
            if indexw <= len(self.variable_list) - 1:
                print(self.variable_list[indexw - self.n_num_variable] + ":-", end='', file=weight_file)       # output head variables
                print(self.variable_list[indexw - self.n_num_variable] + "(1,T) :- ", end='', file=lfit_file)  # output head variables
            # For negative head rules
            # else:
            #     print("not " + self.variable_list[indexw % len(self.variable_list)] + ":-", end='',
            #             file=weight_file)
            #     print(self.variable_list[indexw % len(self.variable_list)] + "(0,T) :- ", end='',
            #             file=lfit_file)
            for io_index in range(len(info_1)):    # len(info_1) stands the number of items meeting the specification
                if io_index + 1 > len(list(info_1)) - 1: # When readin the last element in the matrix info_1
                    if info_3[io_index] > len(self.variable_list) - 1:
                        print(
                            "not " + self.variable_list[(info_3[io_index]) % (len(self.variable_list))],
                            file=weight_file)
                        print(
                            self.variable_list[
                                (info_3[io_index]) % (len(self.variable_list))] + '(0,T-1).',
                            file=lfit_file)
                    else:
                        print(self.variable_list[info_3[io_index]], file=weight_file)
                        print(self.variable_list[info_3[io_index]] + '(1,T-1).', file=lfit_file)

                elif (info_1[io_index + 1] != info_1[io_index]):        # Triggering this condition will change another logic rule
                    if info_3[io_index] > len(self.variable_list) - 1:
                        print(
                            "not " + self.variable_list[(info_3[io_index]) % (len(self.variable_list))],
                            file=weight_file)
                        print(
                            self.variable_list[
                                (info_3[io_index]) % (len(self.variable_list))] + '(0,T-1).',
                            file=lfit_file)
                    else:
                        print(self.variable_list[info_3[io_index]], file=weight_file)
                        print(self.variable_list[info_3[io_index]] + '(1,T-1).', file=lfit_file)
                    if indexw <= len(self.variable_list) - 1:
                        print(self.variable_list[indexw - self.n_num_variable] + ":-", end='', file=weight_file)
                        print(self.variable_list[indexw - self.n_num_variable] + "(1,T) :- ", end='', file=lfit_file)
                    else:
                        print("not " + self.variable_list[indexw % len(self.variable_list)] + ":-",
                                end='', file=weight_file)
                        print(self.variable_list[indexw % len(self.variable_list)] + "(0,T) :- ",
                                end='', file=lfit_file)
                else:   # optput the element and corresponding string element
                    if info_3[io_index] > len(self.variable_list) - 1: 
                        print(
                            "not " + self.variable_list[(info_3[io_index]) % (len(self.variable_list))],
                            end=',', file=weight_file)
                        print(
                            self.variable_list[
                                (info_3[io_index]) % (len(self.variable_list))] + '(0,T-1), ', end='',
                            file=lfit_file)
                    else:
                        print(self.variable_list[info_3[io_index]], end=',', file=weight_file)
                        print(self.variable_list[info_3[io_index]] + '(1,T-1), ', end='',
                                file=lfit_file)
                

            # print("The weight after softmax", soft_w, file=weight_file)
            indexw += 1  
        weight_file.close()
        lfit_file.close()
        # print("Record program success!")


    def check_fidelity(self,sess, tem_flag , dataset , index):
        # print("Calculate Fidelity...")
        # strictly one 
        exactly_same = 0
        # condition one
        conditional_same = 0
        condition_base = 0

        test_batch_generator = self.batch_generator(self.test_data, 1 , 1 )

        for test_x, test_y , current_batch, test_iteration, test_total_num_batch in test_batch_generator:

            # print("Test x", test_x, len(test_x))
            # print("Test y", test_y)
            # print("test end epoch", test_total_num_batch)
            test_feed = {self.x: test_x, self.y_actual: test_y, self.num_case: len(test_x)}
            test_predicate_value = sess.run([self.forward_y_predict_or], feed_dict=test_feed)
            # print("NN output", test_predicate_value)
            test_predicate_value = test_predicate_value[0][0]
            test_y = test_y[0]
            # print("NN output", test_predicate_value)
            NN_value = np.where(test_predicate_value >= 0.5)[0]

            rule_predicate_value = check_accuracy.get_next_state(dataset, test_x[0].tolist(), tem_flag, index, self.n_num_variable,  current_index_in_list=self.current_index_in_list, repeated_time= self.repeated_time, incomplete_flag= self.incomplete_flag, fuzzy_flag = self.fuzzy_flag)

            rule_value = np.where(rule_predicate_value >= 0.5)[0]

            original_value = np.where( test_y >= 0.5)[0]
            
            # print("Rule  output", rule_predicate_value)
            # print("Original output", test_y)

            # print("NN output", NN_value)
            # print("Original output", original_value)
            # print("Rule output", rule_value)
            if (np.array_equal(NN_value, rule_value)):
                exactly_same += 1
            if (np.array_equal(original_value, rule_value)):
                condition_base += 1
                if (np.array_equal(rule_value, NN_value)):
                    conditional_same += 1
        
        exactly_fidelity = exactly_same / len(self.test_data)
        if condition_base == 0:
            conditional_fidelity = 0
        else:
            conditional_fidelity = conditional_same / condition_base
        # print("exact:", exactly_fidelity, "condition", conditional_fidelity)
        return exactly_fidelity, conditional_fidelity



    def train(self):

        # collect training data
        fig_loss_train = []

        with  tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(tf.initialize_all_variables())
            sess.run(tf.initialize_local_variables())
            start_time = time.time()
            require_improvement = 1
            flag = False    # Flag indicates whether training process stops in advance. 
            last_improved = 0
            last_epoch_best_val = 10000000
            last_epoch = -1
            best_loss  = last_epoch_best_val - 1
            max_rules_acc = 0
            this_update = 0
            last_update_flag = 1
            times_no_improved_acc = 0
            max_acc_NN = 0
            max_strict_fidelity = 0
            max_condition_fidelity = 0
            batch_generator = self.batch_generator(self.train_data, self.batch_size , self.n_epoch )
            for x, y , epoch, iteration, end_epoch in batch_generator:

                if flag:    
                    break

                feed = {self.x: x,
                        self.y_actual: y,
                        # self.bs: self.batch_size,
                        self.num_case: len(x)}

                _ = sess.run([self.train_step], feed_dict=feed)

                #region
                # loss_or = sess.run(self.loss_or, feed_dict=feed)
                # loss_l1 = sess.run(self.l1_loss, feed_dict=feed)
                # loss_l2 = sess.run(self.l2_loss, feed_dict=feed)
                # final_loss = sess.run(self.loss, feed_dict = feed)
                #print("The input x is", x)
                #print("The label y is", y)
                #print("The W is", WW)
                # print("Sum of parameter is", single_sum)
                # print("COmbine lsit is", COM)
                # print("The 1 is", ol1)
                # print("The 4 is", ol4)
                # print("The 5 is", ol5)
                # print("The or is", y_or)
                # print("The loss of or", loss_or, flush = True)
                # print("The L1 loss  is ", loss_l1,flush = True)
                # print("The L2 loss  is ", loss_l2,flush = True)
                # print("The total loss is", final_loss,flush = True)
                #time.sleep(0.5)
                # print("The sloss",sloss,sloss_2)
                # time.sleep(0.5)
                # # # # #
                # #
                # time.sleep(0.5)
                # y_sor = sess.run(self.y_predicate_or_stand, feed_dict=feed)
                # print("The sor is", y_sor)
                # y_nsor = sess.run(self.neg_part_y_predicate, feed_dict=feed)
                # print("The nsor is", y_nsor)
                # y_fp = sess.run(self.y_final_predicate, feed_dict=feed)
                # print("The final is", y_fp)
                #endregion

                # list_combine_out = sess.run(self.list_combine, feed_dict=feed)
                # forward_list_combine_out = sess.run(self.forward_list_combine, feed_dict=feed)
                # n_w = sess.run(self.W,feed_dict=feed)
                # print("input", x)
                # #print("new W ",n_w)
                # print("List out in list is ", list_combine_out,flush = True)
                # print("Forward out in list is ", forward_list_combine_out,flush = True)
                #y_or = sess.run(self.y_predict_or , feed_dict=feed)
                #print("The or is", y_or)
                #for_y_or = sess.run(self.forward_predicate_or_NN , feed_dict=feed)
                #print("The forward or is", for_y_or)
                #number_of_one = np.count_nonzero(for_y_or[0] == 1)
                
                # print("y is", y)

                # loss_or = sess.run(self.loss_or, feed_dict=feed)
                # final_loss = sess.run(self.loss, feed_dict = feed)
                # print("loss or", loss_or, "final_loss", final_loss)

                # mtr = sess.run(self.multiple_tem_res, feed_dict = feed)
                # st = sess.run(self.tem, feed_dict = feed)
                # sft = sess.run(self.forward_tem, feed_dict = feed)
                # print("single operation",mtr,"tem",st,"forward tem",sft)


                #time.sleep(0.1)


                if iteration % 1 == 0:
                    loss = sess.run(self.loss, feed_dict=feed)
                    fig_loss_train.append(loss)
                    o_learning_rate = sess.run(self.deacy_learning_rate)

                    # if improved, record the model
                    if loss < best_loss:
                        best_loss = loss
                        last_improved = epoch
                        #save_path = tf.train.Saver().save(sess, "../../model/demo/my_model_stop.ckpt")
                        improved_str = 'improved!'
                    else:
                        improved_str = 'no'



                    # record the training result
                    time_dif = self.get_time_dif(start_time)
                    msg = 'Epoch:{0:>4}, Iter: {1:>6}, Loss: {2:.6f}, Time: {3} {4} learning rate:{5:.6f}'
                    #print(msg.format(epoch, iteration, loss, time_dif, improved_str, o_learning_rate))

                if (iteration == end_epoch-1): # When finish one epoch, check the following conditions

                    # record loss decreases or not
                    if best_loss < last_epoch_best_val:
                        improved_str = 'Loss improved⤴️'
                        this_update = 1  # Flag indicates that whether loss decreases. When its value is 1, loss decreased in this epoch.
                        last_epoch_best_val = best_loss
                    else:
                        improved_str = 'No imporved loss⏹'
                        this_update = 0
                    
                    # writing logic and get accuracy in this epoch
                    self.print_res_to_file(sess, final_res = 0)

                    # Check Accuracy of model 
                    # print("Calculating Accuracy")
                    correct_classify_NN = 0
                    num_of_one_out = 0
                    test_batch_generator = self.batch_generator(self.test_data, 1 , 1 )
                    for test_x, test_y , current_batch, test_iteration, test_total_num_batch in test_batch_generator:
            
                        #print("Test x", test_x, len(test_x))
                        #print("Test y", test_y)
                        #print("test end epoch", test_total_num_batch)
                        test_feed = {self.x: test_x, self.y_actual: test_y, self.num_case: len(test_x)}
                        test_predicate_value = sess.run([self.forward_y_predict_or], feed_dict=test_feed)
                        #print("Testing value", test_predicate_value)
                        num_of_one_out += np.count_nonzero(test_predicate_value[0][0] == 1)
                        test_predicate_value = test_predicate_value[0][0]
                        test_y = test_y [0]
                        #print("Testing value", test_predicate_value)
                        index_value = np.where(test_predicate_value >= 0.5)[0]
                        label_value = np.where(test_y >= 0.5)[0]
                        #print("predicate index: ",index_value)
                        #print("original index:",label_value)
                        if (np.array_equal(index_value, label_value)):
                            correct_classify_NN += 1
                    accuracy_NN = correct_classify_NN / len(self.test_data)
                    num_of_one_out = num_of_one_out / len(self.test_data)

                    if accuracy_NN >= max_acc_NN:
                        max_acc_NN = accuracy_NN

                    # Check Fidelity of NN 
                    #check when tem 
                    #save max 
                    strict_fidelity, condition_fidelity = self.check_fidelity(sess, tem_flag = 1, dataset = self.file_name, index = self.index)
                    # if strict_fidelity >= max_strict_fidelity:
                    #     max_strict_fidelity = strict_fidelity
                    # if condition_fidelity >= max_condition_fidelity:
                    #     max_condition_fidelity = condition_fidelity
                    


                    # computing the accuracy of logic program according acc_caculator function in LFIT source code
                    tem_rules_acc, saved_weight_flag = check_accuracy.check_accuracy(dataset = self.file_name, index = self.index, target_feature_name = self.target_feature_name, num_class = self.n_num_variable, current_index_in_list=self.current_index_in_list, repeated_time= self.repeated_time, incomplete_flag= self.incomplete_flag, fuzzy_flag = self.fuzzy_flag)
                    print("disk weight stored info", self.load_saved_weight)
                    print("Current weight stored flag",saved_weight_flag)
                    print("When Disk weight store info is 1, then Curreny flag is 1. Otherwise, updata disk info and rerun program")
                    for check_save_weight_index in range(self.n_num_variable):
                        if saved_weight_flag[check_save_weight_index] == 1 and self.load_saved_weight[check_save_weight_index] == 0:
                            self.load_saved_weight[check_save_weight_index] = 1
                            # Store infomation into disk
                            if self.incomplete_flag == 1 and self.fuzzy_flag == 0:
                                weight_path = '../data/'+self.file_name+'/incomplete/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+'/single_weights_dt/'+str(self.index)+'_num_'+str(check_save_weight_index)+'_head.dt'
                                weight_path_txt = '../data/'+self.file_name+'/incomplete/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+'/single_weights_dt/'+str(self.index)+'_num_'+str(check_save_weight_index)+'_head.txt'
                            
                            elif self.incomplete_flag == 0 and self.fuzzy_flag == 1:
                                weight_path = '../data/'+self.file_name+'/fuzzy/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+'/single_weights_dt/'+str(self.index)+'_num_'+str(check_save_weight_index)+'_head.dt'
                                weight_path_txt = '../data/'+self.file_name+'/fuzzy/'+str(self.current_index_in_list)+"/"+str(self.repeated_time)+'/single_weights_dt/'+str(self.index)+'_num_'+str(check_save_weight_index)+'_head.txt'                               
                            else:                    
                                weight_path = '../data/'+self.file_name+'/single_weights_dt/'+str(self.index)+'_num_'+str(check_save_weight_index)+'_head.dt'
                                weight_path_txt = '../data/'+self.file_name+'/single_weights_dt/'+str(self.index)+'_num_'+str(check_save_weight_index)+'_head.txt'
                            
                            #weight_path = '../data/'+self.file_name+'/single_weights_dt/'+str(self.index)+'_num_'+str(check_save_weight_index)+'_head.dt'
                            #weight_path_txt = '../data/'+self.file_name+'/single_weights_dt/'+str(self.index)+'_num_'+str(check_save_weight_index)+'_head.txt'
                            with open(weight_path, 'wb') as weight_file_write:
                                writable_data = self.W[check_save_weight_index].eval(sess)
                                print("Writable data is", writable_data)
                                pickle.dump(writable_data, weight_file_write)
                                weight_file_write.close()
                            with open(weight_path_txt, 'w') as weight_file_write:
                                print(writable_data,file=weight_file_write)
                                weight_file_write.close()
                            self.terminal_flag = 1 # terminal because we need updata disk info 
                            flag = True
                            #time.sleep(10)
                    
                    
                    best_acc_corrs_strict_fidelity, best_acc_corre_condition_fidelity = 0 , 0 
                    if tem_rules_acc >= max_rules_acc:
                        max_rules_acc = tem_rules_acc
                        self.print_res_to_file(sess, final_res = 1)
                    
                        # check best rule corresponds fidelity
                        # save max if possible
                        best_acc_corrs_strict_fidelity, best_acc_corre_condition_fidelity = self.check_fidelity(sess, tem_flag = 0, dataset = self.file_name, index = self.index)
                        if best_acc_corrs_strict_fidelity >= max_strict_fidelity:
                            max_strict_fidelity = best_acc_corrs_strict_fidelity
                        if best_acc_corre_condition_fidelity >= max_condition_fidelity:
                            max_condition_fidelity = best_acc_corre_condition_fidelity


                    
                    if tem_rules_acc < max_rules_acc-0.9:   # if current logic program lose best logic program over 30% for self.falg_acc_keep_improved times, then training process stops.
                        times_no_improved_acc += 1
                        if times_no_improved_acc >= self.falg_acc_keep_improved:
                            flag = True

                    
                    last_update_flag = this_update

                    time_dif = self.get_time_dif(start_time)
                    time_dif = str(time_dif)
                    #step = sess.run(tf.train.get_global_step())
                    step = 0

                    # print information fecthed in curent epoch 
                    print('Epoch: %d, loss: %f, Time: %s, learning rate: %f, %s, Accuracy of rules: %f, Max rule accuracy: %f\nAccuracy of NN: %f, Max NN accuracy: %f, num_of_one: %f  \nStrictly Fidelity: %f, Condition Fidelity: %f \nBest_acca-correspond-strictly-fidelity: %f, Best_acca-correspond-condition-fidelity: %f '
                          % (epoch, last_epoch_best_val, time_dif, o_learning_rate, improved_str, tem_rules_acc, max_rules_acc, accuracy_NN, max_acc_NN,num_of_one_out, strict_fidelity,condition_fidelity,best_acc_corrs_strict_fidelity, best_acc_corre_condition_fidelity ))
                    print('Best Strictly fidelity: %f, Best condition fidelity: %f'%(max_strict_fidelity, max_condition_fidelity ))
                    # stop using early-stop strategy
                    if self.early_stop and epoch - last_improved > require_improvement:
                        print("No optimization for ", require_improvement, " steps, auto-stop in the ", epoch, " step!")
                        flag = True




            # f, ax1 = plt.subplots()
            # ax2 = ax1.twinx()
            # lns1 = ax1.plot(np.arange(len(fig_loss_train)), np.array(fig_loss_train), label="train loss")
            # x_major_locator = MultipleLocator(self.n_epoch / 10)
            # ax1.xaxis.set_major_locator(x_major_locator)
            # ax1.set_xlabel('epoch')
            # ax1.set_ylabel('training accuracy')
            # ax2.set_ylabel('valid accuracy')
            # # combine the graph
            # lns = lns1
            # labels = ["Train", "Valid"]
            # plt.legend(lns, labels, loc=7)
            # plt.savefig('res_v3.png')

            end_time = time.time()
            print("Total run time is", end_time - start_time)
        # Need to find the best accuracy and weight then we return. (Being statement of researching)
        print("Final Accuracy:",max_rules_acc)
        return max_rules_acc, max_condition_fidelity, max_acc_NN, self.load_saved_weight, self.terminal_flag





    def load(self, checkpoint):
        self.session = tf.Session()
        self.saver.restore(self.session, checkpoint)
        print('Restored from: {}'.format(checkpoint))
