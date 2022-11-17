#-----------------------------------------------------------------------------------------------------
# @author: Kun Gao
# @created: 2020/08/01
# @updated: 2021/02/24
#
#
# @desc: meta-info learner architecture; X is body of clauses. Y is head varibale
#        in clause. 
#-----------------------------------------------------------------------------------------------------
# coding: utf-8
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import time
import os
from datetime import timedelta
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt


class DifferILP:
    def __init__(self, batch_size=64, num_x = 1, num_y = 1,learning_rate=0.001,index_sum = 1, 
                 index_or = 1, n_epoch = 100, early_stop = False, threshold_res = 0.6,file_name = '',
                 require_improvement = 25, index = 1, print_step = 1,incomplete_flag = 0, fuzzy_flag = 0, 
                 current_index_in_parameter_list = 0, repeated_time = 0):
        
        self.batch_size = batch_size
        #self.n_num_variable = n_num_variable 
        self.num_x = num_x
        self.num_y = num_y
        self.learning_rate = learning_rate
        self.index_sum = index_sum
        self.index_or = index_or
        self.n_epoch = n_epoch
        self.early_stop = early_stop
        self.threshold_res = threshold_res
        self.file_name = file_name
        self.require_improvement = require_improvement 
        self.index = index
        self.print_step = print_step
        self.incomplete_flag = incomplete_flag
        self.fuzzy_flag = fuzzy_flag
        self.current_index_in_parameter_list = current_index_in_parameter_list
        self.repeated_time = repeated_time

        tf.reset_default_graph()
        self.build_inputs()
        self.build_infer()
        self.build_loss()
        self.build_accuarcy()
        self.build_optimizer()


    def build_inputs(self):
        with tf.name_scope('inputs'):

            self.x = tf.placeholder(tf.float32, [None, self.num_x], name='x')  # input node
            self.y_actual = tf.placeholder(tf.float32, shape=[None, self.num_y], name='y_actual')  # output node


    #Initiazation parameters
    def build_infer(self):   
        
        ###version 1, constrain weight into [0,1], using adapted sigmoid function 
        # self.W1 = tf.Variable(tf.random.truncated_normal((self.num_y, self.num_x),
        #     mean=0.5, stddev=0.25,dtype=tf.dtypes.float32),  constraint=lambda t: tf.clip_by_value(t, 0, 1)) #initializing parameters

        # one = tf.ones_like(self.y_actual)
        # layer0 = tf.matmul(self.x,  tf.transpose(self.W1[:, :])  ) - one


        # self.y_predict = layer0
        # self.out1 = tf.math.divide(tf.ones_like( self.y_predict) ,  ( tf.ones_like (self.y_predict ) + tf.math.exp ( -4 * (self.y_predict)  )))


        ##version 2: without limitation on weight, keep minus 1, using sigmoid function###
        self.W1 = tf.Variable(tf.random.truncated_normal((self.num_y, self.num_x),
            mean=0.5, stddev=0.25,dtype=tf.dtypes.float32)) #initializing parameters

        one = tf.ones_like(self.y_actual)
        layer0 = tf.matmul(self.x,  tf.transpose(self.W1[:, :])  ) - one

        self.y_predict = layer0
        self.out1 = tf.math.sigmoid(self.y_predict)



    def build_loss(self):
        with tf.name_scope('loss'):
            
            ##version 2
            cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_predict,labels=self.y_actual))

            ##version 1
            #cross_entropy_loss = tf.reduce_mean(self.y_actual * -tf.log((self.out1 + 1e-6)) + (1-self.y_actual) * -tf.log(1-self.out1 + 1e-6)) 
            
            
            # Try L1 loss or L2 loss
            self.l1_loss = tf.reduce_sum(tf.math.abs(tf.trainable_variables())) 
            self.l2_loss = tf.reduce_sum(tf.math.square(tf.trainable_variables()))
        
            coefficient = 0.0
            loss = (1- coefficient) * cross_entropy_loss + coefficient * self.l1_loss

            self.loss = loss

    def build_accuarcy(self):  ##Build MSE as accuracy



        out2 = tf.math.square(tf.math.subtract(self.out1,self.y_actual) )
        self.MSE = tf.reduce_mean(out2)


    def build_optimizer(self):

        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step, 2000, 0.9, staircase=True)
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=global_step)  # Using gardient descent to training

    # def softmax(self, z):
    #     assert len(z.shape) == 2
    #     s = np.max(z, axis=1)
    #     s = s[:, np.newaxis]  # necessary step to do broadcasting
    #     e_x = np.exp(z - s)
    #     div = np.sum(e_x, axis=1)
    #     div = div[:, np.newaxis]  # dito
    #     return e_x / div

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def nor(self, z):
        '''
        Input numpy array x, output normalized array x.
        '''
        nmax = np.max(z, axis=0)
        nmin = np.min(z, axis=0)
        return (z - nmin) / (nmax - nmin)

    def print_res_to_file(self,sess):
        # When loss stops decreasing, then records the parameter and writes into file
        w1 = self.W1.eval()
        w1 = list(w1)
        #print("The weight after training:\n", w1)
        print("sub data number is:", self.index, end = ' ')
        soft_w = []
        for i in w1:
            soft_w.append(self.softmax(np.array(i)))

        nor_soft_w = []
        for i in w1:
            nor_soft_w.append(self.nor(np.array(i)))

        threshold = self.threshold_res
        res = np.array(nor_soft_w)
        res[res > threshold] = 1
        res[res <= threshold] = 0
        # print("After threshold:", res)
        if self.incomplete_flag == 1 and self.fuzzy_flag == 0:
            matrix_file_string = "../data/" + self.file_name + "/incomplete/"+ str(self.current_index_in_parameter_list)+'/'+ str(self.repeated_time) +"/matrix_info/matrix_and_num"+self.index+".txt"
        elif self.incomplete_flag == 0 and self.fuzzy_flag == 1:
            matrix_file_string = "../data/" + self.file_name + "/fuzzy/"+ str(self.current_index_in_parameter_list)+'/'+ str(self.repeated_time) +"/matrix_info/matrix_and_num"+self.index+".txt"
        else:
            matrix_file_string = "../data/" + self.file_name + "/matrix_info/matrix_and_num"+self.index+".txt"
        weight_file = open(matrix_file_string, 'w')  # Warning! Need to change the relative path when running on the top file level
        print("The weight:\n", w1, file=weight_file)
        print("The weight after softmax:\n", soft_w, file=weight_file)
        print("The weight after softmax and normalization:\n", nor_soft_w, file=weight_file)
        print("The final res is:\n", res, file=weight_file)
        self.meta = np.count_nonzero(res == 1, axis = 1)
        print("The length_n is :\n", self.meta, file=weight_file)
        weight_file.close()



    # time recording
    def get_time_dif(self, start_time):
        end_time = time.time()
        time_dif = end_time - start_time
        return timedelta(seconds=int(round(time_dif)))

    def train(self, train, test):

        # collect training data
        fig_loss_train = []

        with  tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            sess.run(tf.initialize_local_variables())
            start_time = time.time()

            flag = False
            last_improved = 0
            last_epoch_best_loss = 10000000 - 1
            self.last_epoch_best_mse = 10000000 - 1
            last_epoch = -1
            best_loss  = 10000000
            best_acc = 10000000

            train_data = tf.data.Dataset.from_tensor_slices(train)
            train_data = train_data.batch(self.batch_size)
            iterator = train_data.make_initializable_iterator()
            next_element = iterator.get_next()

            test_data = tf.data.Dataset.from_tensor_slices(test)
            test_data = test_data.batch(self.batch_size)
            test_iterator = test_data.make_initializable_iterator()
            test_next_element = test_iterator.get_next()

            improved_str_loss = 'No!'
            improved_str_loss_in_epoch = 'No!'
            self.meta = 'ini'

            for epoch in range(self.n_epoch):
                iteration = 0
                sess.run(iterator.initializer)


                while True:
                    sess.run(test_iterator.initializer)
                    try:
                        train_case = sess.run(next_element)
                        # print("Training data instance", train_case)

                        if flag:
                            print("MSE on test data recorded...",self.last_epoch_best_mse)
                            return self.meta, self.last_epoch_best_mse

                        feed = {self.x: train_case[:, 0, :],
                                self.y_actual: train_case[:, 1, 0:self.num_y]}

                        _, xx, yy, ww, tem_lay1, predicate, msemse = sess.run([self.train_step, self.x, self.y_actual, self.W1,self.y_predict,self.out1, self.MSE] , feed_dict=feed)
                        # print("Input is:",xx)
                        # print("label output is",yy)
                        # print("Temp layer is",tem_lay1)
                        # print("predicate output is",predicate)
                        # print("weight is",ww)
                        # print("MSE is",msemse)
                        

                        loss = sess.run(self.loss, feed_dict=feed)

                        # print("Loss per iteration",loss)
                        # time.sleep(0.05)

                        if iteration % self.print_step == 0:  #per 5 iteration to check the MSE on testing dataset
                            test_step = 0
                            MSE = 0
                            while True:
                                try:
                                    test_data = sess.run(test_next_element)
                                    fig_loss_train.append(loss)
                                    MSE += sess.run(self.MSE,
                                                    feed_dict={self.x: test_data[:, 0, :], self.y_actual: test_data[:, 1, 0:self.num_y]})
                                    test_step += 1
                                    predicate_output = sess.run(self.out1, feed_dict={self.x: test_data[:, 0, :], self.y_actual: test_data[:, 1, 0:self.num_y]})
                                    y_actual = sess.run(self.y_actual,feed_dict={self.x: test_data[:, 0, :], self.y_actual: test_data[:, 1, 0:self.num_y]})
                                except tf.errors.OutOfRangeError:
                                    break
                            MSE = MSE / test_step
                            # if improved, record the MSE
                            if MSE < best_acc:
                                best_acc = MSE

                        if loss < best_loss:  #recording the best loss
                            best_loss = loss
                            last_improved = epoch
                            improved_str_loss_in_epoch = 'Improved!'
                        else:
                            improved_str_loss_in_epoch = 'No!'


                        iteration += 1

                    except tf.errors.OutOfRangeError:
                        break

                # recording MSE on testing dataset pre epoch
                if best_acc < self.last_epoch_best_mse: 
                    self.last_epoch_best_mse = best_acc
                    self.print_res_to_file(sess)
                    improved_str_loss = 'improved!'
                else:
                    improved_str_loss = 'No!'

                
                # recording loss on training dataset pre epoch
                if best_loss < last_epoch_best_loss:
                    # improved_str_loss = 'improved!'
                    last_epoch_best_loss = best_loss
                # else:
                    # improved_str_loss = 'No!'


                #print(y_actual)
                #print(predicate_output)
                print('Epoch: %d, loss: %f, MSE: %f , %s' % (epoch, last_epoch_best_loss, self.last_epoch_best_mse, improved_str_loss))

                if self.early_stop and epoch - last_improved > self.require_improvement:  # if loss stop decrsasing in self.early_epoch times, then stop training
                    print("No optimization for ", self.require_improvement + 1, " steps, auto-stop in the ", epoch," step!")
                    flag = True

            print("MSE on test data recorded...", self.last_epoch_best_mse)


            # recording the parameters after training process
            # self.print_res_to_file(sess)


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
            # plt.savefig('../data/'+self.file_name+'/'+self.index+'train.png')

            end_time = time.time()
            print("Total run time is", end_time - start_time)
            print("The MSE on meta-info learner is: ", self.last_epoch_best_mse)

            return  self.meta , self.last_epoch_best_mse
