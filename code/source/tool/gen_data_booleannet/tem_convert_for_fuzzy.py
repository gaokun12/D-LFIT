import sys
import time
import pickle
import numpy as np
import random

'''
Tranfer .dt dataset in the formor experiment into 10-fold experimental dataset
Available for fission, budding, ara, and mam datasets. 
'''
def exe(file_name):
    #file_name = ['ara','budding','fission','mam']
    
    def gen_data_fuzzy(file_name):
        print("current file", file_name)
        time.sleep(3)
        incomplete_list = [5,10,15,20,25,30,35,40,45,50]
        repeated_times = 2
        for element in incomplete_list:
            for j in range(repeated_times):
                with open('../data/'+file_name+'/in_format/train_with_negation.dt', 'rb') as file:
                    train_data = pickle.load(file)
                    number_points = len(train_data)
                    print("Original length is", number_points)
                    random.shuffle(train_data)  
                    #print("Original data is",train_data)
                    total_number = number_points * (len(train_data[0][0]) // 2 ) 
                    print("number data length is",total_number)
                    mutation_index = [random.randint(0, total_number -1) for _ in range(int( total_number * element * 0.01 ) )]
                    print("Mutation length",len(mutation_index))
                    print("mutation rate", len(mutation_index) / total_number )
                    index = 0
                    for i in mutation_index:
                        row = i //  (len(train_data[0][0]) // 2)
                        column = i - (( len(train_data[0][0]) // 2 )* (row)) 
                        #print(row, column)
                        value = train_data[row][0][column]
                        #print("value is",value)
                        train_data[row][0][column]  = 1- value    
                        value2 = train_data[row][0][column + (len(train_data[0][0]) // 2) ] 
                        train_data[row][0][column + (len(train_data[0][0]) // 2) ] = 1 - value2
                        #print(value, value2)
                    #time.sleep(500)
                    print("After add noise")    
                    #print(train_data)
                    number_points = len(train_data)
                    print("After splitting", number_points)
                    time.sleep(5)
                    
                    number_variable =  len(train_data[0][0]) // 2
                    print(number_variable)
                    data = []
                    for i in range(len(train_data)):
                        one_data = []
                        one_data_x = []
                        one_data_y = []
                        one_data_x = train_data[i][0]
                        one_data_y = train_data[i][1][:number_variable]
                        for i in range(number_variable):
                            one_data_y.append(-2)
                        one_data.append(one_data_x)
                        one_data.append(one_data_y)
                        data.append(one_data)
                    #print(data)
                    head_variable = []   #
                    for i in range(number_variable):                
                        head_variable.append( chr( ord('a') + i ) )
                    print("Head variable is", head_variable)
                    attribute_name = []  # Store attribute and head variable in order ["Attribute name + Head variabel name"]
                    for i in range(number_variable):
                        attribute_name.append( chr( ord('a') + i ))
                    for i in range(number_variable):
                        attribute_name.append( "Not_"+chr(ord('a') + i) )
                    
                    for i in head_variable:    # Add the name of variable cause we will use it when extracting rules during writing logic into file 
                        attribute_name.append('Next_'+i)    
                    print("Attributable name  is",attribute_name)
                    file.close()



                # generate 5-fold validation data
                n_fold = 5
                random.shuffle(data)
                single_part = int(number_points / n_fold)
                for step in range(n_fold):
                    testing_data = data[(step) * single_part: (step + 1) * single_part]
                    training_data = data[:(step)*single_part] + data[(step+1) * single_part:]
                    with open('../data/'+file_name+'/in_format/fuzzy/'+str(element)+'/'+str(j)+'/'+file_name+str(step+1)+'Training.data', 'wb') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        pickle.dump(training_data,file_data)
                        file_data.close()   
                    with open('../data/'+file_name+'/in_format/fuzzy/'+str(element)+'/'+str(j)+'/'+file_name+str(step+1)+'Testing.data', 'wb') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        pickle.dump(testing_data,file_data)
                        file_data.close()   
                    # for python 2 pickle
                    with open('../data/'+file_name+'/in_format/fuzzy/'+str(element)+'/'+str(j)+'/'+file_name+str(step+1)+'Training.data.p2', 'wb') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        pickle.dump(training_data,file_data, protocol=2)
                        file_data.close()   
                    with open('../data/'+file_name+'/in_format/fuzzy/'+str(element)+'/'+str(j)+'/'+file_name+str(step+1)+'Testing.data.p2', 'wb') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        pickle.dump(testing_data,file_data, protocol = 2)
                        file_data.close()   
                        
                    with open('../data/'+file_name+'/in_format/fuzzy/'+str(element)+'/'+str(j)+'/'+file_name+str(step+1)+'Training.txt', 'w') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        print(training_data,file = file_data)
                        file_data.close()   
                    with open('../data/'+file_name+'/in_format/fuzzy/'+str(element)+'/'+str(j)+'/'+file_name+str(step+1)+'Testing.txt', 'w') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        print(testing_data,file = file_data)
                        file_data.close()   
                    with open('../data/'+file_name+'/in_format/fuzzy/'+str(element)+'/'+str(j)+'/'+file_name+'_'+str(step+1)+'.feature', 'wb') as file_feature:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        pickle.dump(attribute_name,file_feature)
                        file_data.close()
                        
                for head_index in range(len(head_variable)):
                    head_variable_index = -1 * len(head_variable) + head_index
                    target_name = attribute_name[head_variable_index]
                    create_arff(file_name, file_name, data, attribute_name, element, j, head_variable_index, target_name, len(head_variable))
                        

    def create_arff(file_name, task_name, data, attribute_name, element, repeate_times, head_variable_index, head_name, head_num):

        wp = open('../data/'+file_name+"/jrip/fuzzy/"+str(element)+ '/' + str(repeate_times)+'/'+task_name+str(element)+head_name+".arff",'w')
        train = data
        num_variable = len(attribute_name) - head_num
        wp.write("@Relation "+task_name+'\n')
        wp.write('\n')
        for i in range(num_variable):
            wp.write("@Attribute "+ str(attribute_name[i]) + " { '0', '1'}\n")
        wp.write("@Attribute " + str(attribute_name[head_variable_index])  +" { '0', '1'} \n")
        wp.write("\n")
        wp.write("@Data\n")
        for i in train:
            left = i[0]
            right = i[1]
            str_left = ''
            str_right = ''
            for i_left in left:
                str_left += str(left[i_left]) + ","
            #print(head_num + head_variable_index)
            
            str_right =  str(right[head_num + head_variable_index])     # for all relation datasets, the number of head variable is 1
            line = str(str_left + str_right)
            #print(line)
            wp.write(line+'\n')
        wp.close()

    # for i in file_name:
    gen_data_fuzzy(file_name)