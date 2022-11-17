import sys
import time
import pickle
import numpy as np
import random

'''
Tranfer .dt dataset in the formor experiment into 10-fold experimental dataset
Available for fission, budding, ara, and mam datasets. 
'''

file_name = ['tmax']
#file_name = ['ara','budding','fission','mam']
def gen_data(file_name):
    print("File:",file_name)
    time.sleep(3)
    with open('../data/'+file_name+'/in_format/train_with_negation.dt', 'rb') as file:
        train_data = pickle.load(file)
        number_points = len(train_data)
        #print(train_data)
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



    # generate 10-fold validation data
    n_fold = 10
    random.shuffle(data)
    single_part = int(number_points / n_fold)
    for step in range(n_fold):
        testing_data = data[(step) * single_part: (step + 1) * single_part]
        training_data = data[:(step)*single_part] + data[(step+1) * single_part:]
        with open('../data/'+file_name+'/in_format/'+file_name+str(step+1)+'Training.data', 'wb') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
            pickle.dump(training_data,file_data)
            file_data.close()   
        with open('../data/'+file_name+'/in_format/'+file_name+str(step+1)+'Testing.data', 'wb') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
            pickle.dump(testing_data,file_data)
            file_data.close()   
        with open('../data/'+file_name+'/in_format/'+file_name+str(step+1)+'Training.txt', 'w') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
            print(training_data,file = file_data)
            file_data.close()   
        with open('../data/'+file_name+'/in_format/'+file_name+str(step+1)+'Testing.txt', 'w') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
            print(testing_data,file = file_data)
            file_data.close()   
        with open('../data/'+file_name+'/in_format/'+file_name+'_'+str(step+1)+'.feature', 'wb') as file_feature:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
            pickle.dump(attribute_name,file_feature)
            file_data.close()


for i in file_name:
    gen_data(i)









  
