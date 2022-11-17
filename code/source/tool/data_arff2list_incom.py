import pickle 
import numpy as np
import random
import sys
import time

def exe():
    def arff_to_list(file_name, target_feature_name):
        n = 1
        data = []
        for data_type in ['Training','Testing']: # change into (1, 1)
            f = open("../data/"+file_name+"/arff/"+target_feature_name+str(n)+data_type+".arff","r")
            #wp = open("../../../data/krk/in_format/illegal"+str(n)+"Testing.txt","w")
            line = f.readline()
            i = 0
            while line:
                if len(line) > 1  and "@" not in line:
                    i += 1
                    #print(line,end='')
                    one_data = []
                    x_data = []
                    y_data = []
                    line = line.split(",")
                    #print(line)
                    #print(i)
                    for item in line[:-1]:
                        if "-" in item:
                            x_data.append(-1)
                        else:
                            #print(item)
                            x_data.append(int(item))
                    if "-" in line[-1]:
                        y_data.append(0)  #0 is negative label standing -1
                    else: 
                        y_data.append(1)
                    for times in range(len(line) - 2):
                        y_data.append(-2)
                    
                    one_data.append((x_data))
                    one_data.append((y_data))
                    #print(len(x_data))
                    #print(len(y_data))
                    #print(one_data)
                    data.append(one_data)
                line = f.readline()
            #print(i)
            f.close()
            
        

            f = open("../data/"+file_name+"/arff/"+target_feature_name+str(n)+data_type+".arff","r")
            #wp = open("../data/krk/in_format/illegal"+str(n)+"Testing.txt","w")
            line = f.readline()
            i = 0
            feature = []
            while line:
                if "@Attribute" in line:
                    i += 1
                    #print(line,end='')
                    index_brack = line.index("{")
                    single_feature = line[len("@Attribute "):index_brack-1]
                    single_feature = single_feature.replace('(','[')
                    single_feature = single_feature.replace(')',']')
                    feature.append(single_feature)
                line = f.readline()
            print(i)
            print(feature)
            time.sleep(1)
            attribute_name = feature
            f.close()
            #print(data,file=wp,end='wp')


        print("File:",file_name)
        time.sleep(3)
        incomplete_list = [10,20,30,40,50,60,70,80,90,100]
        repeated_times = 2
        for element in incomplete_list:
            for j in range(repeated_times):
            
                
                number_points = len(data)
                ol = number_points
                print("Original length is", number_points)
                #print("before shuffle", data)
                random.shuffle(data)  
                new_data = data[0: int(element * 0.01 * number_points) ]
                print("After shuffle and split")    
                #print(data)
                number_points = len(new_data)
                print("After splitting", number_points)
                print("split rate", number_points / ol )
                #print("simple data",data[0])
                time.sleep(1)
                
                
            



                # generate 5-fold validation data
                n_fold = 5
                random.shuffle(new_data)
                single_part = int(number_points / n_fold)
                for step in range(n_fold):
                    testing_data = new_data[(step) * single_part: (step + 1) * single_part]
                    training_data = new_data[:(step)*single_part] + new_data[(step+1) * single_part:]
                    with open('../data/'+file_name+'/in_format/incomplete/'+str(element)+'/'+str(j)+'/'+target_feature_name+str(step+1)+'Training.data', 'wb') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        pickle.dump(training_data,file_data)
                        file_data.close()   
                    with open('../data/'+file_name+'/in_format/incomplete/'+str(element)+'/'+str(j)+'/'+target_feature_name+str(step+1)+'Testing.data', 'wb') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        pickle.dump(testing_data,file_data)
                        file_data.close()   
                    with open('../data/'+file_name+'/in_format/incomplete/'+str(element)+'/'+str(j)+'/'+target_feature_name+str(step+1)+'Training.txt', 'w') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        print(training_data,file = file_data)
                        file_data.close()   
                    with open('../data/'+file_name+'/in_format/incomplete/'+str(element)+'/'+str(j)+'/'+target_feature_name+str(step+1)+'Testing.txt', 'w') as file_data:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        print(testing_data,file = file_data)
                        file_data.close()   
                    with open('../data/'+file_name+'/in_format/incomplete/'+str(element)+'/'+str(j)+'/'+file_name+'_'+str(step+1)+'.feature', 'wb') as file_feature:  #save training data in format of [[[data],[label]],...,[[data],[label]]]
                        pickle.dump(attribute_name,file_feature)
                        file_data.close()
                # create arff file 
                create_arff(file_name, target_feature_name, new_data, attribute_name, element, j)
            
        
        
            
            #print(data,file=wp,end='wp')
    def create_arff(file_name, task_name, data, attribute_name, element, repeate_times):

        wp = open('../data/'+file_name+"/jrip/incomplete/"+str(element)+ '/' + str(repeate_times)+'/'+task_name+str(element)+".arff",'w')
        train = data
        num_variable = len(attribute_name)
        wp.write("@Relation "+task_name+'\n')
        wp.write('\n')
        for i in range(num_variable):
            wp.write("@Attribute "+ str(attribute_name[i]) + " { '0', '1'}\n")
        #wp.write("@attribute 'n" + chr(ord('a')+position) +"' { '0', '1'} \n")
        wp.write("\n")
        wp.write("@Data\n")
        for i in train:
            left = i[0]
            right = i[1]
            str_left = ''
            str_right = ''
            for i_left in left:
                str_left += str(left[i_left]) + ","
            str_right =  str(right[0])     # for all relation datasets, the number of head variable is 1
            line = str(str_left + str_right)
            #print(line)
            wp.write(line+'\n')
        wp.close()

            
    def data_arff_list(file_name, target_feature_name):
        arff_to_list(file_name, target_feature_name)

    #amine  great_ne
    #mutagenesis active
    #uw-cse uwcse1i

    file_name = ['amine','uw-cse','mutagenesis']
    target_feature_name = ['great_ne','uwcse1i','active']
    for i in range(len(file_name)):
        data_arff_list(file_name[i], target_feature_name[i])