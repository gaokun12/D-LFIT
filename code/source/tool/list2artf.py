import pickle
task_name = 'fission'
position = 9
fp = open("../data/" + task_name + "/train_only_positive.dt", "rb")
wp = open('../data/'+task_name+"/fission_"+str(position+1)+".arff",'w')
train = pickle.load(fp)
num_variable = len(train[0][0])
wp.write("@relation "+task_name+'\n')
for i in range(num_variable):
    wp.write("@attribute '"+ chr(ord('a') + i) + "' { '0', '1'}\n")
wp.write("@attribute 'n" + chr(ord('a')+position) +"' { '0', '1'} \n")
wp.write("@data\n")
for i in train:
    left = i[0]
    right = i[1]
    str_left = ''
    str_right = ''
    for i_left in left:
        str_left += "'" + str(left[i_left]) + "',"
    str_right = "'"+ str(right[position]) + "'"
    line = str(str_left + str_right)
    print(line)
    wp.write(line+'\n')

fp.close()
wp.close()
