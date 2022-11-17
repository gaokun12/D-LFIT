import os

file_folder = ['logic_program_final', 'logic_program_tem', 'matrix_info', 'single_weights_dt', 'weight_info_final', 'weight_info_tem']
dataset = ['mutagenesis', 'uw-cse', 'amine','fission','mam','ara','budding']
#dataset = ['tmax']
incomplete = ['10','20','30','40','50','60','70','80','90','100']
fuzzy = ['5','10','15','20','25','30','35','40','45','50']
repeate_times = ['0','1']
type = ['fuzzy', 'incomplete']

def main():
    # Create directory
    # Create incomplete folder
        for i_1 in dataset:
            for i_2 in incomplete:
                for i_3 in repeate_times:
                    for i_4 in file_folder:
                        dir = '../data/'+i_1 +'/incomplete/'+i_2+'/'+i_3+'/'+i_4
                        # Create target Directory if don't exist
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                            print("Directory " , dir ,  " Created ")
                        else:    
                            print("Directory " , dir ,  " already exists")
    # Create fuzzy folder
        for i_1 in dataset:
            for i_2 in fuzzy:
                for i_3 in repeate_times:
                    for i_4 in file_folder:
                        dir = '../data/'+i_1 +'/fuzzy/'+i_2+'/'+i_3+'/'+i_4
                        # Create target Directory if don't exist
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                            print("Directory " , dir ,  " Created ")
                        else:    
                            print("Directory " , dir ,  " already exists")
    # create in_format folder
        for i_1 in dataset:
            for i_2 in type:
                if i_2 == 'fuzzy':
                    for i_3 in fuzzy:
                        for i_4 in repeate_times:
                            dir = '../data/'+i_1 +'/in_format/fuzzy/'+i_3+'/'+i_4
                            # Create target Directory if don't exist
                            if not os.path.exists(dir):
                                os.makedirs(dir)
                                print("Directory " , dir ,  " Created ")
                            else:    
                                print("Directory " , dir ,  " already exists")
                                
                if i_2 == 'incomplete':
                    for i_3 in incomplete:
                        for i_4 in repeate_times:
                            dir = '../data/'+i_1 +'/in_format/incomplete/'+i_3+'/'+i_4
                            # Create target Directory if don't exist
                            if not os.path.exists(dir):
                                os.makedirs(dir)
                                print("Directory " , dir ,  " Created ")
                            else:    
                                print("Directory " , dir ,  " already exists")
    # create others 
        for i_1 in dataset:
            for i_2 in file_folder:
                dir = '../data/'+i_1+'/'+i_2
                # Create target Directory if don't exist
                if not os.path.exists(dir):
                    os.makedirs(dir)
                    print("Directory " , dir ,  " Created ")
                else:    
                    print("Directory " , dir ,  " already exists")
                


    
def create_jrpi():
    # create in_format folder
        for i_1 in dataset:
            for i_2 in type:
                if i_2 == 'fuzzy':
                    for i_3 in fuzzy:
                        for i_4 in repeate_times:
                            dir = '../data/'+i_1 +'/jrip/fuzzy/'+i_3+'/'+i_4
                            # Create target Directory if don't exist
                            if not os.path.exists(dir):
                                os.makedirs(dir)
                                print("Directory " , dir ,  " Created ")
                            else:    
                                print("Directory " , dir ,  " already exists")
                                
                if i_2 == 'incomplete':
                    for i_3 in incomplete:
                        for i_4 in repeate_times:
                            dir = '../data/'+i_1 +'/jrip/incomplete/'+i_3+'/'+i_4
                            # Create target Directory if don't exist
                            if not os.path.exists(dir):
                                os.makedirs(dir)
                                print("Directory " , dir ,  " Created ")
                            else:    
                                print("Directory " , dir ,  " already exists")
                                

def create():
    main()
    create_jrpi()