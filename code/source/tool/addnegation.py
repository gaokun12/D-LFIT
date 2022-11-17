#-------------------------------------------------------------------------------
# @author: Kun Gao
# @created: 2020/08/01
# @updated: 2021/03/01
#
#
# @desc: add negation into source data
#-------------------------------------------------------------------------------
import pickle
def addNegation(dataset):
    print("Transfer the data into differentiable LFIT readable data...")
    with open("../data/"+dataset+"/in_format/train_only_positive.dt", "rb") as fp:
        train = pickle.load(fp)
        new = []
        #print(train)
        for i in train:
            #print(i)
            new_data = []
            new_x = []
            neg = []
            new_y = []
            neg_y = []
            for m in i[0]:
                new_x.append(m)
                neg.append(1-m)
            for item in neg:
                new_x.append(item)

            for m in i[1]:
                new_y.append(m)
                neg_y.append(1 - m)
            for item in neg_y:
                new_y.append(item)
            new_data.append(new_x)
            new_data.append(new_y)
            #print(new_data)
            new.append(new_data)
        #print(new)
        with open("../data/"+dataset+"/in_format/train_with_negation.dt", "wb") as fp:
            pickle.dump(new, fp)
    print("Transfer succss!")