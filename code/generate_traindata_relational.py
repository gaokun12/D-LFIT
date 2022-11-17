#-------------------------------------------------------------------------------
# @author: Kun Gao
# @created: 2020/08/01
# @updated: 2021/04/30
#
#
# @desc: generate training data for each benchmarks
#-------------------------------------------------------------------------------
import sys
from source.examples.example_lf1t import generate_trans
from source.tool.create_folder import create
from source.tool.data_arff2list_fuzzy import exe as exe_fuzzy
from source.tool.data_arff2list_incom import exe as exe_incom
import tensorflow.compat.v1 as tf


# create folder to store the training datasets
create()
#Generat the fuzzy and incomplete datasets 
exe_fuzzy()
exe_incom()


