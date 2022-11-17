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
from source.tool.addnegation import addNegation
from source.tool.create_folder import create
from source.tool.gen_data_booleannet.tem_convert_for_fuzzy import exe as exe_fuzzy
from source.tool.gen_data_booleannet.tem_convert_for_incom import exe as exe_incom
import tensorflow.compat.v1 as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('file_name', 'fission', 'the name of running daaset')


file_name = FLAGS.file_name

# create folder to store the training datasets
create()
#generatin the transitions from the logic prorgam
generate_trans(file_name)
#generating the transitions with negations which differentiable LFIT can read
addNegation(file_name)
#Generat the fuzzy and incomplete datasets 
exe_fuzzy(file_name)
exe_incom(file_name)


