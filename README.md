**README**

This code includes three main part: test  D-LFIT on incomplete dataset, testing  D-LFIT on incomplete dataset and testing the accuracy of generated symbolic logic program by  D-LFIT. Remarkably, this code is finish based on the source code of LFIT by Tony Ribeiro (GitHub address: https://github.com/Tony-sama/pylfit). 
The python version: 3.0.0+
The Tensorflow version: 2.0+

Here are some sample commands for test D-LFIT.

0. Run the following file in the path of 'D-LFIT/code/'.

1. Generate the D-LIFT readable data:

   \- For Boolean Networks dataset:

​	 `python generate_traindata_Booleannetworks.py -file_name fission` 

​	(-file_name could be: fission, mam, ara, budding)

   \- For relational datasets:

​	`python generate_traindata_relational.py `

​	(Generating datasets for three relational dataset: 'amine','uw-cse','mutagenesis')

2. Testing the performance of D-LFIT:

   Code formate is:

   **python main.py datasets_name datasets_name incomplete? mislabelled?**
   
   Some super-parameters such as learning rate and training times can be set in `source/train_fol.py` for meta-learner and `source/interpretor_fol.py` for interpretation learner. 

- For example for Boolean datasets:

  - `python main.py fission fission 1 0`, execute D-LFIT on fission incomplete datasets
  - `python main.py budding budding 0 1`, execute D-LFIT on budding fuzzy datasets

- For example for relational datasets: 

  relation dataset name and its target predicate name:  'amine: great_ne','uw-cse: uwcse1i','mutagenesis: active'

  - `python main.py amine great_ne 1 0`, execute D-LFIT on amine incomplete datasets 
  - `python main.py mutagenesis active 0 1`, execute D-LFIT on mutagenesis mislabelled datasets 

