import warnings
warnings.filterwarnings("ignore")

import argparse, os

from nitrotyrosine_modelling import NitrotyrosineModelling


#########################################################################
##### MAIN code block
#########################################################################

if __name__ == "__main__":

    #########################################################################
    ##### Fetch all command line arguments
    #########################################################################
    
    parser = argparse.ArgumentParser("python execute_nitrotyrosine_experiment.py")
    parser.add_argument('-t', '--train', help='Training sequences file path in fasta format', required=True)
    parser.add_argument('-i', '--indpe', help='Independent testing sequences file path in fasta format', required=True)
    parser.add_argument('-f', '--itf', help='Imbalanced training factor (greater than 0, 1=balanced)', required=False)
    parser.add_argument('-e', '--exp', help='Name of experiment (creates a folder where all experiment files are saved)', required=True)
    
    
    # args = parser.parse_args()
    args = parser.parse_known_args()[0]
    input_train_seq_fasta_file = str(args.train)
    input_indpe_seq_fasta_file = str(args.indpe)
    exp_name = str(args.exp)
    try:
        itf = int(str(args.itf))
    except:
        itf = 1
    assert itf > 0
    
    #########################################################################
    ##### Fetch modelling class object
    #########################################################################

    experiment_object = NitrotyrosineModelling(train_fasta=input_train_seq_fasta_file, 
                                               indpe_fasta=input_indpe_seq_fasta_file, 
                                               exp_name=exp_name, 
                                               cw={0:itf,1:1}, shuffle=True, seed=0, rfe_learn_size=0.8, rfe_step=100, n_fold=5
                                              )
    
    experiment_object.build_encodings()
    
    experiment_object.execute_recursive_feature_selection("kfold")
    experiment_object.build_kfold_as_indices()
    experiment_object.kfold_training_evaluation()
    
    experiment_object.execute_recursive_feature_selection("indpe")
    experiment_object.independent_evaluation()
    