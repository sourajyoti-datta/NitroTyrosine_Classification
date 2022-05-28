import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import pandas as pd
import numpy as np
from ilearnplus.util.FileProcessing import Descriptor


def clean_seq_fasta_file(input_seq_fasta_file, output_seq_fasta_file):
    """
    Remove X and - characters from peptide sequences, required for correct MathFeature encoding
    """
    seq_file_input_obj = open(input_seq_fasta_file, 'r')
    Lines = seq_file_input_obj.readlines()

    fixed_Lines = [line.replace("-", "").replace("X", "") if '>' not in line 
                   else line 
                   for line in Lines]

    seq_file_output_obj = open(output_seq_fasta_file, 'w')
    seq_file_output_obj.writelines(fixed_Lines)
    seq_file_output_obj.close()


def generate_kgap_encoding(input_seq_fasta_file, output_csv_file, kgap_max):
    """
    Kgap encoding function, using MathFeature
    """
    print("===================================================================================")
    print("Input file:", input_seq_fasta_file)
    print("Output file:", output_csv_file)
    print("Encoding:", "Kgap"+str(kgap_max))
    
    ## temporary folder to save intermediate generated files
    temp_output_folder = "temp_files"
    if(os.path.isdir(temp_output_folder)):
        shutil.rmtree(temp_output_folder)
    os.makedirs(temp_output_folder)
        
    temp_output_file = os.path.join(temp_output_folder, "temp_kgap_{}.csv")
    
    ## cleaning the fasta file, required by MathFeature encoding
    input_seq_fasta_file_cleaned = os.path.join(temp_output_folder, os.path.split(input_seq_fasta_file)[-1])
    clean_seq_fasta_file(input_seq_fasta_file, input_seq_fasta_file_cleaned)
    
    ## generating temp kgap file, for every kgap value in range (0...kgap_max)
    for kgap in range(kgap_max+1):
        os.system("python Kgap.py -i {} -o {} -l train -k {} -bef 1 -aft 1 -seq 3".format(input_seq_fasta_file_cleaned, 
                                                                                          temp_output_file.format(kgap), kgap))
                                                                                          
    ## merging all kgap data into single pandas dataframe
    for kgap in range(kgap_max+1):
        current_data_filepath = temp_output_file.format(kgap)
        current_data = pd.read_csv(current_data_filepath, sep=',', header=0)
        current_data = current_data.drop('label', axis=1)
        if kgap == 0:
            full_data = current_data
        else:
            full_data = pd.merge(
                full_data,
                current_data,
                how="inner",
                on='nameseq'
            )
            
    ## delete temporary folder
    shutil.rmtree(temp_output_folder)
    
    ## edit full_data to match iLearn format CSV
    label = [val.split('|')[1] for val in full_data['nameseq']]
    SampleName = [val.split('|')[0] for val in full_data['nameseq']]
    full_data.insert(0, 'label', label)
    full_data.insert(0, 'SampleName', SampleName)
    full_data.drop('nameseq', axis=1, inplace=True)
    
    ## save pandas dataframe to csv
    full_data.to_csv(output_csv_file, index=False, header=True)
    
    print("--- Encoding file generated ---")


def generate_iLearn_encodings(input_seq_fasta_file, output_csv_file, enc):
    """
    All other Encoding function, using iLearnPlus
    """
    print("===================================================================================")
    print("Input file:", input_seq_fasta_file)
    print("Output file:", output_csv_file)
    print("Encoding:", enc)
    
    ## define encoding parameters
    if enc == "ASDC":
        param_dict = {}
    elif enc == "CKSAAP4":
        param_dict = {"kspace":4}
    elif enc == "DistancePair":
        param_dict = {"cp":"cp(20)", "distance":6}
    
    ## generate descriptor object
    seq_obj = Descriptor(file = input_seq_fasta_file, kw = param_dict)
    
    ## generate encoding
    if enc == "ASDC":
        seq_obj.Protein_ASDC()
    elif enc == "CKSAAP4":
        seq_obj.Protein_CKSAAP()
    elif enc == "DistancePair":
        seq_obj.Protein_DistancePair()
    
    # np.savetxt(output_csv_file, seq_obj.encoding_array, delimiter=",")
    np.savetxt(output_csv_file, seq_obj.encoding_array, fmt="%s", delimiter=",")
    
    ## generate encoding numpy array to required pandas dataframe
    # full_data = pd.DataFrame(seq_obj.encoding_array[1:, :], 
    #                          columns=seq_obj.encoding_array[0,:])
                             
    ## save pandas dataframe to csv
    # full_data.to_csv(output_csv_file, index=False, header=True)
    
    print("--- Encoding file generated ---")
    