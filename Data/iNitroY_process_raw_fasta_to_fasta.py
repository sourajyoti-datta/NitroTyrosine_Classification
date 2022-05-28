import os 
import pickle
from Bio import SeqIO
from sklearn.model_selection import StratifiedKFold, train_test_split


def read_fasta_file(file_path):
    
    openFile = open(file_path)
    
    fastaSequences = SeqIO.parse(openFile, "fasta")
    
    name_list = []
    seq_list = []
    
    for fasta in fastaSequences: 
        name_list.append(fasta.id)
        seq_list.append(str(fasta.seq))

    openFile.close()
    
    return name_list, seq_list
    

def save_to_fasta(list_header, list_seqs, output_file_path):
    with open(output_file_path, "w") as out_file_obj:
        for header, seq in zip(list_header, list_seqs):
            #Output the header
            out_file_obj.write(">" + header + "\n")
            #Output the sequence
            out_file_obj.write(seq + "\n")
            
            
if __name__ == "__main__":
    
    input_data_folder = "iNitroY_raw"
    neg_data_file = "cdhit70-nitrotyr-neg.fasta"
    pos_data_file = "raw-nitrotyrosine-pos.fasta"

    output_data_folder = "iNitroY_Data_fasta_41"
    train_output_file = "iNitroY_train_data.fasta"
    indpe_output_file = "iNitroY_independent_data.fasta"
    
    if(not os.path.isdir(output_data_folder)):
        os.makedirs(output_data_folder)
    
    ##################################################################################
    ##### read positive and negative files
    ##################################################################################

    pos_file_path = os.path.join(input_data_folder, pos_data_file)
    pos_seq_name_list, pos_seq_list = read_fasta_file(pos_file_path)

    neg_file_path = os.path.join(input_data_folder, neg_data_file)
    neg_seq_name_list, neg_seq_list = read_fasta_file(neg_file_path)
    
    ##################################################################################
    ##### Fix sequence empty values and creating faster sequence headers as required
    ##################################################################################
    
    pos_seq_list = [val.replace('X', '-') for val in pos_seq_list]
    neg_seq_list = [val.replace('X', '-') for val in neg_seq_list]
    
    all_seq_name_list = pos_seq_name_list + neg_seq_name_list

    all_seq_list = pos_seq_list + neg_seq_list

    all_seq_label_list = ([1] * len(pos_seq_list)) + ([0] * len(neg_seq_list))
    
    list_header = [val1+"_"+str(val2)+"|"+str(val3) 
                   for val1, val2, val3 in zip(all_seq_name_list,
                                               range(len(all_seq_name_list)), 
                                               all_seq_label_list)]
    
    train_names, test_names, train_seqs, test_seqs = train_test_split(list_header, all_seq_list,
                                                                      test_size=0.3, stratify=all_seq_label_list, 
                                                                      shuffle=True, random_state=0)
    
    train_names = [val+"|training" for val in train_names]
    test_names = [val+"|testing" for val in test_names]
    
    save_to_fasta(train_names, train_seqs, os.path.join(output_data_folder, train_output_file))
    save_to_fasta(test_names, test_seqs, os.path.join(output_data_folder, indpe_output_file))
    
    
    
    
    
    
    
    