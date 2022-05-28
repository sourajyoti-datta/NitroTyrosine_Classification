import os
import pandas as pd


def convert_csv_to_fasta_ilearn_41(input_file_path, output_file_path):
        ## read raw csv file
        csv_data = pd.read_csv(input_file_path, sep='\t', header=None)
        ## set column names
        csv_data.columns = ['Sequence', 'name', 'id', 'flag', 'label_original', 'type']
        
        # changing column value of name as required
        csv_data['name'] = pd.Series([val.replace("|", "_") for val in csv_data['name']])
        
        if 'indep' in input_file_path.lower():
            data_type = 'testing'
        elif 'train' in input_file_path.lower():
            data_type = 'training'
        
        ## creating header as <unique_header>|<class_label 0/1>|<data label>
        ## structure required by iLearnPlus 
        list_header = [val1+"_"+str(val3)+"|"+str(val2)+"|"+data_type
                       for val1,val2,val3 in zip(list(csv_data['name']), 
                                            [1 if val==1 else 0 for val in csv_data['label_original']], 
                                            range(csv_data['name'].shape[0]))
                      ]
        list_seqs = list(csv_data['Sequence'])
        
        with open(output_file_path, "w") as out_file_obj:
            for header, seq in zip(list_header, list_seqs):
                #Output the header
                out_file_obj.write(">" + header + "\n")
                #Output the sequence
                out_file_obj.write(seq + "\n")


if __name__ == "__main__":

    input_csv_data_folder = "PredNTS_raw"
    output_fasta_data_folder = "PredNTS_Data_fasta_41"
    
    ##################################################################################
    ##### Convert the CSV to Fasta
    ##################################################################################

    if(not os.path.isdir(output_fasta_data_folder)):
        os.makedirs(output_fasta_data_folder)

    for root, dirs, files in os.walk(input_csv_data_folder):
        for file in files:
            input_file_path = os.path.join(root, file)
            out_fasta_file_name = ".".join([file.split(".")[0], "fasta"])
            output_file_path = os.path.join(output_fasta_data_folder, out_fasta_file_name)
            convert_csv_to_fasta_ilearn_41(input_file_path, output_file_path)

