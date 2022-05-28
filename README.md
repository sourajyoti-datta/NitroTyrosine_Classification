# Identification of Nitrotyrosine sites in Protein peptide sequences.

## Experiment overview:
- Use of various feature encodings
- Use of Recursive Feature Elimination
- Stacked modelling using Gradient Boosted Trees and Logistic Regression


## Environment requirements:
Python 3.10.0
```sh  
pip3 install -r requirements.txt
```

## Usage
```sh  
python execute_nitrotyrosine_experiment.py [-h] -t TRAIN -i INDPE [-f ITF] -e EXP
```

### Options:
  - -h,       --help            show this help message and exit
  - -t TRAIN, --train TRAIN     Training sequences file path in fasta format
  - -i INDPE, --indpe INDPE     Independent testing sequences file path in fasta format
  - -f ITF,    --itf ITF           Imbalanced training factor (greater than 0, 1=balanced)
  - -e EXP,   --exp EXP         Name of experiment (creates a folder where all experiment files are saved)

### Examples:
```sh
python execute_nitrotyrosine_experiment.py -t Data\\iNitroY_Data_fasta_41\\iNitroY_train_data.fasta -i Data\\iNitroY_Data_fasta_41\\iNitroY_independent_data.fasta -f 7000 -e nt_site_experiment_iNitroYdata

python execute_nitrotyrosine_experiment.py -t Data\\PredNTS_Data_fasta_41\\Training-datasets-PredNTS.fasta -i Data\\PredNTS_Data_fasta_41\\independent-dataset-PredNTS.fasta -e nt_site_experiment_PredNTSdata
```

(relative file path shown according to Windows, please use proper paths as required in your OS)


### *Note:*

Sequence headers/names should be in this form: <header_name>|<class_labels>|<use_of_data>

  - <header_name> : Must be a unique name for the sample.
  - <class_labels>: Must be a positive integer
  - <use_of_data>: optional, but recommended (for labelling purposes)

Sample fasta files are provided.

