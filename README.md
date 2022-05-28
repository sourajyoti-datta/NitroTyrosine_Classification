Identification of Nitrotyrosine sites in Protein peptide sequences.
- Use of various feature encodings
- Use of Recursive Feature Elimination
- Stacked modelling using Gradient Boosted Trees and Logistic Regression



Environment requirements:
Python 3.10.0

pip3 install -r requirements.txt



usage: python execute_nitrotyrosine_experiment.py [-h] -t TRAIN -i INDPE [-c CW] -e EXP

options:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        Training sequences file path in fasta format
  -i INDPE, --indpe INDPE
                        Independent testing sequences file path in fasta format
  -c CW, --cw CW        Imbalanced training factor (greater than 0, 1=balanced)
  -e EXP, --exp EXP     Name of experiment (creates a folder where all experiment files are saved)

Example (relative file path shown according to Windows, please use proper paths as required in your OS):

python execute_nitrotyrosine_experiment.py -t Data\\iNitroY_Data_fasta_41\\iNitroY_train_data.fasta -i Data\\iNitroY_Data_fasta_41\\iNitroY_independent_data.fasta -c 7000 -e nt_site_experiment_iNitroYdata

python execute_nitrotyrosine_experiment.py -t Data\\PredNTS_Data_fasta_41\\Training-datasets-PredNTS.fasta -i Data\\PredNTS_Data_fasta_41\\independent-dataset-PredNTS.fasta -e nt_site_experiment_PredNTSdata

** These experiment settings reproduce results as desired


*** Note:
Sequence fasta files must have a definite structure:
Sequence headers/names should be in this form:
<header_name>|<label>|<use_of_data>

<header_name> : Must be a unique name for the sample.
<label>: Must be a positive integer
<use_of_data>: optional, but recommended (for labelling purposes)

Sample fasta files (all data) are provided.

