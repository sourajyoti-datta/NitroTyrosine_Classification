import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.feature_selection import RFE

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sequence_encoding import generate_kgap_encoding, generate_iLearn_encodings


class NitrotyrosineModelling():

    def __init__(self, train_fasta, indpe_fasta, exp_name, cw={0:1, 1:1}, shuffle=True, seed=0, rfe_learn_size=0.8, rfe_step=100, n_fold=5):
        """
        Initialization of object.
        Required inputs:
        
        :train_fasta:   path to training fasta file
        :indpe_fasta:   path to independent testing fasta file
        :exp_name:      name of experiment, all results and files generated will be stored in this folder
        
        """
        ##### all encoding parameters
        ## dictionary for recursive feature elimination use
        self.enc_dict_rf_finalFeatureCount = {
            "ASDC": 400,
            "CKSAAP4": 400,
            "DistancePair": 600,
            "Kgap4": 1000,
            "DistancePair_Kgap4": 1100,
            "Kgap4_CKSAAP4_ASDC_DistancePair": 500,
        }
        ## placeholder, values are not for actual trees
        self.enc_dict_rf_treeCount = {
            "ASDC": 1000,
            "CKSAAP4": 1000,
            "DistancePair": 1000,
            "Kgap4": 1000,
            "DistancePair_Kgap4": 1000,
            "Kgap4_CKSAAP4_ASDC_DistancePair": 1000,
        }
        self.encoding_list = ["ASDC", "DistancePair", "CKSAAP4", "Kgap4"]
        
        self.indpe_lr_cw = cw
        
        ##### global experiment parameters
        self.shuffle = shuffle
        self.seed = seed
        self.rfe_step = rfe_step    # step size for recursive feature elimination
        self.exp_name = exp_name
        self.n_fold = n_fold        # k-fold size
        self.rfe_learn_size = rfe_learn_size
        
        ##### initializing all experiment variables here, if required
        self.kfold_list = None
        self.feature_indices_dict = None
        self.full_feature_indices_dict = None
        self.train_evaluations = None
        self.indpe_evaluations = None
        
        ##### experiment file/folder parameters
        self.input_train_seq_fasta_file = train_fasta
        self.input_indpe_seq_fasta_file = indpe_fasta
        self.train_enc_csv_file = "train_{}.csv"
        self.indpe_enc_csv_file = "indpe_{}.csv"
        self.enc_data_folder_path = os.path.join(self.exp_name, "Encoded_Data")
        
        self.foldName = "folds.pickle"
        self.output_train_enc_csv_file = os.path.join(self.enc_data_folder_path, self.train_enc_csv_file)
        self.output_indpe_enc_csv_file = os.path.join(self.enc_data_folder_path, self.indpe_enc_csv_file)
        self.fold_path = os.path.join(self.exp_name, "{}fold".format(self.n_fold))
        self.model_path = os.path.join(self.fold_path, "models")
        self.fold_file = os.path.join(self.fold_path, self.foldName)
        self.ensemble_fold_model_filename = "{}_model_fold{}.hdf5"
        self.fold_lr_model_filename = "full_LR_Model_fold{}.hdf5"
        self.ensemble_model_filename = "{}_model.hdf5"
        self.lr_model_filename = "full_LR_Model.hdf5"
        
        ##### Create and set directory to save folds and model files
        if not os.path.isdir(self.model_path):
            os.makedirs(self.model_path)
        
        ##### create experiment output folders if not available
        if not os.path.isdir(self.enc_data_folder_path):
            os.makedirs(self.enc_data_folder_path)
        
        print("\n======== Nitrotyrosine classification experiment initialization successful ========")
        print("Input train fasta file:", self.input_train_seq_fasta_file)
        print("Input independent testing fasta file:", self.input_indpe_seq_fasta_file)
        print("Experiment name:", self.exp_name)
        
        
    def build_encodings(self):
        """
        Function to build encoding CSV files from the fasta files
        """
        print("\n======================== Generating all required encodings ========================")
        for enc in self.encoding_list:
            if "Kgap" in enc:
                k = int(enc.replace("Kgap", ""))
                generate_kgap_encoding(self.input_train_seq_fasta_file, self.output_train_enc_csv_file.format(enc), k)
                generate_kgap_encoding(self.input_indpe_seq_fasta_file, self.output_indpe_enc_csv_file.format(enc), k)
            else:
                generate_iLearn_encodings(self.input_train_seq_fasta_file, self.output_train_enc_csv_file.format(enc), enc)
                generate_iLearn_encodings(self.input_indpe_seq_fasta_file, self.output_indpe_enc_csv_file.format(enc), enc)
        print("========================== All encoding files generated ===========================")
        
        
    def read_encoding_file_feature_labels(self, file_path):
        data = pd.read_csv(file_path, sep=',', header=0)
        features = np.array(data.drop('SampleName', axis=1).drop('label', axis=1))
        labels = np.array(data['label'])
        return features, labels
    
    
    def pred2label(self, y_pred):
        """
        Function to convert probabilities to class, with automatic rounding
        """
        y_pred = np.round(y_pred).astype(int)
        return y_pred
    
    
    def pred2label_thres(self, y_pred, threshold):
        """
        Function to convert probabilities to class, with defined threshold
        """
        return (y_pred>=threshold).astype(int)
        
    
    def get_model(self, trees = None, cw = None):
        """
        Function to fetch model as required
        """
        if trees is not None:
            model = XGBClassifier(
                objective="binary:logistic",
                booster='gbtree',
                scale_pos_weight = 1/cw[0],
                eval_metric='logloss',
                use_label_encoder=False
            )
        else:
            model = LogisticRegression(class_weight = cw)
        return model
        
        
    def build_kfold_as_indices(self):
        """
        Build k-fold splits
        """
        print("\n===================== Building k-folds for training evaluation ====================")
        
        sample_features, sample_labels = self.read_encoding_file_feature_labels(self.output_train_enc_csv_file.format(self.encoding_list[0]))

        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=self.shuffle, random_state=self.seed)
        self.kfold_list = []
        for train_index, test_index in skf.split(sample_features, sample_labels):
            self.kfold_list.append({
                "train_indices": train_index,
                "test_indices": test_index,
            })
        
        pickle.dump(self.kfold_list, open(self.fold_file, "wb"))
        
        print("Folds:", self.n_fold)
        print("Training records:", sample_features.shape[0])
        print("Fold sizes:", [val['test_indices'].shape[0] for val in self.kfold_list])
        print("Saved to file:", self.fold_file)
        
        print("============================ k-fold generation complete ===========================")
        
        
    def execute_recursive_feature_selection(self, exp_type):
        """
        Compute optimal feature subset using recursive feature selection through elimination
        """
        print("\n============== Generating sub-feature list for each encoding using RFE ============")
        if exp_type == "kfold":
            print("============================ using 80% of training data ===========================\n")
        else:
            print("============================= using full training data ============================\n")
        
        if exp_type == "kfold":
            self.feature_indices_dict = {}
        else:
            self.full_feature_indices_dict = {}
        
        for current_dataset_variety in self.enc_dict_rf_treeCount.keys():
            
            print("Encoding variety:", current_dataset_variety)
            
            train_features = None
            train_labels = None
            
            sub_enc_list = current_dataset_variety.split("_")
            for enc in sub_enc_list:
                
                enc_features, enc_labels = self.read_encoding_file_feature_labels(self.output_train_enc_csv_file.format(enc))
                
                if train_features is not None:
                    train_features = np.concatenate((train_features, enc_features), axis=1)
                else:
                    train_features = enc_features
                    train_labels = enc_labels
            
            print("Total no. of features:", train_features.shape[1])
            
            if exp_type == "kfold":
                rfe_train_features, _, rfe_train_labels, __ = train_test_split(train_features, train_labels,
                                                                               test_size=1-self.rfe_learn_size, 
                                                                               stratify=train_labels, 
                                                                               shuffle=self.shuffle, random_state=self.seed)
            else:
                rfe_train_features = train_features
                rfe_train_labels = train_labels
            
            sub_feature_count = self.enc_dict_rf_finalFeatureCount[current_dataset_variety]
            rfe_model = self.get_model(
                trees = self.enc_dict_rf_treeCount[current_dataset_variety], 
                cw={0:1, 1:1}
            )
            selector = RFE(rfe_model, n_features_to_select=sub_feature_count, step=self.rfe_step)
            selector = selector.fit(rfe_train_features, rfe_train_labels)
            feature_indices = np.where(selector.ranking_ == 1)[0]
            
            print("Final no. of sub-features selected:", feature_indices.shape[0])
            
            if exp_type == "kfold":
                self.feature_indices_dict[current_dataset_variety] = feature_indices
            else:
                self.full_feature_indices_dict[current_dataset_variety] = feature_indices
        
        print("======================== RFE sub-feature information computed =======================")
    
    
    def generate_metrics(self, model, features, true_labels):
        """
        Predict probability and class using provided model.
        Generate metrics from predicted probability, label and true label information
        """
        
        y_pred = model.predict_proba(features)[:, 1]
        label_pred = model.predict(features)
        
        # Compute precision, recall, sensitivity, specifity, mcc
        acc = accuracy_score(true_labels, label_pred)
        mcc = matthews_corrcoef(true_labels, label_pred)

        conf = confusion_matrix(true_labels, label_pred)
        tn, fp, fn, tp = conf.ravel()
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)

        fpr, tpr, thresholds = roc_curve(true_labels, y_pred)
        auc = roc_auc_score(true_labels, y_pred)
        
        return acc, sens, spec, mcc, fpr, tpr, thresholds
        
    
    def save_evaluations(self, evaluation_dict, file):
        """
        Save evaluations dictionary as pickle file
        """
        eval_file_obj = open(file, 'wb')
        pickle.dump(evaluation_dict, eval_file_obj)
        eval_file_obj.close()
        
        
    def save_model(self, model, file):
        """
        Save trained model to file
        """
        model_file_obj = open(file, 'wb')
        pickle.dump(model, model_file_obj)
        model_file_obj.close()
        
        
    def print_evaluations(self, eval_dict, type):
        eval_df = pd.DataFrame.from_dict(eval_dict)
        if type == "kfold":
            grouped_eval_df = eval_df.groupby(["Train_Test"]).mean().filter([
                'Sensitivity', 
                'Specificity', 
                'Accuracy',
                'MCC', 
                'AUC',
            ])
            print(grouped_eval_df)
        else:
            print(eval_df[['Train_Test', 'Sensitivity', 'Specificity', 'Accuracy', 'MCC', 'AUC']])
    
    
    def kfold_training_evaluation(self):
        """
        Train the k-fold on the training data, and generate all evaluations
        """
        print("\n=================== Starting {}-fold evaluation on training dataset ==================".format(self.n_fold))
        
        self.train_evaluations = {
            "Fold" : [],
            "Train_Test" : [],
            "Accuracy" : [],
            "TPR": [],
            "FPR": [],
            "TPR_FPR_Thresholds": [],
            "AUC": [],
            "Sensitivity": [],
            "Specificity": [],
            "MCC":[]
        }
        
        i = -1
        for fold in self.kfold_list:
            
            i += 1
            print("\nTrain/Test ensemble on Fold #"+str(i+1)+".")
            
            ##################################################################################
            ##### Training Ensemble
            ##################################################################################
            
            fold_X_lr_train_proba_list = []
            fold_X_lr_train_label_list = []
            fold_X_lr_test_proba_list = []
            fold_X_lr_test_label_list = []
            
            for current_dataset_variety in self.enc_dict_rf_treeCount.keys():
                
                print("Current encoding variety:", current_dataset_variety)
                
                ##################################################################################
                ##### Fetching training data
                ##################################################################################
                train_features = None
                train_labels = None
                
                sub_enc_list = current_dataset_variety.split("_")
                for enc in sub_enc_list:
                    enc_features, enc_labels = self.read_encoding_file_feature_labels(self.output_train_enc_csv_file.format(enc))
                    if train_features is not None:
                        train_features = np.concatenate((train_features, enc_features), axis=1)
                    else:
                        train_features = enc_features
                        train_labels = enc_labels
                
                fold_train_features = train_features[fold['train_indices'], :]
                fold_train_labels = train_labels[fold['train_indices']]
                fold_test_features = train_features[fold['test_indices'], :]
                fold_test_labels = train_labels[fold['test_indices']]
                
                feature_indices = self.feature_indices_dict[current_dataset_variety]
                
                if feature_indices is not None:
                    fold_train_features = fold_train_features[:, feature_indices]
                    fold_test_features = fold_test_features[:, feature_indices]
                    
                ##################################################################################
                ##### Model training on fold
                ##################################################################################
                
                # adding random shuffling of the dataset for training purpose
                randomized_index_arr = np.arange(fold_train_features.shape[0])
                randomized_index_arr = np.random.permutation(randomized_index_arr)
                    
                # fetch model
                model = self.get_model(trees=self.enc_dict_rf_treeCount[current_dataset_variety], 
                                       cw={0:1, 1:1})

                # train model
                model.fit(X = fold_train_features[randomized_index_arr], y = fold_train_labels[randomized_index_arr])
                
                model_file_path = os.path.join(self.model_path, self.ensemble_fold_model_filename.format(current_dataset_variety, i))
                self.save_model(model, model_file_path)
                
                ##################################################################################
                ##### Prediction and metrics for TRAIN folds
                ##################################################################################
                
                y_pred = model.predict_proba(fold_train_features)[:, 1]
                label_pred = model.predict(fold_train_features)
                
                fold_X_lr_train_proba_list.append(y_pred)
                fold_X_lr_train_label_list.append(label_pred)

                ##################################################################################
                ##### Prediction and metrics for TEST fold
                ##################################################################################
                
                y_pred = model.predict_proba(fold_test_features)[:, 1]
                label_pred = model.predict(fold_test_features)
                
                fold_X_lr_test_proba_list.append(y_pred)
                fold_X_lr_test_label_list.append(label_pred)
                
            ##################################################################################
            ##### Training logistic regression model
            ##################################################################################
            
            print("Training Logistic Regression of Ensemble..")
            
            # generating features from scores
            X_lr_train_features = np.array(fold_X_lr_train_proba_list).T
            X_lr_test_features = np.array(fold_X_lr_test_proba_list).T
            
            # fetch model
            lr_model = self.get_model(trees=None, 
                                      cw={0:1, 1:1})
            
            # train model
            lr_model.fit(X = X_lr_train_features, y = fold_train_labels)

            # saving model to file
            model_file_path = os.path.join(self.model_path, self.fold_lr_model_filename.format(i))            
            self.save_model(lr_model, model_file_path)
            
            ##################################################################################
            ##### Prediction and metrics for TRAIN dataset
            ##################################################################################
            
            acc, sens, spec, mcc, fpr, tpr, thresholds = self.generate_metrics(lr_model, X_lr_train_features, fold_train_labels)
            
            self.train_evaluations["Fold"].append(i)
            self.train_evaluations["Train_Test"].append("Train")
            self.train_evaluations["Accuracy"].append(acc)
            self.train_evaluations["TPR"].append(tpr)
            self.train_evaluations["FPR"].append(fpr)
            self.train_evaluations["TPR_FPR_Thresholds"].append(thresholds)
            self.train_evaluations["AUC"].append(auc)
            self.train_evaluations["Sensitivity"].append(sens)
            self.train_evaluations["Specificity"].append(spec)
            self.train_evaluations["MCC"].append(mcc)

            ##################################################################################
            ##### Prediction and metrics for TEST dataset
            ##################################################################################
            
            acc, sens, spec, mcc, fpr, tpr, thresholds = self.generate_metrics(lr_model, X_lr_test_features, fold_test_labels)
            
            self.train_evaluations["Fold"].append(i)
            self.train_evaluations["Train_Test"].append("Test")
            self.train_evaluations["Accuracy"].append(acc)
            self.train_evaluations["TPR"].append(tpr)
            self.train_evaluations["FPR"].append(fpr)
            self.train_evaluations["TPR_FPR_Thresholds"].append(thresholds)
            self.train_evaluations["AUC"].append(auc)
            self.train_evaluations["Sensitivity"].append(sens)
            self.train_evaluations["Specificity"].append(spec)
            self.train_evaluations["MCC"].append(mcc)
    
        # save evaluations to file
        self.save_evaluations(
            self.train_evaluations, 
            os.path.join(self.exp_name, "training_evaluations_{}fold.pickle".format(self.n_fold))
        )
        
        print("\n============================= {}-fold evaluation results =============================".format(self.n_fold))
        self.print_evaluations(self.train_evaluations, "kfold")
        
        
    def independent_evaluation(self):
        """
        Train the k-fold on the training data, and generate all evaluations
        """
        print("\n============ Starting evaluation on independent dataset using full model ============")
        
        self.indpe_evaluations = {
            "Train_Test" : [],
            "Accuracy" : [],
            "TPR": [],
            "FPR": [],
            "TPR_FPR_Thresholds": [],
            "AUC": [],
            "Sensitivity": [],
            "Specificity": [],
            "MCC":[]
        }
        
        ##################################################################################
        ##### Training Ensemble
        ##################################################################################

        X_lr_train_proba_list = []
        X_lr_train_label_list = []
        X_lr_indpe_proba_list = []
        X_lr_indpe_label_list = []

        for current_dataset_variety in self.enc_dict_rf_treeCount.keys():

            print("Current encoding variety:", current_dataset_variety)
            
            feature_indices = self.full_feature_indices_dict[current_dataset_variety]
            
            ##################################################################################
            ##### Fetching training data
            ##################################################################################
            train_features = None
            train_labels = None
            
            sub_enc_list = current_dataset_variety.split("_")
            for enc in sub_enc_list:
                enc_features, enc_labels = self.read_encoding_file_feature_labels(self.output_train_enc_csv_file.format(enc))
                if train_features is not None:
                    train_features = np.concatenate((train_features, enc_features), axis=1)
                else:
                    train_features = enc_features
                    train_labels = enc_labels
                    
            train_features = train_features[:, feature_indices]
            
            ##################################################################################
            ##### Fetching independent data
            ##################################################################################
            indpe_features = None
            indpe_labels = None
            
            sub_enc_list = current_dataset_variety.split("_")
            for enc in sub_enc_list:
                enc_features, enc_labels = self.read_encoding_file_feature_labels(self.output_indpe_enc_csv_file.format(enc))
                if indpe_features is not None:
                    indpe_features = np.concatenate((indpe_features, enc_features), axis=1)
                else:
                    indpe_features = enc_features
                    indpe_labels = enc_labels
            
            indpe_features = indpe_features[:, feature_indices]
            
            ##################################################################################
            ##### Model training on fold
            ##################################################################################
            
            # adding random shuffling of the dataset for training purpose
            randomized_index_arr = np.arange(train_features.shape[0])
            randomized_index_arr = np.random.permutation(randomized_index_arr)
                
            # fetch model
            model = self.get_model(trees = self.enc_dict_rf_treeCount[current_dataset_variety], 
                                   cw = {0:1, 1:1})

            # train model
            model.fit(X = train_features[randomized_index_arr], y = train_labels[randomized_index_arr])
            
            model_file_path = os.path.join(self.exp_name, self.ensemble_model_filename.format(current_dataset_variety))
            self.save_model(model, model_file_path)

            ##################################################################################
            ##### Prediction and metrics for TRAIN dataset
            ##################################################################################
            
            y_pred = model.predict_proba(train_features)[:, 1]
            label_pred = model.predict(train_features)

            X_lr_train_proba_list.append(y_pred)
            X_lr_train_label_list.append(label_pred)

            ##################################################################################
            ##### Prediction and metrics for INDEPENDENT dataset
            ##################################################################################
            
            y_pred = model.predict_proba(indpe_features)[:, 1]
            label_pred = model.predict(indpe_features)
            
            X_lr_indpe_proba_list.append(y_pred)
            X_lr_indpe_label_list.append(label_pred)
        
        ##################################################################################
        ##### Training logistic regression model
        ##################################################################################

        print("Training Logistic Regression of Ensemble..")

        # generating features from scores
        X_lr_train_features = np.array(X_lr_train_proba_list).T
        X_lr_indpe_features = np.array(X_lr_indpe_proba_list).T

        # fetch assembling model

        lr_model = self.get_model(trees=None, 
                                  cw=self.indpe_lr_cw)

        # train model
        lr_model.fit(X = X_lr_train_features, y = train_labels)
        
        # saving model to file
        model_file_path = os.path.join(self.exp_name, self.lr_model_filename)
        self.save_model(lr_model, model_file_path)

        ##################################################################################
        ##### Prediction and metrics for TRAIN dataset
        ##################################################################################
        
        acc, sens, spec, mcc, fpr, tpr, thresholds = self.generate_metrics(lr_model, X_lr_train_features, train_labels)

        self.indpe_evaluations["Train_Test"].append("Train")
        self.indpe_evaluations["Accuracy"].append(acc)
        self.indpe_evaluations["TPR"].append(tpr)
        self.indpe_evaluations["FPR"].append(fpr)
        self.indpe_evaluations["TPR_FPR_Thresholds"].append(thresholds)
        self.indpe_evaluations["AUC"].append(auc)
        self.indpe_evaluations["Sensitivity"].append(sens)
        self.indpe_evaluations["Specificity"].append(spec)
        self.indpe_evaluations["MCC"].append(mcc)

        ##################################################################################
        ##### Prediction and metrics for TEST dataset
        ##################################################################################
        
        acc, sens, spec, mcc, fpr, tpr, thresholds = self.generate_metrics(lr_model, X_lr_indpe_features, indpe_labels)

        self.indpe_evaluations["Train_Test"].append("Independent")
        self.indpe_evaluations["Accuracy"].append(acc)
        self.indpe_evaluations["TPR"].append(tpr)
        self.indpe_evaluations["FPR"].append(fpr)
        self.indpe_evaluations["TPR_FPR_Thresholds"].append(thresholds)
        self.indpe_evaluations["AUC"].append(auc)
        self.indpe_evaluations["Sensitivity"].append(sens)
        self.indpe_evaluations["Specificity"].append(spec)
        self.indpe_evaluations["MCC"].append(mcc)
        
        # save evaluations to file
        self.save_evaluations(
            self.indpe_evaluations, 
            os.path.join(self.exp_name, "independent_evaluations.pickle")
        )
        
        print("\n=========================== Independent evaluation results ===========================")
        self.print_evaluations(self.indpe_evaluations, "indpe")
        