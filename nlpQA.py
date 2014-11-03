#================================================
#  Main file for problem set 3
#================================================

import sys
import numpy as np
import math
import random
from sklearn.ensemble import RandomForestClassifier
from sentence import Sentence
from davila_features import DavilaFeatures
from stanton_features import StantonFeatures
from renner_features import RennerFeatures

#============================================
# Read dataset file
#============================================
def import_dataset(filename):
    #....Read input file
    input_file = open(filename, 'r')
    all_lines = input_file.readlines()
    input_file.close()

    #...now, for each line...
    dataset = []
    for line_idx, line in enumerate(all_lines):
        #...preprocess, extract everything....
        try:
            #...construct the sentence with its attributs from raw text...
            new_sentence = Sentence.create_from_raw_text(line)
            dataset.append(new_sentence)
        except Exception as e:
            print("Error found while processing <" + filename + ">, line: " + str(line_idx + 1))
            print(line)
            print(e)

    return dataset

#===============================================
#  Gets the features from a training set
#===============================================
def get_dataset_features(dataset, feature_extractors):
    all_features = []

    for sentence in dataset:
        current_features = []
        for extractor in feature_extractors:
            current_features += extractor.extract_fetures(sentence)

        all_features.append(current_features)

    return np.mat(all_features)

#===============================================
#  Train a Random Forest classifier and test it
#===============================================
def train_rf_classifier(train_data, train_labels, test_data, test_labels):
    #...training....
    rf_classifier = RandomForestClassifier(n_estimators = 10, criterion="entropy", n_jobs=5)
    rf_classifier.fit(train_data, train_labels)
    
    #...training error
    n_train_samples = train_data.shape[0]
    pred_labels = rf_classifier.predict(train_data)
    n_correct = 0
    for i in range(n_train_samples):
        if pred_labels[i] == train_labels[i]:
            n_correct += 1

    train_accuracy = n_correct / float(n_train_samples)

    #...testing error
    test_errors = []
    if not test_data is None:
        n_test_samples = test_data.shape[0]
        pred_labels = rf_classifier.predict(test_data)
        n_correct = 0
        for i in range(n_test_samples):
            if pred_labels[i] == test_labels[i]:
                n_correct += 1
            else:
                test_errors.append(i)

        test_accuracy = n_correct / float(n_test_samples)
    else:
        test_accuracy = None
        
    return rf_classifier, train_accuracy, test_accuracy, test_errors


#===========================================
# Main Function
#===========================================
def main():
    #usage check
    if len(sys.argv) < 2:
        print("Usage: python nlpQA.py training")
        print("Where")
        print("\ttraining\t= Path to file for training in CSV format")
        return

    #...read training set...
    print("Loading Training Set...")
    training_set = import_dataset(sys.argv[1])

    all_apostrophe = {}
    for sentence in training_set:
        for token in sentence.clean_tokens:
            if "'" in token:
                if not token in all_apostrophe:
                    all_apostrophe[token] = 1
                else:
                    all_apostrophe[token] += 1
    print(all_apostrophe)

    #shuffle...
    random.shuffle(training_set)

    #...extract labels....
    n_samples = len(training_set)
    question_labels = np.zeros(n_samples)
    emotion_labels = np.zeros(n_samples)
    #...for each sentence....
    for idx, sentence in enumerate(training_set):
        #...label extraction...
        question_labels[idx] = 1 if sentence.label_question == "Q" else 0
        emotion_labels[idx] = 1 if sentence.label_emotion == "E" else 0

    #...start cross-validation....
    n_folds = 10
    fold_size = int(math.ceil(n_samples / float(n_folds)))

    question_train_acc = np.zeros(n_folds)
    question_test_acc = np.zeros(n_folds)
    emotion_train_acc = np.zeros(n_folds)
    emotion_test_acc = np.zeros(n_folds)
    for fold in range(n_folds):        
        #...split data....
        start_index = fold * fold_size
        end_index = (fold + 1) * fold_size

        print("Processing Fold #" + str(fold + 1) + " [" + str(start_index) + ", " + str(end_index) + "]")

        #create a subset of the training set and the testing set...
        sub_training_set = training_set[:start_index] + training_set[end_index:]
        sub_testing_set = training_set[start_index:end_index]

        #...create the feature extractors....
        feature_extractors = [DavilaFeatures(), StantonFeatures(), RennerFeatures()]
        for extractor in feature_extractors:
            extractor.fit(sub_training_set)

        #...extract features...
        print("...Extracting Features...")
        train_samples = get_dataset_features(sub_training_set, feature_extractors)
        train_question_labels = np.concatenate((question_labels[:start_index], question_labels[end_index:]))
        train_emotion_labels = np.concatenate((emotion_labels[:start_index], emotion_labels[end_index:]))

        test_samples = get_dataset_features(sub_testing_set, feature_extractors)
        test_question_labels = question_labels[start_index:end_index]
        test_emotion_labels = emotion_labels[start_index:end_index]

        print("...Training... (USING " + str(train_samples.shape[1]) + " FEATURES)")

        #...questions....
        class_data = train_rf_classifier(train_samples, train_question_labels, test_samples, test_question_labels)
        rf_classifier, train_accuracy, test_accuracy, question_errors = class_data

        question_train_acc[fold] = train_accuracy
        question_test_acc[fold] = test_accuracy

        #...emotion....
        class_data = train_rf_classifier(train_samples, train_emotion_labels, test_samples, test_emotion_labels)
        rf_classifier, train_accuracy, test_accuracy, emotion_errors = class_data

        """
        print("...Testing errors (EMOTIONS).... ")
        for e_idx in emotion_errors:
            print(str(training_set[start_index + e_idx]))
        """

        emotion_train_acc[fold] = train_accuracy
        emotion_test_acc[fold] = test_accuracy
        
    
    print("Question ... ")
    print("Train Accuracy (AVG):\t" + str(question_train_acc.mean() * 100.0))
    print("Train Accuracy (STD):\t" + str(question_train_acc.std() * 100.0))
    print("")
    print("Test Accuracy (AVG):\t" + str(question_test_acc.mean() * 100.0))
    print("Test Accuracy (STD):\t" + str(question_test_acc.std() * 100.0))
    print("")

    print("Emotion ... ")
    print("Train Accuracy (AVG):\t" + str(emotion_train_acc.mean()* 100.0))
    print("Train Accuracy (STD):\t" + str(emotion_train_acc.std()* 100.0))
    print("")
    print("Test Accuracy (AVG):\t" + str(emotion_test_acc.mean()* 100.0))
    print("Test Accuracy (STD):\t" + str(emotion_test_acc.std()* 100.0))
    print("")
    
    

    print("Finished Successfully!")

#... start program here....
if __name__ == "__main__":
    main()
