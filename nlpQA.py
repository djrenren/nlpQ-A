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
    if not test_data is None:
        n_test_samples = test_data.shape[0]
        pred_labels = rf_classifier.predict(test_data)
        n_correct = 0
        for i in range(n_test_samples):
            if pred_labels[i] == test_labels[i]:
                n_correct += 1

        test_accuracy = n_correct / float(n_test_samples)
    else:
        test_accuracy = None
        
    return rf_classifier, train_accuracy, test_accuracy


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

    #shuffle...
    random.shuffle(training_set)

    #for t in training_set:
    #    print t.original_text

    #...create the feature extractors....
    feature_extractors = [DavilaFeatures(), StantonFeatures(), RennerFeatures()]
    for extractor in feature_extractors:
        extractor.fit(training_set)

    #...extract features...
    print("Extracting Features...")
    all_features = []
    #...also, extract labels....
    n_samples = len(training_set)
    question_labels = np.zeros(n_samples)
    emotion_labels = np.zeros(n_samples)
    #...for each sentence....
    for idx, sentence in enumerate(training_set):
        #...feature extraction....
        current_features = []
        for extractor in feature_extractors:
            current_features += extractor.extract_fetures(sentence)

        all_features.append(current_features)

        #...label extraction...
        question_labels[idx] = 1 if sentence.label_question == "Q" else 0
        emotion_labels[idx] = 1 if sentence.label_emotion == "E" else 0
        

    #...here do training of a classifier...
    #data = np.mat(all_features)
    
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

        train_samples = np.mat(all_features[:start_index] + all_features[end_index:])
        train_question_labels = np.concatenate((question_labels[:start_index],question_labels[end_index:]))
        train_emotion_labels = np.concatenate((emotion_labels[:start_index],emotion_labels[end_index:]))

        test_samples = np.mat(all_features[start_index:end_index])
        test_question_labels = question_labels[start_index:end_index]
        test_emotion_labels = emotion_labels[start_index:end_index]

        #...questions....
        rf_classifier, train_accuracy, test_accuracy = train_rf_classifier(train_samples, train_question_labels, test_samples, test_question_labels)

        question_train_acc[fold] = train_accuracy
        question_test_acc[fold] = test_accuracy

        #...emotion....
        rf_classifier, train_accuracy, test_accuracy = train_rf_classifier(train_samples, train_emotion_labels, test_samples, test_emotion_labels)

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
