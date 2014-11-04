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
    n_features = train_data.shape[1]
    rf_classifier = RandomForestClassifier(n_estimators = 100, criterion="gini", n_jobs=5,
                                           max_depth=7, max_features=min(5, n_features))
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
        n_classes = int(train_labels.max() - train_labels.min() + 1)
        confusion_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_test_samples):
            #add to confussion matrix...
            confusion_matrix[int(test_labels[i]), int(pred_labels[i])] += 1.0

            #creating list of errors...
            if pred_labels[i] == test_labels[i]:
                n_correct += 1
            else:
                test_errors.append(i)

        test_accuracy = n_correct / float(n_test_samples)
    else:
        test_accuracy = None
        confusion_matrix = None
        pred_labels = None
        
    return rf_classifier, train_accuracy, test_accuracy, test_errors, confusion_matrix, pred_labels


def extract_labels(dataset):
    #...extract labels....
    n_samples = len(dataset)
    question_labels = np.zeros(n_samples)
    emotion_labels = np.zeros(n_samples)
    #...for each sentence....
    for idx, sentence in enumerate(dataset):
        #...label extraction...
        question_labels[idx] = 1 if sentence.label_question == "Q" else 0
        emotion_labels[idx] = 1 if sentence.label_emotion == "E" else 0

    return question_labels, emotion_labels

def print_confusion_matrix(confusion, labels):
    print("\t\t\tPredicted")
    print("\t\t\t" + labels[0] + "\t" + labels[1])
    print("Expected\t" + labels[0] + " |" + "\t" + str(confusion[0, 0]) + "\t" + str(confusion[0, 1]))
    print("\t\t" + labels[1] + " |" + "\t" + str(confusion[1, 0]) + "\t" + str(confusion[1, 1]))

#===========================================
# Main Function
#===========================================
def main():
    #usage check
    if len(sys.argv) < 2:
        print("Usage: python nlpQA.py training [testing] [output]")
        print("Where")
        print("\ttraining\t= Path to file for training in CSV format")
        print("\ttesting\t= Optional, path to file for testing in CSV format")
        print("\toutput\t= Optional, path to store testing results in CSV format")
        print("")
        print("\t\tIf no testing file is specified, cross-validation will be performed, ")
        print("\t\tand no output is produced")
        print("")
        return

    #...read training set...
    print("Loading Training Set...")
    training_set = import_dataset(sys.argv[1])

    if len(sys.argv) >= 3:
        testing_set = import_dataset(sys.argv[2])
    else:
        testing_set = None

    if len(sys.argv) >= 4:
        output_filename = sys.argv[3]
    else:
        output_filename = None

    #shuffle...
    random.shuffle(training_set)

    #...extract labels....
    #...training....
    question_labels, emotion_labels = extract_labels(training_set)

    if not testing_set is None:
        #use testing set mode....
        #...testing....
        test_question_labels, test_emotion_labels = extract_labels(testing_set)

        #...create the feature extractors....
        feature_extractors = [DavilaFeatures(), StantonFeatures(), RennerFeatures()]
        
        for extractor in feature_extractors:
            extractor.fit(training_set)

        #...extract features...
        print("...Extracting Features...")
        train_samples = get_dataset_features(training_set, feature_extractors)
        test_samples = get_dataset_features(testing_set, feature_extractors)

        print("...Training... (USING " + str(train_samples.shape[1]) + " FEATURES)")

        #...questions....
        class_data = train_rf_classifier(train_samples, question_labels, test_samples, test_question_labels)
        classifier, train_accuracy, test_accuracy, question_errors, confusion_matrix, out_question_labels = class_data

        print("Question ... ")
        print("Train Accuracy:\t" + str(train_accuracy * 100.0))
        print("Test Accuracy:\t" + str(test_accuracy * 100.0))
        print("Test Confusion Matrix:")
        print_confusion_matrix(confusion_matrix, ["A", "Q"])
        print("")


        #...emotion....
        class_data = train_rf_classifier(train_samples, emotion_labels, test_samples, test_emotion_labels)
        classifier, train_accuracy, test_accuracy, emotion_errors, confusion_matrix, out_emotion_labels = class_data

        print("Emotion ... ")
        print("Train Accuracy:\t" + str(train_accuracy * 100.0))
        print("Test Accuracy:\t" + str(test_accuracy * 100.0))
        print("Test Confusion Matrix:")
        print_confusion_matrix(confusion_matrix, ["M", "E"])
        print("")

        #output samples to a file....
        if not output_filename is None:
            out_file = open(output_filename, "w")
            for idx, sentence in enumerate(testing_set):
                out_str = sentence.subject_id + "," + sentence.image_id + "," + sentence.question_id
                out_str += "," + ("Q" if out_question_labels[idx] == 1.0 else "A")
                out_str += "," + ("E" if out_emotion_labels[idx] == 1.0 else "M")
                out_str += "," + sentence.original_text
                #out_str += "\r\n"

                out_file.write(out_str)

            out_file.close()
    else:
        #do cross-validation mode...

        #...start cross-validation....
        n_samples = len(training_set)
        n_folds = 10
        fold_size = int(math.ceil(n_samples / float(n_folds)))

        print("APPLYING CROSS-VALIDATION WITH " + str(n_folds) + " folds!")

        question_train_acc = np.zeros(n_folds)
        question_test_acc = np.zeros(n_folds)
        emotion_train_acc = np.zeros(n_folds)
        emotion_test_acc = np.zeros(n_folds)

        question_confusion = np.zeros((2, 2))
        emotion_confusion = np.zeros((2, 2))


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
            rf_classifier, train_accuracy, test_accuracy, question_errors, confusion_matrix, out_labels = class_data

            question_train_acc[fold] = train_accuracy
            question_test_acc[fold] = test_accuracy
            question_confusion += confusion_matrix

            #...emotion....
            class_data = train_rf_classifier(train_samples, train_emotion_labels, test_samples, test_emotion_labels)
            rf_classifier, train_accuracy, test_accuracy, emotion_errors, confusion_matrix, out_labels = class_data

            """
            print("...Testing errors (EMOTIONS).... ")
            for e_idx in emotion_errors:
                print(str(training_set[start_index + e_idx]))
            """

            emotion_train_acc[fold] = train_accuracy
            emotion_test_acc[fold] = test_accuracy
            emotion_confusion += confusion_matrix

        question_confusion /= n_folds
        emotion_confusion /= n_folds

        print("Question ... ")
        print("Train Accuracy (AVG):\t" + str(question_train_acc.mean() * 100.0))
        print("Train Accuracy (STD):\t" + str(question_train_acc.std() * 100.0))
        print("")
        print("Test Accuracy (AVG):\t" + str(question_test_acc.mean() * 100.0))
        print("Test Accuracy (STD):\t" + str(question_test_acc.std() * 100.0))
        print("Test Average of Confusion Matrices")
        print_confusion_matrix(question_confusion, ["A", "Q"])
        print("")

        print("Emotion ... ")
        print("Train Accuracy (AVG):\t" + str(emotion_train_acc.mean()* 100.0))
        print("Train Accuracy (STD):\t" + str(emotion_train_acc.std()* 100.0))
        print("")
        print("Test Accuracy (AVG):\t" + str(emotion_test_acc.mean()* 100.0))
        print("Test Accuracy (STD):\t" + str(emotion_test_acc.std()* 100.0))
        print("Test Average of Confusion Matrices")
        print_confusion_matrix(emotion_confusion, ["M", "E"])
        print("")
    
    

    print("Finished Successfully!")

#... start program here....
if __name__ == "__main__":
    main()
