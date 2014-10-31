#================================================
#  Main file for problem set 3
#================================================

import sys
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

    #for t in training_set:
    #    print t.original_text

    #...create the feature extractors....
    feature_extractors = [DavilaFeatures(), StantonFeatures(), RennerFeatures()]
    for extractor in feature_extractors:
        extractor.fit(training_set)

    #...extract features...
    print("Extracting Features...")
    all_features = []
    for sentence in training_set:
        current_features = []
        for extractor in feature_extractors:
            current_features += extractor.extract_fetures(sentence)

        all_features.append(current_features)

    #...here do training of a classifier...

    print("Finished Successfully!")

#... start program here....
main()