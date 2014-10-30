class RennerFeatures:
    def __init__(self):
        pass

    def fit(self, training_set):
        #... do here any training required...
        pass

    def extract_fetures(self, sentence):
        #example of a feature... remove this...
        count = len(sentence.clean_tokens)

        #return extracted features as a list...
        return [count]