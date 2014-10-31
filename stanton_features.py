import sentence

# Word markers
qwords = ["who", "what", "when", "where", "why", "how"]
interjections = ["um", "hm", "uh", "mm", "well", "so", "alright"]
cognitives = ["feel", "think", "guess", "know", "remember", "like", "happen", "looks"]
uncertainties = ["approximately", "maybe", "perhaps", "probably", "might", "unknown"]

# Phrase Markers


class StantonFeatures:
    def __init__(self):
        pass

    def fit(self, training_set):
        #... do here any training required...
        pass

    def extract_fetures(self, sentence):
        #words = nltk.word_tokenize(sentence.original_text)
        words = sentence.clean_tokens
        
        #print sentence

        # Searches for question words at beginning of sentence
        found_qwords = []
        for qword in qwords:
            if qword in words:
                if words.index(qword) < 3:
                    found_qwords.append(qword)

        # Searches for interjections
        found_interjections = []
        for interjection in interjections:
            if interjection in words:
                found_interjections.append(interjection)
                
        # Cognitive actions
        found_cognitives = []
        for cognitive in cognitives:
            if cognitive in words:
                found_cognitives.append(cognitive)
                
        # Words of uncertainty
        found_uncertainties = []
        for uncertainty in uncertainties:
            if uncertainty in words:
                found_uncertainties.append(uncertainty)
                
        # Sound markers
        found_sounds = sentence.special_tokens

        #return extracted features as a list...
        features = [len(found_qwords) > 0, len(found_interjections), len(found_cognitives), len(found_uncertainties), len(found_sounds)]
        
        #print features
        return features
