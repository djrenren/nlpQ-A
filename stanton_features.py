import nltk

# Word markers
qwords = ["who", "what", "when", "where", "why", "how"]
interjections = ["um", "hm", "uh", "mm", "well", "so", "alright"]
cognitives = ["feel", "think", "guess", "know", "remember", "like", "happen", "looks", "look like"]
uncertainties = ["approximately", "maybe", "perhaps", "probably", "might", "unknown"]

# Sound markers
sounds = ["sl", "sp", "lg", "ls", "cg"]

# Phrase Markers


class StantonFeatures:
    def __init__(self):
        pass

    def fit(self, training_set):
        #... do here any training required...
        pass

    def extract_fetures(self, sentence):
        words = nltk.word_tokenize(sentence.original_text)
        
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
        found_sounds = []
        for sound in sounds:
            if sound in words:
                found_sounds.append(sound)
        

        #return extracted features as a list...
        features = [found_qwords, found_interjections, found_cognitives, found_uncertainties, found_sounds]        
        
        print features
        return features