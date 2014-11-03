import math
import nltk
from nltk.corpus import wordnet as wn

class DavilaFeatures:
    #TOP_EMOTION_WORDS = 1.0
    #TOP_QUESTION_WORDS = 1.0
    #N_GRAM_SIZES = [1, 2, 3, 4]
    #USE_POS = True

    #Size, Top Emotion (%), Top Question (%), POS, Total W, First Match
    #n, TopEmotion, TopQuestion, UsePOS, UseTotalW, UseFirstW
    N_GRAMS = [
        (1, 1.0, 1.0, False, True, True),
        (2, 1.0, 1.0, False, True, True),
        (1, 1.0, 1.0, True, True, True),
        (2, 1.0, 1.0, True, True, True),
    ]


    def __init__(self):
        self.top_question_ngrams = {}
        self.top_emotion_ngrams = {}

        self.majority_label = {"Q": 0, "E": 0}

    @staticmethod
    def get_roots(sentence):
        roots = []
        for idx, token in enumerate(sentence.clean_tokens):
            if sentence.tokens_pos[idx] == "VB":
                root = wn.morphy(token, wn.VERB)
            else:
                root = wn.morphy(token)

            if root is None:
                root = token

            roots.append(root)

        return roots

    @staticmethod
    def get_sentence_n_grams(sentence, n, usePOS):
        if usePOS:
            list_tokens = [pos for token, pos in sentence.tokens_pos]
        else:
            list_tokens = sentence.clean_tokens

        last_tokens = []
        n_grams = []
        for token in list_tokens:
            #compute last n-gram...
            last_tokens.append(token)
            if len(last_tokens) > n:
                del last_tokens[0]

            if len(last_tokens) == n:
                n_grams.append(list(last_tokens))

        return n_grams

    @staticmethod
    def get_weighted_n_grams(training_set, n, usePOS):
        #... extract.....
        count_n_grams = {}

        #...for each sentence....
        for sentence in training_set:
            if usePOS:
                list_tokens = [pos for token, pos in sentence.tokens_pos]
            else:
                list_tokens = sentence.clean_tokens

            last_tokens = []
            for token in list_tokens:
                #compute last n-gram...
                last_tokens.append(token)
                if len(last_tokens) > n:
                    del last_tokens[0]

                if len(last_tokens) == n:
                    key = "-".join(last_tokens)

                    if not key in count_n_grams:
                        count_n_grams[key] = {
                            "A": 0, "Q": 0,
                            "E": 0, "M": 0,
                        }

                    #count...
                    count_n_grams[key][sentence.label_emotion] += 1
                    count_n_grams[key][sentence.label_question] += 1

        #...compute Log-Likelihoods....
        likelihoods_questions = []
        likelihoods_emotions = []
        alpha = 0.1
        for key in count_n_grams:
            w_question = abs(math.log((count_n_grams[key]["A"] + alpha) / (count_n_grams[key]["Q"] + alpha), 2.0))
            w_emotion = abs(math.log((count_n_grams[key]["E"] + alpha) / (count_n_grams[key]["M"] + alpha), 2.0))

            likelihoods_questions.append((w_question, key, count_n_grams[key]["A"], count_n_grams[key]["Q"]))
            likelihoods_emotions.append((w_emotion, key, count_n_grams[key]["E"], count_n_grams[key]["M"]))

        likelihoods_questions = sorted(likelihoods_questions, reverse=True)
        likelihoods_emotions = sorted(likelihoods_emotions, reverse=True)

        #print("TOTAL N-GRAMS: " + str(len(count_n_grams.keys())) + ", n = " + str(n))

        return likelihoods_questions, likelihoods_emotions

    def fit(self, training_set):
        #... do here any training required...

        #DO N-GRAMS!!
        for idx, params in enumerate(DavilaFeatures.N_GRAMS):
            n, TopEmotion, TopQuestion, UsePOS, UseTotalW, UseFirstW = params

            likelihoods_questions, likelihoods_emotions = DavilaFeatures.get_weighted_n_grams(training_set, n, UsePOS)

            #... questions....
            n_question_words = int(TopQuestion * len(likelihoods_questions))
            self.top_question_ngrams[idx] = []
            for w, key, count_a, count_q in likelihoods_questions[:n_question_words]:
                self.top_question_ngrams[idx].append((w, key, "A" if count_a > count_q else "Q"))


            #.... emotions...
            n_emotion_words = int(TopEmotion * len(likelihoods_emotions))
            self.top_emotion_ngrams[idx] = []
            for w, key, count_e, count_m in likelihoods_emotions[:n_emotion_words]:
                self.top_emotion_ngrams[idx].append((w, key, "E" if count_e > count_m else "M"))

        #compute majority labels...
        count_label_emotion = {"E": 0, "M": 0}
        count_label_question = {"A": 0, "Q": 0}

        for sentence in training_set:
            count_label_emotion[sentence.label_emotion] += 1
            count_label_question[sentence.label_question] += 1

        self.majority_label["E"] = 1 if count_label_emotion["E"] > count_label_emotion["M"] else 0
        self.majority_label["Q"] = 1 if count_label_question["Q"] > count_label_question["A"] else 0

        #print(self.majority_label)

        #print(self.top_question_ngrams)
        #print(self.top_emotion_ngrams)


    def extract_fetures(self, sentence):
        features = []

        #count....
        count = len(sentence.clean_tokens)

        features.append(count)

        list_tokens = sentence.clean_tokens
        #list_tokens = DavilaFeatures.get_roots(sentence)

        for idx, params in enumerate(DavilaFeatures.N_GRAMS):
            n, TopEmotion, TopQuestion, UsePOS, UseTotalW, UseFirstW = params

            n_grams = DavilaFeatures.get_sentence_n_grams(sentence, n, UsePOS)
            all_keys = ["-".join(n_gram) for n_gram in n_grams]

            #check question words...
            first_emotion_label = None
            total_w = {"A": 0.0, "Q": 0.0}
            for w, key, label in self.top_question_ngrams[idx]:
                #features.append(1 if word in list_tokens else 0)
                if key in all_keys:
                    total_w[label] += w

                    if first_emotion_label is None:
                        first_emotion_label = 1 if label == "E" else 0

            if first_emotion_label is None:
                first_emotion_label = self.majority_label["E"]

            #add active features....
            if UseTotalW:
                features.append(total_w["A"])
                features.append(total_w["Q"])
            if UseFirstW:
                features.append(first_emotion_label)

            #check emotion words...
            first_question_label = None
            total_w = {"E": 0.0, "M": 0.0}
            for w, key, label in self.top_emotion_ngrams[idx]:
                #features.append(1 if word in list_tokens else 0)
                if key in list_tokens:
                    total_w[label] += w

                    if first_question_label is None:
                        first_question_label = 1 if label == "Q" else 0

            if first_question_label is None:
                first_question_label = self.majority_label["Q"]

            if UseTotalW:
                features.append(total_w["E"])
                features.append(total_w["M"])
            if UseFirstW:
                features.append(first_question_label)


        #print(features)

        #return extracted features as a list...
        return features