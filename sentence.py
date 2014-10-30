
import nltk



class Sentence:
    SPECIAL_TOKENS = ["sp"]

    def __init__(self, subject_id, image_id, question_id, label_question, label_emotion, original_text):
        self.subject_id = subject_id
        self.image_id = image_id
        self.question_id = question_id
        self.label_question = label_question
        self.label_emotion = label_emotion
        self.original_text = original_text.lower()  # ...ensure lowercase...

        self.raw_tokens = Sentence.get_tokens(self.original_text)
        self.clean_tokens = []

        self.interruptions = []
        self.special_tokens = []

        #...find special tokens and interruptions...
        for position in range(len(self.raw_tokens)):
            token = self.raw_tokens[position]
            special = False

            #...special tokens....
            if token[0] == "{" or token in Sentence.SPECIAL_TOKENS:
                self.special_tokens.append((position, token))
                special = True

            #.....interruptions.....
            if token[0] == "<":
                self.interruptions.append((position, token))
                special = True

            if not special:
                self.clean_tokens.append(token)

        self.tokens_pos = nltk.pos_tag(self.clean_tokens)

        """
        #Un-comment and change id's for debugging purpuses..
        if subject_id == "AS2" and image_id == "n1" and self.question_id == "q2":
            print(self.original_text)
            print(self.raw_tokens)
            print(self.clean_tokens)
            print(self.special_tokens)
            print(self.interruptions)
        """


    def __unicode__(self):
        return u"<" + self.label_question + u", " + self.label_emotion + u">" + self.original_text

    def __str__(self):
        return "<" + self.label_question + ", " + self.label_emotion + ">: " + self.original_text


    @staticmethod
    def create_from_raw_text(raw_text):

        #split the text...
        segments = raw_text.split(",")
        subject_id = segments[0]
        image_id = segments[1]
        question_id = segments[2]
        label_question = segments[3]
        label_emotion  = segments[4]
        original_text = segments[5]

        #create the sentence...
        sent = Sentence(subject_id, image_id, question_id, label_question, label_emotion, original_text)

        return sent

    @staticmethod
    def get_tokens(raw_text):
        #...extract tokens...
        raw_tokens = nltk.wordpunct_tokenize(raw_text)

        #...merge all tokens marked inside {} or <>
        #...merge tokens with ' ...
        #...also, remove * ...
        position = 0
        while position < len(raw_tokens):
            #check if special token...
            if raw_tokens[position] == "{" or raw_tokens[position] == "<":
                #the current end token...
                if raw_tokens[position] == "{":
                    end_token = "}"
                else:
                    end_token = ">"

                #merge tokens until closing symbol is found...
                merged = ""
                while position + 1 < len(raw_tokens) and raw_tokens[position + 1] != end_token:
                    merged += (" " if len(merged) > 0 else "") + raw_tokens[position + 1]
                    del raw_tokens[position + 1]

                if position + 1 < len(raw_tokens):
                    del raw_tokens[position + 1]

                merged = raw_tokens[position] + merged + end_token

                raw_tokens[position] = merged

            #join words with '
            if raw_tokens[position] == "'" and position > 0 and position + 1 < len(raw_tokens):
                #join the parts of the word separated by the tokenizer into a single word
                raw_tokens[position - 1] = raw_tokens[position - 1] + "'" + raw_tokens[position + 1]
                del raw_tokens[position + 1]
                del raw_tokens[position]

                #move to previous word...
                position -= 1

            #...remove * ...
            if raw_tokens[position] == "*":
                del raw_tokens[position]
                #move to previous word...
                position -= 1

            #move to next word...
            position += 1

        return raw_tokens