FEELING_WORDS = ["happy","alive","good","understanding","great",
"playful","calm","confident","gay","courageous",
"peaceful","reliable","joyous","energetic","ease",
"easy","lucky","liberated","comfortable","amazed",
"fortunate","optimistic","pleased","free","delighted",
"provocative","encouraged","sympathetic","overjoyed","impulsive",
"clever","interested","gleeful","free","surprised",
"satisfied","thankful","frisky","content","receptive",
"important","animated","quiet","accepting","festive",
"spirited","certain","kind","ecstatic","thrilled",
"relaxed","satisfied","wonderful","serene","glad",
"free","easy","cheerful","bright",
"sunny","blessed","merry","reassured","elated",
"jubilant","love","interested","positive","strong",
"loving","concerned","eager","impulsive","considerate",
"affected","keen","affectionate","fascinated",
"earnest","sensitive","intrigued","intent",
"certain","tender","absorbed","anxious","rebellious",
"devoted","inquisitive","inspired","unique","attracted",
"nosy","determined","dynamic","passionate","snoopy",
"excited","tenacious","admiration","engrossed","enthusiastic",
"hardy","warm","curious","bold","secure",
"touched","brave","sympathy","daring",
"challenged","loved","optimistic","comforted","re-enforced","drawn","toward","confident","hopeful","unpleasant",
"feelings","angry","depressed","confused","helpless",
"irritated","lousy","upset","incapable","enraged",
"disappointed","doubtful","alone","hostile","discouraged",
"uncertain","paralyzed","insulting","ashamed","indecisive",
"fatigued","sore","powerless","perplexed","useless",
"annoyed","diminished","embarrassed","inferior","upset",
"guilty","hesitant","vulnerable","hateful","dissatisfied",
"shy","unpleasant","miserable","stupefied",
"forced","offensive","detestable","disillusioned","hesitant",
"bitter","repugnant","unbelieving","despair","aggressive",
"despicable","skeptical","frustrated","resentful","disgusting",
"distrustful","distressed","inflamed","abominable","misgiving",
"woeful","provoked","terrible","lost","pathetic",
"incensed","despair","unsure","tragic",
"infuriated","sulky","uneasy",
"cross","bad","pessimistic","dominated",
"worked","sense",
"loss","tense","boiling","fuming","indignant",
"indifferent","afraid","hurt","sad","insensitive",
"fearful","crushed","tearful","dull","terrified",
"tormented","sorrowful","nonchalant","suspicious","deprived",
"pained","neutral","anxious","pained","grief",
"reserved","alarmed","tortured","anguish","weary",
"panic","dejected","desolate","bored","nervous",
"rejected","desperate","preoccupied","scared","injured",
"pessimistic","cold","worried","offended","unhappy",
"disinterested","frightened","afflicted","lonely","lifeless",
"timid","aching","grieved","shaky","victimized",
"mournful","restless","heartbroken","dismayed","doubtful",
"agonized","threatened","appalled","cowardly","humiliated",
"quaking","wronged","menaced","alienated","wary"]

OPINION_WORDS = ['could', 'might', 'think', 'how', 'why']

FUTURE_WORDS = ['will', 'if']

class RennerFeatures:
    def __init__(self):
        pass

    def fit(self, training_set):
        #... do here any training required...
        pass

    def extract_fetures(self, sentence):
        # number of tokens that are in the feeling word list
        feeling_word_count = len([ i for i in sentence.clean_tokens if i in FEELING_WORDS])

        opinion_word_count = len([i for i in sentence.clean_tokens if i in OPINION_WORDS])

        future_word_count = len([i for i in sentence.clean_tokens if i in FUTURE_WORDS])

        contains_modal = len([i for i in sentence.tokens_pos if i[1] in ['MD', 'IN']])




        #return extracted features as a list...
        return [feeling_word_count, opinion_word_count, future_word_count, contains_modal]