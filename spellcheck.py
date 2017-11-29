import enchant
from nltk.metrics import edit_distance

class SpellingReplacer(object):
    def __init__(self, max_dist=3):
        self.spell_dict = enchant.request_pwl_dict("dico.txt")
        self.max_dist = max_dist

    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        suggestions = self.spell_dict.suggest(word)

        if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word
