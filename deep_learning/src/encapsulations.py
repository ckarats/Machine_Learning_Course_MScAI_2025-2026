import preprocessor as tweet_preprocessoR

class Tweet(object):

    def __init__(self, post_id, tweet_content,
                 harrasment=None,
                 indirectH=None,
                 physicalH=None,
                 sexualH=None ):
        self.post_id = post_id
        self.text = tweet_content
        self.target = harrasment
        self.indirectH = indirectH
        self.physicalH = physicalH
        self.sexualH = sexualH

        self.label = self.target
        self.tokens = [self.clean_token(x) for x in self.tokenize(tweet_content, True).split() if len(x) > 0]
        self.text_size = self.get_text_size()
        self.softmax_label = self.get_label_for_softmax()

    def get_text_size(self):
        return len(self.tokens)

    @staticmethod
    def clean_token(tkn):
        if tkn not in ["\'s", "\'ve", "n\'t", "\'d", "\'ll"]:
            tkn = tkn.replace("'", "")
        return tkn

    @staticmethod
    def tokenize(text, lower=True):
        cleaned_text = tweet_preprocessor.tokenize(text)
        return cleaned_text.lower() if lower else cleaned_text

    def get_label_for_softmax(self):
        if self.label == 0:
            return 0
        else:
            if self.indirectH == 1:
                return 1
            elif self.sexualH == 1:
                return 2
            else:
                return 3








