from flair.models import MultiTagger
from flair.data import Sentence
from flair.tokenization import SegtokSentenceSplitter

tagger = MultiTagger.load(['flair/ner-multi-fast', 'flair/upos-multi-fast'])
# initialize sentence splitter
splitter = SegtokSentenceSplitter()

def tag(text):
    sentences = splitter.split(text)
    tagger.predict(sentences, mini_batch_size=10)

    return sentences