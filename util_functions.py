from __future__ import print_function, division, absolute_import
from joblib import Parallel, delayed  # for parallel processes

from gensim.models.phrases import Phraser, Phrases
import re  # for preprocessing the corpus


load_dataset = lambda file,tar: tar.extractfile(file).read().decode("utf-8")

tokenize = lambda text: text.split("\n")

w2id = lambda vocabs, len_special_tokens: dict(
    zip(vocabs, range(len_special_tokens,len(vocabs)+len_special_tokens)))

id2w = lambda w2ids: dict([(v,k) for k,v in w2ids.items()])

append_EOS_token = lambda sentences: [s + ["<EOS>"] for s in sentences]

sentence_ids_to_words = lambda ids_list, id2w_dict: " ".join([id2w_dict[i] for i in ids_list])

sentence_to_ids = lambda sentence, w2id_dict:[
    w2id_dict[w] if w in w2id_dict else w2id_dict["<UNK>"] for w in sentence]

lists_of_ids = lambda sentences, w2id_dict:[
    sentence_to_ids(s, w2id_dict) for s in sentences]

def sentence_to_wordlist(raw:str, translation_table = str.maketrans("éàèùâêîôûçşöüı", "eaeuaeioucsoui")):
    raw = raw.lower().translate(translation_table)
    raw = re.sub("\d+","#", raw)
    raw = re.sub("'+","_", raw)
    return re.sub("[^A-Za-z_#]"," ", raw).split()

# converting sentences to wordlists, utilizing all the cpu cores
def tokenize_sentences(func, raw_sentences):
    return Parallel(n_jobs=-1)(
        delayed(func)(
            raw_sentence) for raw_sentence in raw_sentences)

# Grouping words like "new" "york" into one word (i.e new_york)
def bigram_sentences(tokenized_sentences):
    phrases = Phrases(tokenized_sentences)
    bigram = Phraser(phrases)
    return list(bigram[tokenized_sentences])

def get_vocabs(sentences, max_len=10000):
    vocabs = {}  # word:freq
    for s in sentences:
        for w in s:
            vocabs.setdefault(w, 0)
            vocabs[w] += 1
    return [i[0] for i in sorted(vocabs.items(), key=lambda i:i[1], reverse=True)][:max_len], len(vocabs)
