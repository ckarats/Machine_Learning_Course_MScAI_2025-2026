from globals import GLOVE_EMBEDDINGS_PATH
from encapsulations import Tweet
from tqdm import tqdm
import numpy as np
import pandas as pd


def get_embeddings(tokens, path=GLOVE_EMBEDDINGS_PATH):
    embeddings_dict = {}
    with open(path,'r', encoding='utf8') as f:
        for line in tqdm(f):
                values = line.strip().split(" ")
                try:
                    tokens[values[0]]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_dict[values[0]] = coefs
                except KeyError:
                    pass
    return embeddings_dict


def create_freq_vocabulary(tokenized_texts):
    token_dict = {}
    for text in tokenized_texts:
        for token in text:
            try:
                token_dict[token] += 1
            except KeyError:
                token_dict[token] = 1
    return token_dict


def get_frequent_words(token_dict, min_freq):
    return {x: None for x in token_dict if token_dict[x] >= min_freq}


def get_unique_tokens(posts, min_freq):
    tokenized_texts = [x.tokens for x in posts]
    voc = create_freq_vocabulary(tokenized_texts)
    print("tokens found in training data set:", len(voc))
    freq_words = get_frequent_words(voc, min_freq)
    print("tokens with frequency >= %d: %d" % (min_freq, len(freq_words)))
    return freq_words


def create_final_dictionary(freq_words, embeddings_dict, unk_token, pad_token):
    words = list(set(freq_words).intersection(embeddings_dict.keys()))
    print("embedded tokens: %d" % (len(words)))
    words = [pad_token, unk_token] + words
    return {w: i for i, w in enumerate(words)}


def get_embeddings_matrix(word_dict, embeddings_dict, size):
    embs = np.zeros(shape=(len(word_dict), size))
    for word in word_dict:
        try:
            embs[word_dict[word]] = embeddings_dict[word]
        except KeyError:
            print('no embedding for: ', word)
    embs[1] = np.mean(embs[2:])
    return embs


def get_indexed_value(w2i, word, unk_token):
    try:
        return w2i[word]
    except KeyError:
        return w2i[unk_token]


def get_indexed_text(w2i, words, unk_token):
    return [get_indexed_value(w2i, word, unk_token) for word in words]


def pad_text(tokenized_text, maxlen, pad_tkn):
    if len(tokenized_text) < maxlen:
        return [pad_tkn] * (maxlen - len(tokenized_text)) + tokenized_text
    else:
        return tokenized_text[len(tokenized_text) - maxlen:]


def load_data(csv_file):
    data = pd.read_csv(csv_file)
    data = data.fillna({'tweet_content': "nan"})
    data['post_id'] = np.arange(len(data))
    return data


def get_comments(df, is_train=True):
    comments = []
    if is_train:
        for i, row in df.iterrows():

            post = Tweet(post_id=row['post_id'],
                         tweet_content=row['tweet_content'],
                         harrasment=row['harassment'],
                         indirectH=row['IndirectH'],
                         physicalH=row['PhysicalH'],
                         sexualH=row['SexualH'])

            comments.append(post)
    else:
        for i, row in df.iterrows():
            post = Tweet(post_id=row['post_id'],
                         tweet_content=row['tweet_content'],
                         harrasment=None,
                         indirectH=None,
                         physicalH=None,
                         sexualH=None)
            comments.append(post)
    return comments



