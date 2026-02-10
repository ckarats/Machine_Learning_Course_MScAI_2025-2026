import numpy as np
import gc

from preprocess_utils import get_indexed_text, pad_text, load_data, get_comments, \
    get_embeddings, get_embeddings_matrix, \
    get_unique_tokens, create_final_dictionary
from globals import TRAIN_DATA_PATH, VALID_DATA_PATH,  TEST_DATA_PATH, PAD_TOKEN, UNK_TOKEN, TRANSLATED_DE_PATH, TRANSLATED_GR_PATH, TRANSLATED_FR_PATH


def create_batches(comments, w2i, pad_tnk=PAD_TOKEN, unk_tkn=UNK_TOKEN, batch_size=128,
                   max_len=100, sort_data=True):
    if sort_data:
        comments.sort(key=lambda x: -len(x.tokens))
    offset = 0
    batches = []
    while offset < len(comments):
        batch_texts = []
        batch_targets = []
        batch_ids = []
        batch_aux = []
        batch_indirect = []
        batch_sexual = []
        batch_physical = []
        batch_full = []
        start = offset
        end = min(offset + batch_size, len(comments))
        batch_max_size = comments[start].text_size if sort_data else max(list(map(lambda x: x.text_size, comments[start:end])))
        for i in range(start, end):
            batch_texts.append(get_indexed_text(w2i, pad_text(comments[i].tokens, max(min(max_len, batch_max_size), 1), pad_tkn=pad_tnk), unk_token=unk_tkn))
            batch_targets.append(comments[i].target)
            batch_indirect.append(comments[i].indirectH)
            batch_physical.append(comments[i].physicalH)
            batch_sexual.append(comments[i].sexualH)
            batch_aux.append([comments[i].target, comments[i].sexualH,
                              comments[i].physicalH, comments[i].indirectH])
            batch_ids.append(comments[i].post_id)
            batch_full.append(comments[i].softmax_label)
        batches.append({'text': np.array(batch_texts), 'aux': np.array(batch_aux, dtype='float32'),
                        'target': np.array(batch_targets, dtype='float32'),
                        'indirect': np.reshape(np.array(batch_indirect, dtype='float32'), newshape=[len(batch_indirect), 1]),
                        'physical': np.reshape(np.array(batch_physical, dtype='float32'), newshape=[len(batch_physical), 1]),
                        'sexual': np.reshape(np.array(batch_sexual, dtype='float32'), newshape=[len(batch_sexual), 1]),
                        'post_id': batch_ids,
                        'softmax_label':np.reshape(np.array(batch_full, dtype='float32'), newshape=[len(batch_full), 1])})
        offset += batch_size
    return batches


def get_label_dict(posts):
    return {x.post_id: x.target for x in posts}


def generate_data(embs_path,
                  batch_size=128,
                  min_freq=1,
                  maxlen=1000):
    df = load_data(TRAIN_DATA_PATH)
    df_val = load_data(VALID_DATA_PATH)
    df_test = load_data(TEST_DATA_PATH)
    train_posts = get_comments(df)

    translated_posts_de = get_comments(load_data(TRANSLATED_DE_PATH))
    train_posts += translated_posts_de
    translated_posts_gr = get_comments(load_data(TRANSLATED_GR_PATH))
    train_posts += translated_posts_gr
    translated_posts_fr = get_comments(load_data(TRANSLATED_FR_PATH))
    train_posts += translated_posts_fr

    val_posts = get_comments(df_val)
    test_posts = get_comments(df_test, False)
    print('posts for training:', len(train_posts))
    print('posts for validation:', len(test_posts))
    val_label_dict = get_label_dict(val_posts)
    posts = train_posts + val_posts + test_posts
    tokens = get_unique_tokens(posts, min_freq)
    embeddings_dict = get_embeddings(tokens, embs_path)

    w2i = create_final_dictionary(tokens, embeddings_dict, unk_token=UNK_TOKEN, pad_token=PAD_TOKEN)
    emb_matrix = get_embeddings_matrix(w2i, embeddings_dict, 200)
    del embeddings_dict
    gc.collect()
    train_batches = create_batches(train_posts, w2i, max_len=maxlen, batch_size=batch_size)

    val_batches = create_batches(val_posts, w2i, max_len=maxlen, batch_size=batch_size)
    test_batches = create_batches(test_posts, w2i, max_len=maxlen, batch_size=batch_size)

    return {'emb_matrix': emb_matrix,
            'train_batches': train_batches,
            'test_batches': test_batches,
            'val_batches': val_batches,
            'val_labels:': val_label_dict}

