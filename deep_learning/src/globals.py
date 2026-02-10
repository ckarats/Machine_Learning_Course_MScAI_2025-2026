# global variables
PAD_TOKEN = "*$*PAD*$*"
UNK_TOKEN = "*$*UNK*$*"

SEED = 1985
MODELS_DIR = "models/"
DATA_DIR = "data/"
GLOVE_EMBEDDINGS_PATH = 'data/embeddings/glove.twitter.27B/glove.twitter.27B.200d.txt'
TRAIN_DATA_PATH = 'data/Train_data_compeition.csv'
VALID_DATA_PATH = 'data/Validation_data_competition.csv'
TEST_DATA_PATH = "data/testset-competition.csv"
TEST_RESULTS = "data/testdata_gold_labels.csv"
TRANSLATED_DE_PATH ="data/translations-de.csv"
TRANSLATED_FR_PATH = "data/translations-fr.csv"
TRANSLATED_GR_PATH = "data/translations-el.csv"
SUBMISSION_PATH = 'data/submission.csv'
RESULTS_DIR = 'results/'
IMPLEMENTED_MODELS = ['vanilla_last', 'vanilla_avg', 'vanilla_projected_last',
                      'vanilla_projected_avg', 'attention', 'projected_attention',
                      'multi_attention', 'multi_projected_attention']
CONFIG= {'lr': 0.001,
         'dropout': 0.3,
         'epochs': 20,
         'patience': 10,
         'batch_size': 32,
         'maxlen': 100,
         'trainable_embeddings': False,
         'iterations': 10
         }