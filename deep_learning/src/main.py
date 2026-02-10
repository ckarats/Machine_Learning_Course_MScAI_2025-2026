from globals import GLOVE_EMBEDDINGS_PATH, SEED, MODELS_DIR, TEST_RESULTS,  CONFIG, IMPLEMENTED_MODELS, RESULTS_DIR
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from utils import train, load_model, generate_test_submission_values, generate_results, set_seeds
import pandas as pd
import os
from sklearn.metrics import f1_score, precision_score, recall_score
from preprocess import generate_data
from modeling.multi_attention_rnn import MultiAttentionRNN
from modeling.projected_multi_attention_rnn import ProjectedMultiAttentionRNN
from modeling.projected_vanilla_rnn import ProjectedVanillaRNN
from modeling.vanilla_rnn import VanillaRnn
from modeling.attention_rnn import AttentionRNN
from modeling.projected_attention import ProjectedAttentionRNN


def run_model(model_name, data_dict, cuda):
    print("running ", model_name)
    if model_name not in IMPLEMENTED_MODELS:
        NotImplementedError("You must choose one of these:{}".format(IMPLEMENTED_MODELS))
    else:

        emb_matrix = data_dict['emb_matrix']
        train_batches = data_dict['train_batches']
        val_batches = data_dict['val_batches']
        test_batches = data_dict['test_batches']
        set_seeds(SEED)

        harassment_f1_scores = []
        harassment_recall_scores = []
        harassment_precision_scores = []

        indirect_f1_scores = []
        indirect_recall_scores = []
        indirect_precision_scores = []

        sexual_f1_scores = []
        sexual_recall_scores = []
        sexual_precision_scores = []

        physical_f1_scores = []
        physical_recall_scores = []
        physical_precision_scores = []

        runs = CONFIG['iterations']
        for i in range(1, runs+1):
            print("***** iteration: ", i)
            if model_name == "vanilla_projected_last":
                model = ProjectedVanillaRNN(emb_matrix, embeddings_dropout=CONFIG['dropout'])
            elif model_name == "vanilla_projected_avg":
                model = ProjectedVanillaRNN(emb_matrix, avg_pooling=True, embeddings_dropout=CONFIG['dropout'])
            elif model_name == "multi_attention":
                model = MultiAttentionRNN(emb_matrix, embeddings_dropout=CONFIG['dropout'],
                                          trainable_embeddings=CONFIG['trainable_embeddings'])
            elif model_name == "multi_projected_attention":
                model = ProjectedMultiAttentionRNN(emb_matrix, embeddings_dropout=CONFIG['dropout'])
            elif model_name == "attention":
                model = AttentionRNN(emb_matrix, embeddings_dropout=CONFIG['dropout'], trainable_embeddings=CONFIG['trainable_embeddings'])
            elif model_name == "projected_attention":
                model = ProjectedAttentionRNN(emb_matrix, embeddings_dropout=CONFIG['dropout'])
            elif model_name == "vanilla_avg":
                model = VanillaRnn(emb_matrix, avg_pooling=True,
                                   embeddings_dropout=CONFIG['dropout'],
                                   trainable_embeddings=CONFIG['trainable_embeddings'])

            else:
                model = VanillaRnn(emb_matrix, embeddings_dropout=CONFIG['dropout'],
                                   trainable_embeddings=CONFIG['trainable_embeddings'])
            optimizer = Adam(model.params, CONFIG['lr'])
            criterion = BCEWithLogitsLoss()
            train(model=model, train_batches=train_batches, test_batches=val_batches,
                  optimizer=optimizer, criterion=criterion, epochs=CONFIG['epochs'], init_patience=CONFIG['patience'], cuda=cuda)
            model = load_model(model)
            d = generate_results(model, test_batches, cuda)
            df = generate_test_submission_values(harassment_dict=d['harassment'],
                                                 sexual_dict=d['sexual'], physical_dict=d['physical'],
                                                 indirect_dict=d['indirect'])
            df_results = pd.read_csv(TEST_RESULTS)
            harassment_f1_scores.append(f1_score(df_results.harassment.values, df.Harassment.values))
            harassment_precision_scores.append(precision_score(df_results.harassment.values, df.Harassment.values))
            harassment_recall_scores.append(recall_score(df_results.harassment.values, df.Harassment.values))

            indirect_f1_scores.append(f1_score(df_results.IndirectH.values, df.IndirectH.values))
            indirect_precision_scores.append(precision_score(df_results.IndirectH.values, df.IndirectH.values))
            indirect_recall_scores.append(recall_score(df_results.IndirectH.values, df.IndirectH.values))

            sexual_f1_scores.append(f1_score(df_results.SexualH.values, df.SexualH.values))
            sexual_precision_scores.append(precision_score(df_results.SexualH.values, df.SexualH.values))
            sexual_recall_scores.append(recall_score(df_results.SexualH.values, df.SexualH.values))

            physical_f1_scores.append(f1_score(df_results.PhysicalH.values, df.PhysicalH.values))
            physical_precision_scores.append(precision_score(df_results.PhysicalH.values, df.PhysicalH.values))
            physical_recall_scores.append(recall_score(df_results.PhysicalH.values, df.PhysicalH.values))

        results_dict = {
                        'model': [model_name for _ in range(runs)],
                        'harassment_f1_score': harassment_f1_scores,
                        'harassment_recall': harassment_recall_scores,
                        'harassment_precision': harassment_precision_scores,
                        'indirect_f1_score': indirect_f1_scores,
                        'indirect_recall': indirect_recall_scores,
                        'indirect_precision': indirect_precision_scores,
                        'sexual_f1_score': sexual_f1_scores,
                        'sexual_recall': sexual_recall_scores,
                        'sexual_precision': sexual_precision_scores,
                        'physical_f1_score': physical_f1_scores,
                        'physical_recall': physical_recall_scores,
                        'physical_precision': physical_precision_scores
                        }
        df = pd.DataFrame.from_dict(results_dict)
        if "results.csv" in os.listdir(RESULTS_DIR):
            df_old = pd.read_csv(RESULTS_DIR+"results.csv")
            df = pd.concat([df_old, df])
        df.to_csv(RESULTS_DIR+"results.csv", index=False)

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    data = generate_data(embs_path=GLOVE_EMBEDDINGS_PATH, maxlen=CONFIG['maxlen'], batch_size=CONFIG['batch_size'])

    run_model("vanilla_last", data, False)
    run_model("vanilla_projected_last", data, False)
    run_model("vanilla_avg", data, False)
    run_model("vanilla_projected_avg", data, False)
    run_model("multi_attention", data, False)
    run_model("multi_projected_attention", data, False)
    run_model("projected_attention", data, False)
    run_model("attention", data, False)


