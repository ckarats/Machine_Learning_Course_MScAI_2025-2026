from sklearn.metrics import roc_auc_score
import torch
from torch.autograd import Variable
from time import time
import random
import numpy as np
import os
from preprocess import load_data

from globals import SUBMISSION_PATH, TEST_DATA_PATH, MODELS_DIR
SEED = 1985


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, train_batches, test_batches, optimizer,  criterion,
                          epochs, init_patience, cuda=True):
    patience = init_patience
    best_auc = 0.0
    for i in range(1, epochs + 1):
        start = time()
        auc = run_epoch(model, train_batches, test_batches, optimizer,  criterion,
                                         cuda)
        end = time()
        print('epoch %d, auc: %2.3f  Time: %d minutes, %d seconds'
              % (i, 100 * auc, (end - start) / 60, (end - start) % 60))
        if best_auc < auc:
            best_auc = auc
            patience = init_patience
            save_model(model)
            if i > 1:
                print('best epoch so far')
        else:
            patience -= 1
        if patience == 0:
            break
    return best_auc


def run_epoch(model, train_batches, test_batches, optimizer, criterion, cuda):
    model.train(True)
    perm = np.random.permutation(len(train_batches))
    for i in perm:
        batch = train_batches[i]
        inner_perm = np.random.permutation(len(batch['text']))
        data = []
        for inp in model.input_list:
            if cuda:
                data.append(Variable(torch.from_numpy(batch[inp][inner_perm]).long().cuda()))
            else:
                data.append(Variable(torch.from_numpy(batch[inp][inner_perm]).long()))
        if cuda:
            aux = Variable(torch.from_numpy(batch['aux'][inner_perm]).cuda())
        else:
            aux = Variable(torch.from_numpy(batch['aux'][inner_perm]))
        outputs = model(*data)
        # loss = 0.20*criterion(outputs[:, 0], aux[:, 0])+0.2*criterion(outputs[:, 1], aux[:, 1])+0.4 * criterion(outputs[:,2], aux[:,2]) + 0.4 * criterion(outputs[:,3], aux[:,3])
        loss = 0.5*criterion(outputs[:, 0], aux[:, 0])+0.5*(0.2*criterion(outputs[:, 1], aux[:, 1])+0.4 * criterion(outputs[:,2], aux[:,2]) + 0.4 * criterion(outputs[:,3], aux[:,3]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return evaluate(model, test_batches, cuda)


def get_scores_and_labels(model, test_batches, cuda):
    har_scores_list, sex_scores_list, phy_scores_list, ind_scores_list = [], [], [], []
    har_list, sex_list, phy_list, ind_list = [], [], [], []

    for batch in test_batches:
        data = []
        for inp in model.input_list:
            if cuda:
                data.append(Variable(torch.from_numpy(batch[inp]).long().cuda()))
            else:
                data.append(Variable(torch.from_numpy(batch[inp]).long()))
        outputs = model(*data)
        haras_scores, sex_scores, phy_scores, ind_scores =outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        haras_scores = torch.sigmoid(haras_scores)
        sex_scores = torch.sigmoid(sex_scores)
        phy_scores = torch.sigmoid(phy_scores)
        ind_scores = torch.sigmoid(ind_scores)
        har_list.extend(batch['target'].tolist())
        ind_list.extend(batch['indirect'])
        sex_list.extend(batch['sexual'])
        phy_list.extend(batch['physical'])

        har_scores_list.extend(haras_scores.data.view(-1).tolist())
        ind_scores_list.extend(ind_scores.data.view(-1).tolist())
        sex_scores_list.extend(sex_scores.data.view(-1).tolist())
        phy_scores_list.extend(phy_scores.data.view(-1).tolist())
    return {'scores': {'harassment': har_scores_list, 'sexual': sex_scores_list,
                       'physical': phy_scores_list, 'indirect': ind_scores_list},
            'labels': {'harassment': har_list, 'sexual': sex_list,
                       'physical': phy_list, 'indirect': ind_list}}


def evaluate(model, test_batches, cuda):
    model.train(False)
    results = get_scores_and_labels(model, test_batches, cuda)
    auc_scores = []
    for k in results['scores']:
        auc = roc_auc_score(np.asarray(results['labels'][k],  dtype='float32'),
                            np.asarray(results['scores'][k], dtype='float32'))
        print("{} - auc:{}".format(k, auc))
        auc_scores.append(auc)
    return np.mean(auc_scores)


def get_scores_and_ids(model, test_batches, cuda):
    scores_list,  sexual_list, physical_list, indirect_list = [], [], [], []
    ids = []
    for batch in test_batches:
        data = []
        for inp in model.input_list:
            if cuda:
                data.append(Variable(torch.from_numpy(batch[inp]).long().cuda()))
            else:
                data.append(Variable(torch.from_numpy(batch[inp]).long()))
        out = model(*data)
        # outputs = model(*data)
        harassment,  sexual, physical, indirect = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        # outputs = torch.sigmoid(outputs)
        scores_list.extend(torch.sigmoid(harassment).data.view(-1).tolist())
        indirect_list.extend(torch.sigmoid(indirect).data.view(-1).tolist())
        sexual_list.extend(torch.sigmoid(sexual).data.view(-1).tolist())
        physical_list.extend(torch.sigmoid(physical).data.view(-1).tolist())
        ids.extend(batch['post_id'])
    return {'id': ids, 'harassment': scores_list, 'indirect': indirect_list,
            'sexual': sexual_list, 'physical': physical_list}



def predict(scores, thr):
    return [x > thr for x in scores]


def save_model(model):
    torch.save(model.state_dict(), MODELS_DIR + model.name + '.pkl')


def load_model(model):
    model.load_state_dict(torch.load(MODELS_DIR + model.name + '.pkl'))
    return model


def generate_results(model, test_batches, cuda):
    sc = get_scores_and_ids(model, test_batches, cuda)
    d_harassment = {i: s for i, s in zip(sc['id'], sc['harassment'])}
    d_sexual = {i: s for i, s in zip(sc['id'], sc['sexual'])}
    d_indirect = {i: s for i, s in zip(sc['id'], sc['indirect'])}
    d_physical = {i: s for i, s in zip(sc['id'], sc['physical'])}

    return {'harassment': d_harassment, 'sexual': d_sexual, 'indirect': d_indirect, 'physical': d_physical}


def generate_test_submission_values(harassment_dict, indirect_dict, physical_dict, sexual_dict, thr=0.33):
    def final_score(idx, dict_of_values):
        return dict_of_values[idx]

    df = load_data(TEST_DATA_PATH)

    df['Harassment'] = df['post_id'].map(lambda x: final_score(x, dict_of_values=harassment_dict))
    df['IndirectH'] = df['post_id'].map(lambda x: final_score(x, dict_of_values=indirect_dict))
    df['PhysicalH'] = df['post_id'].map(lambda x: final_score(x, dict_of_values=physical_dict))
    df['SexualH'] = df['post_id'].map(lambda x: final_score(x, dict_of_values=sexual_dict))
    final_scores = {}
    for i, d in df.iterrows():
        final_scores[d['post_id']] = {}
        if d['Harassment'] < thr:
            final_scores[d['post_id']]['Harassment'] = 0
            final_scores[d['post_id']]['IndirectH'] = 0
            final_scores[d['post_id']]['PhysicalH'] = 0
            final_scores[d['post_id']]['SexualH'] = 0
        else:
            if d['IndirectH'] > d['PhysicalH']:
                if d['IndirectH'] > d['SexualH']:
                    final_scores[d['post_id']]['Harassment'] = 1
                    final_scores[d['post_id']]['IndirectH'] = 1
                    final_scores[d['post_id']]['PhysicalH'] = 0
                    final_scores[d['post_id']]['SexualH'] = 0
                else:
                    final_scores[d['post_id']]['Harassment'] = 1
                    final_scores[d['post_id']]['IndirectH'] = 0
                    final_scores[d['post_id']]['PhysicalH'] = 0
                    final_scores[d['post_id']]['SexualH'] = 1
            else:
                if d['PhysicalH'] > d['SexualH']:
                    final_scores[d['post_id']]['Harassment'] = 1
                    final_scores[d['post_id']]['IndirectH'] = 0
                    final_scores[d['post_id']]['PhysicalH'] = 1
                    final_scores[d['post_id']]['SexualH'] = 0
                else:
                    final_scores[d['post_id']]['Harassment'] = 1
                    final_scores[d['post_id']]['IndirectH'] = 0
                    final_scores[d['post_id']]['PhysicalH'] = 0
                    final_scores[d['post_id']]['SexualH'] = 1

    df['Harassment'] = df['post_id'].map(lambda x: final_scores[x]["Harassment"])
    df['IndirectH'] = df['post_id'].map(lambda x: final_scores[x]["IndirectH"])
    df['PhysicalH'] = df['post_id'].map(lambda x: final_scores[x]["PhysicalH"])
    df['SexualH'] = df['post_id'].map(lambda x: final_scores[x]["SexualH"])

    df = df.drop("post_id", axis=1)
    df.to_csv(SUBMISSION_PATH, index=False)
    return df




