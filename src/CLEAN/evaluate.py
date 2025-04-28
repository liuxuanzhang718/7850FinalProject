import csv
import pickle
from .utils import *
from .distance_map import *
from .evaluate import *
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, \
    roc_auc_score, accuracy_score, f1_score, average_precision_score
from tqdm import tqdm
import numpy as np
from collections import Counter


def maximum_separation(dist_lst, first_grad, use_max_grad):
    opt = 0 if first_grad else -1
    gamma = np.append(dist_lst[1:], np.repeat(dist_lst[-1], 10))
    sep_lst = np.abs(dist_lst - np.mean(gamma))
    sep_grad = np.abs(sep_lst[:-1]-sep_lst[1:])
    if use_max_grad:
        # max separation index determined by largest grad
        max_sep_i = np.argmax(sep_grad)
    else:
        # max separation index determined by first or the last grad
        large_grads = np.where(sep_grad > np.mean(sep_grad))
        max_sep_i = large_grads[-1][opt]
    # if no large grad is found, just call first EC
    if max_sep_i >= 5:
        max_sep_i = 0
    return max_sep_i


def write_max_sep_choices(df, csv_name, first_grad=True, use_max_grad=False, gmm = None):
    out_file = open(csv_name + '_maxsep.csv', 'w', newline='')
    out_file_confidence = open(csv_name + '_maxsep_confidence.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    csvwriter_confidence = csv.writer(out_file_confidence, delimiter=',')
    all_test_EC = set()
    for col in df.columns:
        ec = []
        ec_confidence = []
        smallest_10_dist_df = df[col].nsmallest(10)
        dist_lst = list(smallest_10_dist_df)
        max_sep_i = maximum_separation(dist_lst, first_grad, use_max_grad)
        for i in range(max_sep_i+1):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            if gmm != None:
                gmm_lst = pickle.load(open(gmm, 'rb'))
                mean_confidence_i, std_confidence_i = infer_confidence_gmm(dist_i, gmm_lst)
                confidence_str = "{:.4f}_{:.4f}".format(mean_confidence_i, std_confidence_i)
                ec_confidence.append('EC:' + str(EC_i) + '/' + confidence_str)
            dist_str = "{:.4f}".format(dist_i)
            all_test_EC.add(EC_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        csvwriter.writerow(ec)
        if gmm != None:
            ec_confidence.insert(0, col)
            csvwriter_confidence.writerow(ec_confidence)
    return

def infer_confidence_gmm(distance, gmm_lst):
    confidence = []
    for j in range(len(gmm_lst)):
        main_GMM = gmm_lst[j]
        a, b = main_GMM.means_
        true_model_index = 0 if a[0] < b[0] else 1
        certainty = main_GMM.predict_proba([[distance]])[0][true_model_index]
        confidence.append(certainty)
    return np.mean(confidence), np.std(confidence)

def write_pvalue_choices(df, csv_name, random_nk_dist_map, p_value=1e-5, gmm=None):
    out_file = open(csv_name + '_pvalue.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    out_file_confidence = open(csv_name + '_pvalue_confidence.csv', 'w', newline='')
    csvwriter_confidence = csv.writer(out_file_confidence, delimiter=',')
    all_test_EC = set()
    nk = len(random_nk_dist_map.keys())
    threshold = p_value*nk
    for col in tqdm(df.columns):
        ec = []
        ec_confidence = []
        smallest_10_dist_df = df[col].nsmallest(10)
        for i in range(10):
            EC_i = smallest_10_dist_df.index[i]
            # find all the distances in the random nk w.r.t. EC_i
            # then sorted the nk distances
            rand_nk_dists = [random_nk_dist_map[rand_nk_id][EC_i]
                             for rand_nk_id in random_nk_dist_map.keys()]
            rand_nk_dists = np.sort(rand_nk_dists)
            # rank dist_i among rand_nk_dists
            dist_i = smallest_10_dist_df[i]
            rank = np.searchsorted(rand_nk_dists, dist_i)
            if (rank <= threshold) or (i == 0):
                dist_str = "{:.4f}".format(dist_i)
                all_test_EC.add(EC_i)
                ec.append('EC:' + str(EC_i) + '/' + dist_str)
                if gmm != None:
                    gmm_lst = pickle.load(open(gmm, 'rb'))
                    mean_confidence_i, std_confidence_i = infer_confidence_gmm(dist_i, gmm_lst)
                    confidence_str = "{:.4f}_{:.4f}".format(mean_confidence_i, std_confidence_i)
                    ec_confidence.append('EC:' + str(EC_i) + '/' + confidence_str)
            else:
                break
        ec.insert(0, col)
        csvwriter.writerow(ec)
        if gmm != None:
            ec_confidence.insert(0, col)
            csvwriter_confidence.writerow(ec_confidence)
    return


def write_random_nk_choices_prc(df, csv_name, random_nk_dist_map, p_value=1e-4, 
                                upper_bound=0.0025, steps=24):
    out_file = open(csv_name + '_randnk.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    all_test_EC = set()
    nk = len(random_nk_dist_map.keys())
    threshold = np.linspace(p_value, upper_bound, steps)*nk
    for col in tqdm(df.columns):
        ec = []
        smallest_10_dist_df = df[col].nsmallest(10)
        for i in range(10):
            EC_i = smallest_10_dist_df.index[i]
            # find all the distances in the random nk w.r.t. EC_i
            # then sorted the nk distances
            rand_nk_dists = [random_nk_dist_map[rand_nk_id][EC_i]
                             for rand_nk_id in random_nk_dist_map.keys()]
            rand_nk_dists = np.sort(rand_nk_dists)
            # rank dist_i among rand_nk_dists
            dist_i = smallest_10_dist_df[i]
            rank = np.searchsorted(rand_nk_dists, dist_i)
            if (rank <= threshold[-1]) or (i == 0):
                if i != 0:
                    dist_str = str(np.searchsorted(threshold, rank))
                else:
                    dist_str = str(0)
                all_test_EC.add(EC_i)
                ec.append('EC:' + str(EC_i) + '/' + dist_str)
            else:
                break
        ec.insert(0, col)
        csvwriter.writerow(ec)
    return


def write_top_choices(df, csv_name, top=30):
    out_file = open(csv_name + '_top' + str(top)+'.csv', 'w', newline='')
    csvwriter = csv.writer(out_file, delimiter=',')
    dists = []
    for col in df.columns:
        ec = []
        dist_lst = []
        smallest_10_dist_df = df[col].nsmallest(top)
        for i in range(top):
            EC_i = smallest_10_dist_df.index[i]
            dist_i = smallest_10_dist_df[i]
            dist_str = "{:.4f}".format(dist_i)
            dist_lst.append(dist_i)
            ec.append('EC:' + str(EC_i) + '/' + dist_str)
        ec.insert(0, col)
        dists.append(dist_lst)
        csvwriter.writerow(ec)
    return dists


def random_nk_model(id_ec_train, ec_id_dict_train, emb_train, n=10, weighted=False):
    ids = list(id_ec_train.keys())
    nk = n * 1000
    if weighted:
        P = []
        for id in id_ec_train.keys():
            ecs_id = id_ec_train[id]
            ec_densities = [len(ec_id_dict_train[ec]) for ec in ecs_id]
            # the prob of calling this id is inversely prop to 1/max(density)
            P.append(1/np.max(ec_densities))
        P = P/np.sum(P)
        random_nk_id = np.random.choice(
            range(len(ids)), nk, replace=True, p=P)
    else:
        random_nk_id = np.random.choice(range(len(ids)), nk, replace=False)

    random_nk_id = np.sort(random_nk_id)
    chosen_ids = [ids[i] for i in random_nk_id]
    chosen_emb_train = emb_train[random_nk_id]
    return chosen_ids, chosen_emb_train


def update_dist_dict_blast(emb_test, emb_train, dist, start, end,
                           id_ec_test, id_ec_train):

    id_tests = list(id_ec_test.keys())
    id_trains = list(id_ec_train.keys())
    dist_matrix = torch.cdist(emb_test[start:end], emb_train)
    for i, id_test in tqdm(enumerate(id_tests[start:end])):
        dist[id_test] = {}
        # continue adding EC/dist pairs until have 20 EC
        idx_train_closest_sorted = torch.argsort(dist_matrix[i], dim=-1)
        count = 0
        while len(dist[id_test]) <= 10:
            idx_train_closest = idx_train_closest_sorted[count]
            dist_train_closest = dist_matrix[i][idx_train_closest].cpu().item()
            count += 1
            id_train_closest = id_trains[idx_train_closest]
            ECs_train_closest = id_ec_train[id_train_closest]
            for EC in ECs_train_closest:
                # if EC is not added to the dict
                if EC not in dist[id_test]:
                    # add EC/dist pair
                    dist[id_test][EC] = dist_train_closest
    return dist


def get_true_labels(file_name):
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter='\t')
    all_label = set()
    true_label_dict = {}
    header = True
    count = 0
    for row in csvreader:
        # don't read the header
        if header is False:
            count += 1
            true_ec_lst = row[1].split(';')
            true_label_dict[row[0]] = true_ec_lst
            for ec in true_ec_lst:
                all_label.add(ec)
        if header:
            header = False
    true_label = [true_label_dict[i] for i in true_label_dict.keys()]
    return true_label, all_label


def get_pred_labels(out_filename, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = pred_ec_dist.split(":")[1].split("/")[0]
            preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)
    return pred_label

def get_pred_probs(out_filename, pred_type="_maxsep"):
    file_name = out_filename + pred_type
    # For knn method, parse EC:number/probability
    if pred_type == "_knn":
        pred_probs = []
        with open(file_name + '.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                # row[1] 格式为 'EC:3.5.2.6/0.1234'
                prob = float(row[1].split(":")[1].split("/")[1])
                pred_probs.append(torch.tensor([prob]))
        return pred_probs

    # Other methods keep the original distance-to-probability logic
    result = open(file_name + '.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_probs = []
    for row in csvreader:
        preds_with_dist = row[1:]
        probs = torch.zeros(len(preds_with_dist))
        for i, pred_ec_dist in enumerate(preds_with_dist):
            dist_val = float(pred_ec_dist.split(":")[1].split("/")[1])
            probs[i] = dist_val
        probs = (1 - torch.exp(-1/probs)) / (1 + torch.exp(-1/probs))
        probs = probs / torch.sum(probs)
        pred_probs.append(probs)
    return pred_probs



def get_pred_labels_prc(out_filename, cutoff, pred_type="_maxsep"):
    file_name = out_filename+pred_type
    result = open(file_name+'.csv', 'r')
    csvreader = csv.reader(result, delimiter=',')
    pred_label = []
    for row in csvreader:
        preds_ec_lst = []
        preds_with_dist = row[1:]
        for pred_ec_dist in preds_with_dist:
            # get EC number 3.5.2.6 from EC:3.5.2.6/10.8359
            ec_i = pred_ec_dist.split(":")[1].split("/")[0]
            if int(pred_ec_dist.split(":")[1].split("/")[1]) <= cutoff:
                preds_ec_lst.append(ec_i)
        pred_label.append(preds_ec_lst)
    return pred_label


# def get_eval_metrics(pred_label, true_label, all_label):
#     mlb = MultiLabelBinarizer()
#     mlb.fit([list(all_label)])
#     n_test = len(pred_label)
#     pred_m = np.zeros((n_test, len(mlb.classes_)))
#     true_m = np.zeros((n_test, len(mlb.classes_)))
#     for i in range(n_test):
#         pred_m[i] = mlb.transform([pred_label[i]])
#         true_m[i] = mlb.transform([true_label[i]])
#     pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
#     rec = recall_score(true_m, pred_m, average='weighted')
#     f1 = f1_score(true_m, pred_m, average='weighted')
#     roc = roc_auc_score(true_m, pred_m, average='weighted')
#     acc = accuracy_score(true_m, pred_m)
#     return pre, rec, f1, roc, acc

def get_ec_pos_dict(mlb, true_label, pred_label):
    ec_list = []
    pos_list = []
    for i in range(len(true_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([true_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([true_label[i]]))[1])
    for i in range(len(pred_label)):
        ec_list += list(mlb.inverse_transform(mlb.transform([pred_label[i]]))[0])
        pos_list += list(np.nonzero(mlb.transform([pred_label[i]]))[1])
    label_pos_dict = {}
    for i in range(len(ec_list)):
        ec, pos = ec_list[i], pos_list[i]
        label_pos_dict[ec] = pos
        
    return label_pos_dict

def get_eval_metrics(pred_label, pred_probs, true_label, all_label):
    mlb = MultiLabelBinarizer()
    mlb.fit([list(all_label)])
    n_test = len(pred_label)
    pred_m = np.zeros((n_test, len(mlb.classes_)))
    true_m = np.zeros((n_test, len(mlb.classes_)))
    # for including probability
    pred_m_auc = np.zeros((n_test, len(mlb.classes_)))
    label_pos_dict = get_ec_pos_dict(mlb, true_label, pred_label)
    for i in range(n_test):
        pred_m[i] = mlb.transform([pred_label[i]])
        true_m[i] = mlb.transform([true_label[i]])
         # fill in probabilities for prediction
        labels, probs = pred_label[i], pred_probs[i]
        for label, prob in zip(labels, probs):
            if label in all_label:
                pos = label_pos_dict[label]
                pred_m_auc[i, pos] = prob
    pre = precision_score(true_m, pred_m, average='weighted', zero_division=0)
    rec = recall_score(true_m, pred_m, average='weighted')
    f1 = f1_score(true_m, pred_m, average='weighted')
    try:
        roc = roc_auc_score(true_m, pred_m_auc, average='weighted')
    except:
        roc = 0
    acc = accuracy_score(true_m, pred_m)
    return pre, rec, f1, roc, acc, true_m, pred_m, pred_m_auc

def write_knn_choices(train_ids, id_ec_train, id_ec_test, dist_matrix, csv_name, k=5):
    import math, csv
    alpha = 5 
    with open(csv_name + '_knn.csv','w',newline='') as f:
        writer = csv.writer(f)
        for i, test_id in enumerate(id_ec_test.keys()):
            dists, idxs = torch.topk(dist_matrix[i], k, largest=False)
            weight_sum = {}
            for dist_f, idx in zip(dists.tolist(), idxs.tolist()):
                w = math.exp(-alpha*dist_f)
                for ec in id_ec_train[train_ids[idx]]:
                    weight_sum[ec] = weight_sum.get(ec,0.0) + w
            if weight_sum:
                ec_pred, w_max = max(weight_sum.items(), key=lambda x: x[1])
                prob = w_max / sum(weight_sum.values())
            else:
                ec_pred, prob = 'UNKNOWN', 0.0
            writer.writerow([test_id, f'EC:{ec_pred}/{prob:.4f}'])


def write_neighbor_choices(train_ids, id_ec_train, id_ec_test, dist_matrix,
                           csv_name, k=5, delta=1.0, fallback_level=3, alpha=1.0):
    with open(csv_name + '_neighbor.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, test_id in enumerate(id_ec_test.keys()):
            dists = dist_matrix[i]
            # 1) Only keep neighbors with distance < δ
            within = (dists < delta).nonzero().flatten().tolist()
            fallback = False
            if len(within) >= k:
                idxs = sorted(within, key=lambda idx: dists[idx])[:k]
            else:
                idxs = torch.topk(dists, k, largest=False).indices.tolist()
                fallback = True

            # 2) Weighted voting
            weight_sum = {}
            for idx in idxs:
                w = math.exp(-alpha * dists[idx])
                for ec in id_ec_train[train_ids[idx]]:
                    weight_sum[ec] = weight_sum.get(ec, 0.0) + w

            # 3) Top-1 result
            if weight_sum:
                ec_pred, w_max = max(weight_sum.items(), key=lambda x: x[1])
                prob = w_max / sum(weight_sum.values())
            else:
                ec_pred, prob = 'UNKNOWN', 0.0

            # 4) If there are not enough neighbors, truncate to the fallback_level EC
            if fallback and ec_pred != 'UNKNOWN':
                parts = ec_pred.split('.')
                ec_pred = '.'.join(parts[:fallback_level])

            # 5) Hierarchical correction: if level 1 and level 2 are inconsistent, fallback to the majority at level 2
            if ec_pred != 'UNKNOWN' and '.' in ec_pred:
                level1 = ec_pred.split('.')[0]
                second_levels = ['.'.join(ec.split('.')[:2]) for ec in weight_sum]
                if second_levels:
                    major2 = Counter(second_levels).most_common(1)[0][0]
                    if major2.split('.')[0] != level1:
                        ec_pred = major2
                        prob = 1.0

            writer.writerow([test_id, f'EC:{ec_pred}/{prob:.4f}'])


def write_integrated_choices(test_data, lambda_, csv_name):
    """
    Read kNN and maxsep prediction results, and perform weighted integration according to score = λ * p_kNN + (1-λ) * p_maxsep
    Output only the single Top-1 prediction to csv.
    """
    # 读取原始标签与概率
    knn_labels = get_pred_labels(csv_name, pred_type='_knn')
    knn_probs  = get_pred_probs(csv_name, pred_type='_knn')  # list of tensors [p_k]
    max_labels = get_pred_labels(csv_name, pred_type='_maxsep')
    max_probs  = get_pred_probs(csv_name, pred_type='_maxsep')  # list of tensors [p_m1, p_m2, ...]

    id_ec_test, _ = get_ec_id_dict(f'./data/{test_data}.csv')
    with open(csv_name + '_integrated.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, test_id in enumerate(id_ec_test.keys()):
            # Candidate ECs
            candidates = set(knn_labels[i] + max_labels[i])
            best_ec = None
            best_score = -1.0
            for ec in candidates:
                # kNN probability
                p_k = knn_probs[i].item() if ec == knn_labels[i][0] else 0.0
                # max-sep probability
                if ec in max_labels[i]:
                    idx = max_labels[i].index(ec)
                    p_m = max_probs[i][idx].item()
                else:
                    p_m = 0.0
                # Weighted fusion
                score = lambda_ * p_k + (1 - lambda_) * p_m
                if score > best_score:
                    best_score = score
                    best_ec = ec
            writer.writerow([test_id, f'EC:{best_ec}/{best_score:.4f}'])


def write_triple_integrated_choices(test_data, alpha, beta, gamma, csv_name):
    """
    Integrate predictions from kNN, max-sep, and p-value methods:
      score(EC) = α p_kNN + β p_maxsep + γ p_pvalue, α+β+γ=1
    Only output the Top-1 EC and its integrated score.
    """
    # 读取所有预测标签和概率
    knn_labels = get_pred_labels(csv_name, pred_type='_knn')
    knn_probs  = get_pred_probs(csv_name, pred_type='_knn')
    max_labels = get_pred_labels(csv_name, pred_type='_maxsep')
    max_probs  = get_pred_probs(csv_name, pred_type='_maxsep')
    pval_labels = get_pred_labels(csv_name, pred_type='_pvalue')
    pval_probs  = get_pred_probs(csv_name, pred_type='_pvalue')

    id_ec_test, _ = get_ec_id_dict(f'./data/{test_data}.csv')
    with open(csv_name + '_triple.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, test_id in enumerate(id_ec_test.keys()):
            # Merge candidate ECs
            candidates = set(knn_labels[i] + max_labels[i] + pval_labels[i])
            best_ec = None
            best_score = -1.0
            for ec in candidates:
                # Get probabilities from each method
                p_k = knn_probs[i].item() if ec == knn_labels[i][0] else 0.0
                p_m = 0.0
                if ec in max_labels[i]:
                    idx_m = max_labels[i].index(ec)
                    p_m = max_probs[i][idx_m].item()
                p_p = 0.0
                if ec in pval_labels[i]:
                    idx_p = pval_labels[i].index(ec)
                    p_p = pval_probs[i][idx_p].item()
                # Integrated score
                score = alpha * p_k + beta * p_m + gamma * p_p
                if score > best_score:
                    best_score = score
                    best_ec = ec
            writer.writerow([test_id, f'EC:{best_ec}/{best_score:.4f}'])


def write_softmax_knn_choices(train_ids, id_ec_train, id_ec_test, dist_matrix,
                              csv_name, N=10, T=1.0):
    """
    Softmax weighted kNN:
      1. For the N nearest training samples with distances d_i, compute p_i = exp(-d_i/T) / sum_j exp(-d_j/T)
      2. Accumulate p_i for each training sample's EC, and select the EC with the largest weighted sum as the final prediction
    """
    with open(csv_name + '_softmax_knn.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, test_id in enumerate(id_ec_test.keys()):
            dists = dist_matrix[i] 
            vals, idxs = torch.topk(dists, N, largest=False)  
            exps = torch.exp(-vals / T)                       
            probs = exps / exps.sum()                      

      
            weight_sum = {}
            for prob, idx in zip(probs.tolist(), idxs.tolist()):
                train_id = train_ids[idx]
                for ec in id_ec_train[train_id]:
                    weight_sum[ec] = weight_sum.get(ec, 0.0) + prob

            best_ec, best_p = max(weight_sum.items(), key=lambda x: x[1])
            writer.writerow([test_id, f'EC:{best_ec}/{best_p:.4f}'])
