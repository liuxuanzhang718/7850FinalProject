import torch
from .utils import * 
from .model import LayerNormNet, LayerNormNet1, LayerNormNet2, LayerNormNet3
from .distance_map import *
from .evaluate import *
import pandas as pd
import warnings
from collections import defaultdict

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def infer_pvalue(train_data, test_data, p_value = 1e-4, nk_random = 50, 
                 report_metrics = False, pretrained=True, model_name=None, gmm=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet1(512, 128, device, dtype)
    
    checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
        
    model.load_state_dict(checkpoint)
    model.eval()
    
    emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    rand_nk_ids, rand_nk_emb_train = random_nk_model(
        id_ec_train, ec_id_dict_train, emb_train, n=nk_random, weighted=True)
    random_nk_dist_map = get_random_nk_dist_map(
        emb_train, rand_nk_emb_train, ec_id_dict_train, rand_nk_ids, device, dtype)
    ensure_dirs()
    out_filename = "results/" +  test_data
    write_pvalue_choices( eval_df, out_filename, random_nk_dist_map, p_value=p_value, gmm=gmm)
    # optionally report prediction precision/recall/...
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_pvalue')
        pred_probs = get_pred_probs(out_filename, pred_type='_pvalue')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc, _, _, _ = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print(f'############ EC calling results using random '
        f'chosen {nk_random}k samples ############')
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} ')
        print('-' * 75)  
    


def infer_maxsep(train_data, test_data, report_metrics = False, 
                 pretrained=True, model_name=None, gmm = None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/' + test_data + '.csv')
    # load checkpoints
    # NOTE: change this to LayerNormNet(512, 256, device, dtype) 
    # and rebuild with [python build.py install]
    # if inferencing on model trained with supconH loss
    model = LayerNormNet1(512, 128, device, dtype)
    
    checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
            
    model.load_state_dict(checkpoint)
    model.eval()

    emb_train = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs()
    out_filename = "results/" +  test_data
    write_max_sep_choices(eval_df, out_filename, gmm=gmm)

    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc, _, _, _ = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.3} | recall: {rec:.3}'
            f'| F1: {f1:.3} | AUC: {roc:.3} | ACC: {acc:.3}')
        print('-' * 75)



def infer_knn(train_data, test_data, k=3, report_metrics=False, pretrained=True, model_name=None, gmm=None):
    """
    Inference based on nearest neighbor majority voting:
    For each test sample, take the k nearest enzyme vectors from the training set,
    count the frequency of their corresponding EC numbers, and the most frequent one is the final prediction.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32

    # 1) Read the mapping between ID and EC for training/testing
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/' + train_data + '.csv')
    id_ec_test,  _                = get_ec_id_dict('./data/' + test_data  + '.csv')

    # 2) Build the model and load weights
    model = LayerNormNet1(512, 128, device, dtype)
    checkpoint = torch.load(f'./data/model/{model_name}.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # -- New: Load ESM embedding for each train_id in order -- 
    train_ids = list(id_ec_train.keys())   # This order is used for knn indexing later
    esm_list  = []
    for pid in train_ids:
        # Assume each protein's ESM embedding is saved as ./data/esm_data/{pid}.pt
        vec = torch.load(f'./data/esm_data/{pid}.pt')   # raw ESM output
        esm_list.append(format_esm(vec))                # or equivalent processing in your utils
    esm_tensor = torch.stack(esm_list, dim=0).to(device).to(dtype)  # [n_train, 512]
    emb_train  = model(esm_tensor)                               # [n_train, 128]

    # 3) Similarly, manually load for the test set
    test_ids  = list(id_ec_test.keys())
    esm_list2 = [ format_esm(torch.load(f'./data/esm_data/{pid}.pt')).to(device).to(dtype)
                  for pid in test_ids ]
    emb_test  = model(torch.stack(esm_list2,dim=0))               # [n_test, 128]

    # 4) Compute the distance matrix between test and train
    #    dist_mat[i,j] = Euclidean distance between emb_test[i] and emb_train[j]
    dist_mat = torch.cdist(emb_test, emb_train)                   # [n_test, n_train]


    # 5) Output result directory
    ensure_dirs()
    out_filename = "results/" + test_data

    # 6) Write kNN voting results
    write_knn_choices(train_ids, id_ec_train, id_ec_test, dist_mat,
                      csv_name=f'results/{test_data}', k=k)

    # 7) If needed, calculate and print evaluation metrics
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_knn')
        pred_probs = get_pred_probs(out_filename, pred_type='_knn')
        true_label, all_label = get_true_labels('./data/' + test_data)
        pre, rec, f1, roc, acc, _, _, _ = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print(f"############ EC calling results using kNN vote (k={k}) ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
              f'>>> precision: {pre:.3} | recall: {rec:.3}'
              f' | F1: {f1:.3} | AUC: {roc:.3} | ACC: {acc:.3}')
        print('-' * 75)


def infer_filtered_hierarchical_kNN(train_data, test_data, k=3, delta=2,
                            fallback_level=3, alpha=1.0, report_metrics=False,
                            model_name=None):
    # 1) Read mapping
    id_ec_train, _ = get_ec_id_dict(f'./data/{train_data}.csv')
    id_ec_test,  _ = get_ec_id_dict(f'./data/{test_data}.csv')

    # 2) Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    model  = LayerNormNet1(512, 128, device, dtype)
    ckpt   = torch.load(f'./data/model/{model_name}.pth', map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 3) Load ESM embeddings by ID order and generate vectors
    train_ids = list(id_ec_train.keys())
    esm_train = [format_esm(torch.load(f'./data/esm_data/{pid}.pt')) for pid in train_ids]
    emb_train = model(torch.stack(esm_train, dim=0).to(device).to(dtype))

    test_ids  = list(id_ec_test.keys())
    esm_test  = [format_esm(torch.load(f'./data/esm_data/{pid}.pt')) for pid in test_ids]
    emb_test  = model(torch.stack(esm_test, dim=0).to(device).to(dtype))

    # 4) Compute distance matrix
    dist_mat = torch.cdist(emb_test, emb_train)

    # 5) Write predictions
    ensure_dirs()
    out_name = f'results/{test_data}'
    write_neighbor_choices(train_ids, id_ec_train, id_ec_test, dist_mat,
                           out_name, k=k, delta=delta,
                           fallback_level=fallback_level, alpha=alpha)

    # 6) Optional evaluation
    if report_metrics:
        preds = get_pred_labels(out_name, pred_type='_neighbor')
        probs = get_pred_probs(out_name, pred_type='_neighbor')
        true, all_ec = get_true_labels(f'./data/{test_data}')
        pre, rec, f1, auc, acc, *_ = get_eval_metrics(preds, probs, true, all_ec)
        print(f"Neighbor strategy k={k}, δ={delta}")
        print(f"Precision: {pre:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}, Acc: {acc:.3f}")


def infer_integrated(train_data, test_data, lambda_=0.5, 
                              k=5, report_metrics=False, model_name=None):
    """
    1) First run kNN and max-sep inference to generate corresponding CSVs
    2) Call write_integrated_choices for weighted fusion
    3) (Optional) Evaluate the fused prediction metrics
    """
    # 1) Run both algorithms (disable their own evaluation)
    infer_knn(train_data, test_data, k=k, report_metrics=False, model_name=model_name)
    infer_maxsep(train_data, test_data, report_metrics=False, model_name=model_name)

    # 2) Integration
    csv_name = f'results/{test_data}'
    write_integrated_choices(test_data, lambda_, csv_name)

    # 3) Optional evaluation
    if report_metrics:
        preds = get_pred_labels(csv_name, pred_type='_integrated')
        probs = get_pred_probs(csv_name, pred_type='_integrated')  # Use knn parsing method
        true, all_ec = get_true_labels(f'./data/{test_data}')
        pre, rec, f1, auc, acc, *_ = get_eval_metrics(preds, probs, true, all_ec)
        print(f"Integrated λ={lambda_}, k={k}")
        print(f"Precision: {pre:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}, Acc: {acc:.3f}")



def infer_triple_integration(train_data, test_data, alpha=0.33, beta=0.33, gamma=0.34,
                             k=5, p_value=1e-4, report_metrics=False, model_name=None):
    """
    1) Run kNN, max-sep, and p-value inference (do not report metrics)
    2) Call write_triple_integrated_choices to fuse all three
    3) Optionally evaluate the fused metrics
    """
    # First generate each result file
    infer_knn(train_data, test_data, k=k, report_metrics=False, model_name=model_name)
    infer_maxsep(train_data, test_data, report_metrics=False, model_name=model_name)
    infer_pvalue(train_data, test_data, p_value=p_value, 
                 report_metrics=False, model_name=model_name)

    # Fusion
    csv_name = f'results/{test_data}'
    write_triple_integrated_choices(test_data, alpha, beta, gamma, csv_name)

    # Evaluation
    if report_metrics:
        preds = get_pred_labels(csv_name, pred_type='_triple')
        probs = get_pred_probs(csv_name, pred_type='_triple')  # Parse as single probability
        true, all_ec = get_true_labels(f'./data/{test_data}')
        pre, rec, f1, auc, acc, *_ = get_eval_metrics(preds, probs, true, all_ec)
        print(f"Triple Integration α={alpha:.2f}, β={beta:.2f}, γ={gamma:.2f}")
        print(f"Precision: {pre:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}, Acc: {acc:.3f}")


def infer_softmax(train_data, test_data, N=10, T=1.0,
                      report_metrics=False, model_name=None):
    # 1) Read mapping
    id_ec_train, _ = get_ec_id_dict(f'./data/{train_data}.csv')
    id_ec_test,  _ = get_ec_id_dict(f'./data/{test_data}.csv')

    # 2) Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    model  = LayerNormNet1(512, 128, device, dtype)
    checkpoint = torch.load(f'./data/model/{model_name}.pth', map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3) Generate train/test vectors
    train_ids = list(id_ec_train.keys())
    esm_train = torch.stack([
        format_esm(torch.load(f'./data/esm_data/{pid}.pt')) for pid in train_ids
    ]).to(device).to(dtype)
    emb_train = model(esm_train)  # [n_train, dim]

    test_ids  = list(id_ec_test.keys())
    esm_test = torch.stack([
        format_esm(torch.load(f'./data/esm_data/{pid}.pt')) for pid in test_ids
    ]).to(device).to(dtype)
    emb_test = model(esm_test)    # [n_test, dim]

    # 4) Compute distance matrix
    dist_matrix = torch.cdist(emb_test, emb_train)  # [n_test, n_train]

    # 5) Write predictions
    ensure_dirs()
    out_name = f'results/{test_data}'
    write_softmax_knn_choices(train_ids, id_ec_train, id_ec_test,
                              dist_matrix, out_name, N=N, T=T)

    # 6) Optional evaluation
    if report_metrics:
        preds = get_pred_labels(out_name, pred_type='_softmax_knn')
        probs = get_pred_probs(out_name, pred_type='_softmax_knn')
        true, all_ec = get_true_labels(f'./data/{test_data}')
        pre, rec, f1, auc, acc, *_ = get_eval_metrics(preds, probs, true, all_ec)
        print(f"Softmax-kNN N={N}, T={T} → Precision: {pre:.3f}, Recall: {rec:.3f}, "
              f"F1: {f1:.3f}, AUC: {auc:.3f}, Acc: {acc:.3f}")