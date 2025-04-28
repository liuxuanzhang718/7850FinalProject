from src.CLEAN.evaluate import get_pred_labels, get_pred_probs, get_true_labels, get_eval_metrics
import os
import numpy as np

# true_label, all_label = get_true_labels('data/merge_new_price')

for conf in np.linspace(0.1, 1, 10):
    conf_str = f"{conf:.1f}"
    data_file = f"data/selected_2k_testset_pvalue_confidence_0_{conf_str}.csv"
    results_prefix = f"results/selected_2k_testset_pvalue_confidence_0_{conf_str}"
    print(conf_str)
    if not os.path.exists(data_file):
        print(f"[WARN] no data file {data_file}, skipping threshold {conf_str}")
        continue

    true_label, all_label = get_true_labels(data_file[:-4])  # 去掉 .csv
    pred_label = get_pred_labels(results_prefix, '')
    pred_probs = get_pred_probs(results_prefix, '')

    if len(true_label) != len(pred_label):
        print(f"[WARN] length mismatch at threshold {conf_str}: "
              f"true={len(true_label)} vs pred={len(pred_label)}, skipping")
        continue
    
    pre, rec, f1, roc, acc, *_ = get_eval_metrics(pred_label, pred_probs, true_label, all_label)
    print('#' * 80)
    print(f'>>> confidence: {conf_str}')
    print(f'>>> precision: {pre:.3} | recall: {rec:.3} | F1: {f1:.3} | AUC: {float(roc):.3}')
    print('-' * 80)

print()
print('*' * 100)
print('*' * 100)
print()

for conf in ['0_0.5', '0.5_1.0']:
    data_file = f"data/selected_2k_testset_pvalue_lv34_confidence_{conf}.csv"
    results_prefix = f"results/selected_2k_testset_pvalue_lv34_confidence_{conf}"
    if not os.path.exists(data_file):
        print(f"[WARN] no data file {data_file}, skipping lvl3/4 bin {conf}")
        continue

    true_label, all_label = get_true_labels(data_file[:-4])
    pred_label = get_pred_labels(results_prefix, '')
    pred_probs = get_pred_probs(results_prefix, '')

    if len(true_label) != len(pred_label):
        print(f"[WARN] length mismatch at lvl3/4 bin {conf}: "
              f"true={len(true_label)} vs pred={len(pred_label)}, skipping")
        continue
    pre, rec, f1, roc, acc, *_ = get_eval_metrics(pred_label, pred_probs, true_label, all_label)
    print('#' * 80)
    print(f'>>> confidence: {conf}')
    print(f'>>> precision: {pre:.3} | recall: {rec:.3} | F1: {f1:.3} | AUC: {float(roc):.3} | ACC: {acc:.3}')
    print('-' * 80)