from pathlib import Path
from src.CLEAN.infer import *

def main(args):

    print(args)

    if 'maxsep' in args.method:
        infer_maxsep(args.train_data, args.test_data, report_metrics=True, pretrained=False, model_name=args.model_name, gmm=args.gmm)
    elif 'pvalue' in args.method:
        infer_pvalue(args.train_data, args.test_data, report_metrics=True, pretrained=False, model_name=args.model_name, gmm=args.gmm)
    elif 'knn' in args.method:
        infer_knn(args.train_data, args.test_data, report_metrics=True, pretrained=False, model_name=args.model_name)
    elif 'filtered_hierarchical_kNN' in args.method:
        infer_filtered_hierarchical_kNN(args.train_data, args.test_data, report_metrics=True, model_name=args.model_name)
    elif 'integrated' in args.method:
        infer_integrated(args.train_data, args.test_data, lambda_= 0.75, k=5,report_metrics=True, model_name=args.model_name)
    elif 'integrated_triple' in args.method:
        infer_triple_integration(
            args.train_data, args.test_data,
            alpha= 0.34, beta=0.33, gamma=0.33,
            k=5, p_value=1e-4,
            report_metrics=True,
            model_name=args.model_name
        )
    elif 'softmax' in args.method:                                                    
        infer_softmax(args.train_data, args.test_data, T=0.1, report_metrics=True, model_name=args.model_name)
    else:
        raise ValueError(f'Invalid method: {args.method}')

def parse_args():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default='selected_10k', help='Training data name')
    parser.add_argument('--test-data', type=str, default='new', help='Test data name')
    parser.add_argument('--model-name', type=str, default='selected10k_addition_best', help='Trained model file name')
    parser.add_argument('--gmm', type=Path, default=None, help='File name for list of GMM models')
    parser.add_argument('--method', nargs='+', default='maxsep', help='Inference method')

    return parser.parse_args()

if __name__ == "__main__":
    print("--- Script execution starting ---")
    args = parse_args()
    main(args)
    print("--- Script execution finished ---")