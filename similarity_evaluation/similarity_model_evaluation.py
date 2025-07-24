import pandas as pd
import glob
import os
from sklearn.metrics import roc_curve
from similarity_evaluation.similarity_models import SpacySimilarityModel, transformer_models, SentenceTransformerSimilarityModel, CrossEncoderSimilarityModel, encoder_models, AngLEModel
from scipy.stats import pearsonr    


def similarity_model_evaluation(output_dir, ebmsass_path, verbose=True, testing=False, retrain=False):
    """
    Evaluate various similarity models on the EBMSASS and BIOSSES datasets.
    Args:
        output_dir (str): Directory to save the results.
        ebmsass_path (str): Path to the EBMSASS dataset directory.
        verbose (bool): If True, print detailed information during evaluation.
        testing (bool): If True, use a subset of the data for testing purposes.
    """

    # Load the data
    ebmsass_df = load_ebmsass_data(ebmsass_path)
    # https://huggingface.co/datasets/tabilab/biosses
    biossess_df = pd.read_parquet("hf://datasets/tabilab/biosses/data/train-00000-of-00001.parquet")

    overall_df = pd.concat([ebmsass_df, biossess_df], ignore_index=True)
    if testing:
        print("TESTING MODE: using a subset of the data")
        overall_df = overall_df.sample(frac=0.01).reset_index(drop=True)


    if retrain:
        # split data into train, validation, and test sets
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(overall_df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)


    models = [
        SpacySimilarityModel(),
        AngLEModel()
    ]

    sentence_transformer_models = [
        SentenceTransformerSimilarityModel(model_name=model_name)
        for model_name in transformer_models
    ]
    cross_encoder_models = [
        CrossEncoderSimilarityModel(model_name=model_name)
        for model_name in encoder_models
    ]

    models = models + sentence_transformer_models + cross_encoder_models

    
    results = []
    for model in models:
        model_name = model.name
        if verbose:
            print(f"Evaluating {model_name}...")
        
        if retrain:
            # Train the model on the training set
            if hasattr(model, 'train'):
                model.train(train_df, val_df)
            else:
                print(f"Model {model_name} does not support training. Skipping...")

        scores = []
        if retrain:
            # Use the test set for evaluation
            scores = test_df.apply(lambda row: model.compute_similarity(row['sentence1'], row['sentence2']), axis=1)
        else:
            scores = overall_df.apply(lambda row: model.compute_similarity(row['sentence1'], row['sentence2']), axis=1)
        scores = scores.to_numpy()
        
        overall_df[model_name] = scores
        
        # Calculate Pearson correlation
        corr, _ = pearsonr(overall_df[model_name], overall_df['score'])

        # compute auc for detecting a simiarity of at least 3
        from sklearn.metrics import roc_auc_score
        y_true = (overall_df['score'] >= 3).astype(int)
        y_scores = overall_df[model_name]
        auc = roc_auc_score(y_true, y_scores)
        j_index = cutoff_youdens_j(*roc_curve(y_true, y_scores)[:3])

        results.append({'model': model_name, 'pearson_correlation': corr, 'auc': auc, 'youden_index': j_index})

    results_df = pd.DataFrame(results)
    results_df['retrained'] = retrain
    results_df.to_csv(os.path.join(output_dir, 'similarity_model_evaluation_results.csv'), index=False)

    # save the overall_df with the scores
    overall_df.to_csv(os.path.join(output_dir, 'dataset_with_similarity_scores.csv'), index=False)


def cutoff_youdens_j(fpr,tpr,thresholds):
    j_scores = tpr-fpr
    j_ordered = sorted(zip(j_scores,thresholds))
    return j_ordered[-1][1]

# read all txt files (tsv) in ebmsass_path and load as a DataFrame
def load_ebmsass_data(path):   
    files = glob.glob(os.path.join(path, '*.txt'))
    dataframes = []
    
    for file in files:
        df = pd.read_csv(file, sep='\t', header=None, names=['id', 'sentence1', 'sentence2', 'score'])
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate similarity models on EBMSASS and BIOSSES datasets.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Output directory to save results.')
    parser.add_argument('-d', '--data_path', type=str, required=True, help='Path to the EBMSASS dataset directory.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('-t', '--testing', action='store_true', help='Enable testing mode (use a subset of the data).')
    parser.add_argument('-r', '--retrain', action='store_true', help='Retrain models on the training set.')
    args = parser.parse_args()

    similarity_model_evaluation(args.output_dir, args.data_path, args.verbose, args.testing, args.retrain)