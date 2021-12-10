import pandas as pd
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import argparse

factors = ['coherence', 'consistency', 'fluency', 'relevance']
levels = ['expert']
human_cols = sum([[f'{level}_{factor}' for factor in factors] for level in levels], [])
method_cols = ['shannon_score', 'info_diff', 'blanc_shannon']
models = ['small', 'medium', 'large', 'xl', 'gpt1', 'xlnet', 'transformerxl']

def load(input_file):
    with open(input_file) as reader:
        result_blobs = [json.loads(line) for line in reader]

    for blob in result_blobs:
        S = blob['S']
        blob['blanc_shannon'] = (S[0][1] - S[1][0]) / sum(sum(S, []))
        blob['s_impr'] = S[0][1] / (S[0][0] + S[1][1] + S[0][1])
        del blob['S']

    result_df = pd.DataFrame.from_records(result_blobs)

    with open('/nfs/data/summeval/docs-dynamicmix.json') as reader:
        input_blobs = json.load(reader)

    annotation_blobs = []
    for blob in input_blobs:
        annotation_blob = {'id': blob['id']}
        for level in levels:
            annotations = blob.get(f'{level}_annotations')
            if annotations is None:
                continue
            for factor in factors:
                mean = sum([annotation[factor] for annotation in annotations]) / len(annotations)
                annotation_blob[f'{level}_{factor}'] = mean
        annotation_blob['lower'] = all([c.lower() == c for c in blob['summ']])
        annotation_blobs.append(annotation_blob)

    annotation_df = pd.DataFrame.from_records(annotation_blobs)
    df = result_df.join(annotation_df)
    df['shannon_score'] = (df.ll_help - df.ll_base) / (df.ll_full - df.ll_base)
    df['info_decr'] = (df.ll_base - df.ll_help) / df.ll_base
    df['compression'] = df.num_summ_tokens / df.num_doc_tokens
    df['cond_lik'] = df.ll_help
    df['info_diff'] = df.ll_help - df.ll_base
    df['avg_cond_lik'] = df.ll_help / df.num_doc_tokens

    systems = df.groupby('system').mean()
    return df, systems

def eval_many(models, filenames):
    shannon_scores, dfs = {}, []
    for model, filename in zip(models, filenames):
        df, systems = load(filename)
        shannon_scores[model] = df.shannon_score
        print(f'{model} system kendall-b')
        print(systems.corr(method='kendall')[human_cols].loc[method_cols])
        df['model_name'] = model
        dfs.append(df)
    shannon_scores = pd.DataFrame(shannon_scores)
    combined = pd.concat(dfs).reset_index()
    print(shannon_scores.corr())

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--upstream', action='store_true')
args = parser.parse_args()

if args.upstream:
    eval_many(range(5), [f'out/context-{context}-small.jsonl' for context in range(5)])
elif args.model is None:
    eval_many(models, [f'out/17-{model}.jsonl' for model in models])
else:
    df, systems = load(f'out/17-{args.model}.jsonl')
    print('Overall Pearson')
    print(df.corr(method='pearson')[human_cols].loc[method_cols])
    print('Overall Spearman')
    print(df.corr(method='spearman')[human_cols].loc[method_cols])
    print('Overall Kendall-b')
    print(df.corr(method='kendall')[human_cols].loc[method_cols])

    print('System Pearson')
    print(systems.corr(method='pearson')[human_cols].loc[method_cols])
    print('System Spearman')
    print(systems.corr(method='spearman')[human_cols].loc[method_cols])
    print('System Kendall-b')
    print(systems.corr(method='kendall')[human_cols].loc[method_cols])
