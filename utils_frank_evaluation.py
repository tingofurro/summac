import json
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
import numpy as np
from scipy import stats
import sys
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(description='Arguments for the evaluation script.')

    baseline_metrics = [
        # 'Bleu',
        # 'Meteor',
        # 'Rouge 1',
        # 'Rouge 2',
        'Rouge L',
        'BertScore P Art',
        # 'BertScore R Art',
        # 'BertScore F1 Art',
        # 'FEQA',
        'QAGS',
        # 'OpenIE',
        'Dep Entail',
        'FactCC',
    ]
    ablations_cols = [
        'Flip_Semantic_Frame_Errors', 'Flip_Discourse_Errors', 'Flip_Content_Verifiability_Errors',
        # 'Flip_RelE', 'Flip_EntE', 'Flip_CircE', 'Flip_OutE', 'Flip_GramE', 'Flip_CorefE', 'Flip_LinkE', 'Flip_Other'
    ]
    model_names = [
        'bart','pgn', 'bus', 'bert_sum', 's2s',
        'TranS2S', 'TConvS2S', 'PtGen', 'BERTS2S'
    ]

    parser.add_argument('--mode', default='hm-correlation', choices=['hm-correlation', 'ablations', 'ablations-plot', 'mm-correlation'], help=(  
        'This script can calculate correlation with human judgments (hm-correlation),'
        ' evaluate the performance of the evaluation metrics at capturing different types of factual errors (ablations),'
        ' output the ablation as a plot (ablations-plot), and compute the Williams test (mm-correlation)'
    ))
    parser.add_argument('--human_eval_path', default='/home/phillab/data/frank/human_annotations.json', help='file containing human annotations expects csv file.')
    parser.add_argument('--baseline_metrics_outputs', default='/home/phillab/data/frank/baseline_factuality_metrics_outputs.json', help='file name containing outputs of baseline factuality metrics.')
    parser.add_argument('--baseline_metrics', nargs='+', default=baseline_metrics, help='baseline metrics to evaluate on (should match the name in the baseline metrics output file).')
    parser.add_argument('--no_baseline_metrics', action='store_true', help='If set, does not evaluate the baseline metrics')
    parser.add_argument('--metrics_outputs', default=None, help='names of json files containing metric outputs with key "score"')
    parser.add_argument('--metrics_outputs_info', default=None, help='json file describing how to parse metrics output files. This allows to customize the name of the score key and to have several metrics in one json file.')
    parser.add_argument('--ablations', nargs='+', default=ablations_cols, help='column names for ablations.')
    parser.add_argument('--human', default='Factuality', help='column for human judgements.')
    parser.add_argument('--no_partial_correlation', action='store_true')
    parser.add_argument('--partial_correlation_variable', default='model_name', help='what column to use as confounding to calculate partial correlations')
    parser.add_argument('--store_path', default=None)
    parser.add_argument('--dataset', default=None, choices=[None, 'cnndm', 'bbc'], help='if None use all data')
    parser.add_argument('--model_name', nargs='+', default=None, help=f'by default use all data, availble model names {model_names}')
    args = parser.parse_args(args)
    return vars(args)

def williams_test(r12, r13, r23, n):
    """The Williams test (Evan J. Williams. 1959. Regression Analysis, volume 14. Wiley, New York, USA)

    A test of whether the population correlation r12 equals the population correlation r13.
    Significant: p < 0.05

    Arguments:
        r12 (float): correlation between x1, x2
        r13 (float): correlation between x1, x3
        r23 (float): correlation between x2, x3
        n (int): size of the population

    Returns:
        t (float): Williams test result
        p (float): p-value of t-dist
    """
    if r12 < r13:
        print('r12 should be larger than r13')
        sys.exit()
    elif n <= 3:
        print('n should be larger than 3')
        sys.exit()
    else:
        K = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23
        denominator = np.sqrt(2*K*(n-1)/(n-3) + (((r12+r13)**2)/4)*((1-r23)**3))
        numerator = (r12-r13) * np.sqrt((n-1)*(1+r23))
        t = numerator / denominator
        p = 1 - stats.t.cdf(t, df=n-3) # changed to n-3 on 30/11/14
        return t, p

def human_metric_correlation(
    data_df,
    human_col,
    metrics_cols,
    partial_correlation=True,
    partial_correlation_variable=None
):
    """
    human_df: pandas dataframe, should only contain one column corresponding to human judgements
    metrics_df: pandas dataframe, columns are metrics.
    partial_correlation: bool - whether to use partial correlations.

    returns a pandas dataframe with pearson and spearman correlation results
    """
    correlations = []
    named_correlations = dict()
    for metric in metrics_cols:
        if metric not in data_df:
            correlations.append([0, 0, 0, 0])
            named_correlations[metric] = [0, 0, 0, 0]
            print(f'Warning: {metric} not in dataframe.')
            continue
        mask = (data_df[metric].isnull() == False) & (data_df[human_col].isnull() == False)
        X = data_df[metric][mask]
        Y = data_df[human_col][mask]
        if partial_correlation:
            assert partial_correlation_variable is not None, f'You must specify a column to use as confounding variable for partial correlation calculation'
            Q = np.array(data_df[mask][partial_correlation_variable])
            enc = OneHotEncoder(handle_unknown='ignore')
            Q = enc.fit_transform(Q.reshape(-1, 1))
            pred_X = LinearRegression().fit(Q, X).predict(Q)
            pred_Y = LinearRegression().fit(Q, Y).predict(Q)
            X = X - pred_X
            Y = Y - pred_Y
        print(f'Info: metric {metric} used {len(X)} summaries to calculate correlation.')
        pr, pp = pearsonr(X, Y)
        sr, sp = spearmanr(X, Y)
        correlations.append([pr, pp, sr, sp])
        named_correlations[metric] = [pr, pp, sr, sp]
    correlation_df = pd.DataFrame.from_dict(
        named_correlations,
        orient='index',
        columns=['pearson', 'pearson p-value', 'spearman', 'spearman p-value']
    )
    return correlation_df

def metric_metric_correlation(
    data_df,
    human_col,
    metrics_cols,
    partial_correlation=True,
    partial_correlation_variable=None
):
    """
    metrics_df: pandas dataframe, columns taken as metrics
    partial_correlation: bool - whether to use partial correlations.

    returns of tuple with two dataframes: (correlation_df, williams_df)
    correlation_df is a dataframe that contains metric-metric pearson correlation
    williams_df is a dataframe of booleans on weather the two metrics are different in statistically significant terms
    """
    correlations = []
    williams = []
    for i, metric1 in enumerate(metrics_cols):
        correlation_metric = []
        williams_metric = []
        for j, metric2 in enumerate(metrics_cols):
            if j == i:
                correlation_metric.append(1)
                williams_metric.append(False)
                continue
            mask1 = (data_df[metric1].isnull() == False) & (data_df['model_name'] != 'reference')
            mask2 = (data_df[metric2].isnull() == False) & (data_df['model_name'] != 'reference')
            mask3 = (data_df[human_col].isnull() == False)
            mask = mask1 & mask2 & mask3
            X = data_df[metric1][mask]
            Y = data_df[metric2][mask]
            Z = data_df[human_col][mask]
            if partial_correlation_variable is not None:
                Q = np.array(data_df[mask][partial_correlation_variable])
                enc = OneHotEncoder(handle_unknown='ignore')
                Q = enc.fit_transform(Q.reshape(-1, 1))
                pred_X = LinearRegression().fit(Q, X).predict(Q)
                pred_Y = LinearRegression().fit(Q, Y).predict(Q)
                pred_Z = LinearRegression().fit(Q, Z).predict(Q)
                X = X - pred_X
                Y = Y - pred_Y
                Z = Z - pred_Z
            r12, _ = pearsonr(X, Z)
            r13, _ = pearsonr(Y, Z)
            r23, _ = pearsonr(X, Y)
            n = min(len(X), len(Y))
            if r12 < r13:
                r12, r13 = r13, r12
            _, p = williams_test(r12, r13, r23, n)
            correlation_metric.append(r23)
            williams_metric.append(p)
        correlations.append(correlation_metric)
        williams.append(williams_metric)
    correlations_df = pd.DataFrame(correlations, index=metrics_cols, columns=metrics_cols)
    williams_df = pd.DataFrame(williams, index=metrics_cols, columns=metrics_cols)
    return (correlations_df, williams_df)

def ablation(
    data_df, 
    human_col, 
    ablations_cols, 
    metrics_cols, 
    partial_correlation=True, 
    partial_correlation_variable=None
):
    """
    human_df: pandas dataframe, should only contain one column corresponding to human judgements
    ablations_df: pandas dataframe, each column corresponds to a different ablation of the human judgements
    metrics_df: pandas dataframe, columns are metrics.
    partial_correlation: bool - whether to use partial correlations.

    returns a dataframe each row corresponding to a different ablation
    """
    ablations_dict = dict()
    
    human_df = human_metric_correlation(data_df, human_col, metrics_cols, partial_correlation=partial_correlation, partial_correlation_variable=partial_correlation_variable)
    human_correlation = human_df['pearson']

    for ablation in ablations_cols:
        ablation_df = human_metric_correlation(data_df, ablation, metrics_cols, partial_correlation=partial_correlation, partial_correlation_variable=partial_correlation_variable)
        ablation_correlation = ablation_df['pearson']
        ablations_dict[ablation] = human_correlation - ablation_correlation
    ablations_df = pd.DataFrame(ablations_dict, index=metrics_cols)
    return ablations_df

def plot_ablations(ablation_df, save_path):
    """
    ablation_df: pandas dataframe, the output of ablation function
    save_path: str, where to save the plot

    Plots the ablation_df and possibly saves it to the location
    """
    ax = ablation_df.plot.bar(figsize=(10, 4), rot=0)
    plt.xticks(rotation=45)
    if not save_path:
        save_path = '.'
    fig = ax.get_figure()
    fig.savefig(os.path.join(save_path, 'ablations_plot.pdf'), bbox_inches='tight')

def main(args):
    """
    Depending on the `mode` used, this script computes correlation between factuality metrics
    and human judgments of factuality on the FRANK benchmark data. It can also measure how well
    a metric captures certain types of errors.

    The code uses baseline metric outputs provided as part of FRANK (in `baseline_facutlaity_metrics_outputs.json`).
    The user can specify which metrics among the baseline metrics to use in the computation.

    In addition to the baseline metrics, this tool allows users to evaluate their own factuality metric outputs on
    FRANK. There are two ways to do so:
    1. By providing a FRANK benchmark submission file: a `json` files containing a list of records, each record
       having both `hash` and `model_name` fields as well as a `score` field with the metric output.
    2. By defining a `json` file with information on how to parse the metric output files. 
       the schema should look like:
       [
           {
               "path": "PATH_TO_JSON_FILE_WITH_OUTPUS"
               "scores": [
                   {"name": "PRETTY NAME FOR THE METRIC 1", "key": "THE KEY CONTAINING THE METRIC 1 OUTPUT"},
                   {"name": "PRETTY NAME FOR THE METRIC 2", "key": "THE KEY CONTAINING THE METRIC 2 OUTPUT"},
                   ...
               ]
           },...
       ]
       Note that the output files should still be `json` files with a list of records with `hash` and 
       `model_name` keys, but they can contain several metrics outputs in each record .
       This allows to specify a name for each metric, and allows several metrics for each output file. 
    """
    # Load the human judgements.
    data_df = pd.read_json(args['human_eval_path'])

    human_col = args['human']
    ablations_cols = args['ablations']
    metrics_cols = []
    
    # Load the metric outputs.
    if not args['no_baseline_metrics']:
        metric_df = pd.read_json(args['baseline_metrics_outputs'])
        for baseline_metric in args['baseline_metrics']:
            assert baseline_metric in metric_df, baseline_metric + ' not found. Your metrics_output_info file is likely not well defined.'
        data_df = data_df.merge(metric_df[['hash', 'model_name'] + args['baseline_metrics']], on=['hash', 'model_name'], validate='one_to_one')
        metrics_cols += args['baseline_metrics']
    
    if args['metrics_outputs']:
        metric_df = pd.read_json(args['metrics_outputs'])
        assert 'score' in metric_df, 'The metric output should be in a field named "score"'
        data_df = data_df.merge(metric_df[['hash', 'model_name', 'score']], on=['hash', 'model_name'], validate='one_to_one')
        metrics_cols += ['score']

    if args['metrics_outputs_info']:
        with open(args['metrics_outputs_info']) as infile:
            metrics_info = json.loads(infile.read())
        for metric_info in metrics_info:
            metric_df = pd.read_json(metric_info['path'])
            keys = []
            for score_info in metric_info['scores']:
                assert score_info['key'] in metric_df, score_info['key']+' not found. Your metrics_output_info file is likely not well defined.'
                keys.append(score_info['key'])
            data_df = data_df.merge(metric_df[['hash', 'model_name'] + keys], on=['hash', 'model_name'], validate='one_to_one')
            data_df = data_df.rename(columns={score_info['key']:score_info['name'] for score_info in metric_info['scores']})
            metrics_cols += [score_info['name'] for score_info in metric_info['scores']]

    # Select dataset and models if specified.
    if args['dataset']:
        mask = (data_df['dataset'] == args['dataset']) & (data_df['model_name'] != 'reference')
        data_df = data_df[mask]
    if args['model_name']:
        mask = (data_df['model_name'].isin(args['model_name'])) & (data_df['model_name'] != 'reference')
        data_df = data_df[mask]

    out_df, williams_df = None, None
    if args['mode'] == 'hm-correlation':
        out_df = human_metric_correlation(
            data_df,
            human_col,
            metrics_cols,
            partial_correlation=not args['no_partial_correlation'],
            partial_correlation_variable=args['partial_correlation_variable']
        )
    elif args['mode'] == 'mm-correlation':
        out_df, williams_df = metric_metric_correlation(
            data_df,
            human_col,
            metrics_cols,
            partial_correlation=not args['no_partial_correlation'],
            partial_correlation_variable=args['partial_correlation_variable']
        )
    elif args['mode'] == 'ablations':
        out_df = ablation(
            data_df, 
            human_col, 
            ablations_cols, 
            metrics_cols, 
            partial_correlation=not args['no_partial_correlation'], 
            partial_correlation_variable=args['partial_correlation_variable']
        )
    elif args['mode'] == 'ablations-plot':
        out_df = ablation(
            data_df, 
            human_col, 
            ablations_cols, 
            metrics_cols, 
            partial_correlation=not args['no_partial_correlation'], 
            partial_correlation_variable=args['partial_correlation_variable']
        )
        plot_ablations(out_df, args['store_path'])
    else:
        raise KeyError

    print(out_df)
    if args['store_path']:
        if out_df is not None:
            out_df.to_json(os.path.join(args['store_path'], 'out_df.json'), indent=4)
        if williams_df is not None:
            williams_df.to_json(os.path.join(args['store_path'], 'williams.json'), indent=4)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)