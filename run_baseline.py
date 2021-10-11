from utils_summac_benchmark import load_factcc, load_polytope, load_cogensumm, load_frank, load_summeval, load_xsumfaith
import sklearn, numpy as np, os, pandas as pd, sys, argparse
from model_baseline import BaselineScorer
from utils_scoring import ScorerWrapper
import utils_misc, seaborn as sns
from collections import Counter

sys.path.insert(0, "/home/phillab/feqa/")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# For now, can't use argparse because it is hard-coded in DAE... very shitty
# parser = argparse.ArgumentParser()
# parser.add_argument("--model", type=str, choices=["questeval", "feqa", "dae"], default="questeval")
# parser.add_argument("--questeval_weighter", action="store_true")
# parser.add_argument("--cut", type=str, choices=["val", "test"], default="val")
# args = parser.parse_args()

model = "dae"
cut = "test"

utils_misc.DoublePrint("%s_%s.log" % (model, cut))

def choose_best_threshold(labels, scores):
    best_bacc = 0.0
    best_thresh = 0.0
    thresholds = [np.percentile(scores, p) for p in np.arange(0, 100, 1)]
    for thresh in thresholds:
        preds = [1 if score > thresh else 0 for score in scores]
        bacc_score = sklearn.metrics.balanced_accuracy_score(labels, preds)
        if bacc_score >= best_bacc:
            best_bacc = bacc_score
            best_thresh = thresh
    return best_thresh

def from_score_to_pred(dataset, score_key):
    scores = [d[score_key] for d in dataset]
    labels = [d["label"] for d in dataset]
    thresh = choose_best_threshold(labels, scores)

    pred_key = "pred_%s" % (score_key)
    for d in dataset:
        d[pred_key] = 1 if d[score_key]>thresh else 0


datasets = [{"name": "factcc", "dataset": load_factcc(cut=cut)},
            {"name": "frank", "dataset": load_frank(cut=cut)},
            {"name": "pt_any", "dataset": load_polytope(which_label="overall", cut=cut)},
            {"name": "summ_corr", "dataset": load_cogensumm(cut=cut)},
            {"name": "summeval", "dataset": load_summeval(cut=cut)},
            {"name": "xsumfaith", "dataset": load_xsumfaith(cut=cut)}
            ]

dataset_stats = []
for dataset in datasets:
    N_pos, N_neg = len([d for d in dataset["dataset"] if d["label"]==1]), len([d for d in dataset["dataset"] if d["label"]==0])
    dataset_stats.append({"name": dataset["name"], "N": len(dataset["dataset"]), "N_pos": N_pos, "N_neg": N_neg, "frac_pos": N_pos/(N_pos+N_neg)})

print(pd.DataFrame(dataset_stats))

scorers = []
if model == "questeval":
    scorers.append({"name": "QuestEval", "model": BaselineScorer(model="questeval", do_weighter=args.questeval_weighter), "sign": 1})
elif model == "feqa":
    scorers.append({"name": "FEQA", "model": BaselineScorer(model="feqa"), "sign": 1})
elif model == "dae":
    scorers.append({"name": "DAE", "model": BaselineScorer(model="dae"), "sign": 1})


for scorer in scorers:
    scorer["model"].load_cache()

batch_size = 100
scorer_doc = ScorerWrapper(scorers, scoring_method="sum", max_batch_size=batch_size, use_caching=True)

def compute_doc_level(dataset):
    documents = [d["document"] for d in dataset]
    summaries = [d["claim"] for d in dataset]
    doc_scores = scorer_doc(documents, summaries, progress=True)
    label_keys = [k for k in doc_scores.keys() if "_scores" in k]

    for label_key in label_keys:
        score_key = ("%s|doc" % (label_key)).replace("_scores", "")
        for d, score in zip(dataset, doc_scores[label_key]):
            d[score_key] = score
        from_score_to_pred(dataset, score_key)


results = []
for dataset in datasets:
    print("======= %s ========" % (dataset["name"]))
    datas = dataset["dataset"]
    compute_doc_level(datas)
    for scorer in scorers:
        scorer["model"].save_cache()

    pred_labels = [k for k in datas[0].keys() if "pred_" in k]
    for pred_label in pred_labels:
        preds = [d[pred_label] for d in datas]
        labels = [d["label"] for d in datas]
        model_name, input_type = pred_label.replace("pred_", "").split("|")

        label_counts = Counter([d["label"] for d in dataset["dataset"]])
        pos_label = 0 if label_counts[0] < label_counts[1] else 1

        f1 = sklearn.metrics.f1_score(labels, preds, pos_label=pos_label)
        balanced_acc = sklearn.metrics.balanced_accuracy_score(labels, preds)

        results.append({"model_name": model_name, "dataset_name": dataset["name"], "input": input_type, "%s_f1" % (dataset["name"]): f1, "%s_bacc" % (dataset["name"]): balanced_acc})

cm = sns.light_palette("green", as_cmap=True)

def highlight_max(data):
    is_max = data == data.max()
    return ['font-weight: bold' if v else '' for v in is_max]


df = pd.DataFrame(results)
df = df.groupby(["model_name", "input"]).agg({"%s_bacc" % (d["name"]): "mean" for d in datasets})

df = df.rename(columns={k: k.replace("_bacc", "") for k in df.keys()})
df.drop("total", inplace=True)

print(df)
df.to_csv("/home/phillab/%s_results.csv" % (model))
# print(df.style.apply(highlight_max).background_gradient(cmap=cm, high=1.0, low=0.0).set_precision(3).set_caption("Weighed Accuracy"))
