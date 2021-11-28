import utils_misc, sklearn
import numpy as np

# Choosing threshold
def choose_best_threshold(labels, scores):
    best_f1 = 0.0
    best_thresh = 0.0
    thresholds = [np.percentile(scores, p) for p in np.arange(0, 100, 0.2)]
    for thresh in thresholds:
        preds = [1 if score > thresh else 0 for score in scores]
        # f1_score = sklearn.metrics.f1_score(labels, preds, average='micro')
        f1_score = sklearn.metrics.balanced_accuracy_score(labels, preds)

        if f1_score >= best_f1:
            best_f1 = f1_score
            best_thresh = thresh
    return best_thresh, best_f1

def from_score_to_pred(dataset, score_key):
    scores = [d[score_key] for d in dataset]
    labels = [d["label"] for d in dataset]
    thresh, _ = choose_best_threshold(labels, scores)

    pred_key = "pred_%s" % (score_key)
    for d in dataset:
        d[pred_key] = 1 if d[score_key] > thresh else 0



# Score computation utility

def compute_doc_level(scorer_doc, dataset):
    documents = [d["document"] for d in dataset]
    summaries = [d["claim"] for d in dataset]
    doc_scores = scorer_doc(documents, summaries, progress=True)
    label_keys = [k for k in doc_scores.keys() if "_scores" in k]

    for label_key in label_keys:
        score_key = ("%s|doc" % (label_key)).replace("_scores", "")
        for d, score in zip(dataset, doc_scores[label_key]):
            d[score_key] = score
        utils_misc.from_score_to_pred(dataset, score_key)

def compute_paragraph_level(scorer_para, dataset):
    all_paragraphs = []
    all_summaries = []
    idx_map = []
    for i, d in enumerate(dataset):
        separator = "\n\n" if d["document"].count("\n\n")>0 else "\n"
        paragraphs = d["document"].split(separator)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 0]
        all_paragraphs += paragraphs
        all_summaries += [d["claim"]] * len(paragraphs)
        idx_map += [i] * len(paragraphs)

    para_scores = scorer_para(all_paragraphs, all_summaries, progress=True)
    label_keys = [sname+"_scores" for sname in scorer_para.get_score_names()]
    for label_key in label_keys:
        score_key = ("%s|paras" % (label_key)).replace("_scores", "")
        for d in dataset:
            d[score_key] = []
        for j, score in enumerate(para_scores[label_key]):
            dataset[idx_map[j]][score_key].append(score)

        mean_k, max_k, min_k = score_key+"_mean", score_key+"_max", score_key+"_min"
        for d in dataset:
            d[mean_k] = np.mean(d[score_key])
            d[max_k] = np.max(d[score_key])
            d[min_k] = np.min(d[score_key])
        from_score_to_pred(dataset, mean_k)
        from_score_to_pred(dataset, max_k)
        from_score_to_pred(dataset, min_k)
