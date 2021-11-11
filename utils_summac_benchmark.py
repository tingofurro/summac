import json, os, pandas as pd, numpy as np, sklearn, csv
from datasets import load_dataset
from collections import Counter

# SummaC Benchmark
def load_factcc(dataset_folder="/home/phillab/data/summac_benchmark/factcc/", cut="val", max_entries=-1):
    # Evaluating the Factual Consistency of Abstractive Text Summarization [https://arxiv.org/abs/1910.12840]
    # Dataset for each split must be downloaded from the Github repo: https://github.com/salesforce/factCC
    if cut == "train":
        dataset = []
        with open(os.path.join(dataset_folder, "training_data/data-original/data-train.jsonl"), "r") as f:
            for i, line in enumerate(f):
                if max_entries > 0 and i >= max_entries:
                    break
                D = json.loads(line)
                with open(os.path.join(dataset_folder.replace("factcc/", ""), D["filepath"]), "r") as f2:
                    full_text = f2.read()
                full_text = full_text.split("@highlight")[0].strip()

                label = 1 if D["label"]=="CORRECT" else 0
                datum = {"document": full_text, "claim": D["claim"], "cnndm_id": D["id"], "label": label, "task": "factcc"}
                dataset.append(datum)
        return dataset

    if cut in ["val", "test"]:
        factcc_file = os.path.join(dataset_folder, "%s/data-dev.jsonl" % (cut))
        dataset = []
        with open(factcc_file, "r") as f:
            for line in f:
                dataset.append(json.loads(line))

        benchmark_folder = dataset_folder.replace("factcc", "")

        for d in dataset:
            with open(os.path.join(benchmark_folder, d["filepath"]), "r") as f:
                d["document"] = f.read().split("@highlight")[0].strip()
            d["label"] = 1 if d["label"] == "CORRECT" else 0
            d["annotations"] = [d["label"]]
            d["task"] = "factcc"
        return dataset

def load_polytope(dataset_folder="/home/phillab/data/summac_benchmark/polytope/", which_label="overall", cut="val"):
    # What Have We Achieved on Text Summarization? [https://arxiv.org/abs/2010.04529]
    # Dataset must be downloaded from the Github repo: https://github.com/hddbang/polytope

    assert which_label in ["overall", "omission", "addition", "duplication", "inaccuracy"], "Unrecognized `which label`"
    full_dataset = []
    for fn in os.listdir(dataset_folder):
        fn = os.path.join(dataset_folder, fn)

        all_segments = pd.read_excel(fn, sheet_name="Scores per segment")
        ID2row = {}
        for i, segment in all_segments.iterrows():
            c = "val" if i % 2 == 0 else "test"
            if str(segment["ID"]) != "nan":
                ID2row[segment["ID"]] = {"ID": segment["ID"], "document": segment["Source"], "claim": segment["Target"], "errors": [], "cut": c}

        for i, row in pd.read_excel(fn, sheet_name="Error Log").iterrows():
            if str(row["Subtypes"]) != "nan":
                ID2row[row["ID"]]["errors"].append(row["Subtypes"])

        for ID in ID2row:
            d = ID2row[ID]
            d["overall_label"] = 1 if len(d["errors"]) == 0 else 0
            d["omission_label"] = 0 if "Omission" in d["errors"] else 1
            d["addition_label"] = 0 if "Addition" in d["errors"] else 1
            d["duplication_label"] = 0 if "Duplication" in d["errors"] else 1
            d["inaccuracy_label"] = 0 if "Inaccuracy_internal" in d["errors"] or "Inaccuracy_external" in d["errors"] else 1
            if which_label is not None:
                d["label"] = d["%s_label" % (which_label)]
            d["task"] = "polytope"
            d["annotations"] = [d["label"]]

            full_dataset.append(d)
    cut_dataset = [d for d in full_dataset if d["cut"]==cut]
    return cut_dataset

def cnndm_aid2document(aid, cnndm_folder="/home/phillab/data/summac_benchmark/cnndm/"):
    cnn_file = os.path.join(cnndm_folder, "cnn/stories/%s.story" % (aid))
    dm_file = os.path.join(cnndm_folder, "dailymail/stories/%s.story" % (aid))
    final_file = cnn_file if os.path.isfile(cnn_file) else dm_file
    article = open(final_file, "r").read().split("@highlight")[0].strip()
    return article

def load_cogensumm(dataset_folder="/home/phillab/data/summac_benchmark/cogensumm/", cut="val"):
    # Correctness of Generated Summaries: https://www.aclweb.org/anthology/P19-1213.pdf
    # CoGenSumm: https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2002

    assert cut in ["val", "test"], "cut value unrecognized"

    cnndm_folder = os.path.join(dataset_folder.replace("cogensumm", ""), "cnndm/")

    clean_dataset = []
    for fn in os.listdir(dataset_folder):
        if cut not in fn:
            continue

        with open(os.path.join(dataset_folder, fn), "r") as f:
            dataset = json.load(f)

        if "_org" in fn or fn == "test_chen18_reranked.json":
            for aid in dataset:
                document = cnndm_aid2document(aid, cnndm_folder=cnndm_folder)
                label = 0 if dataset[aid]["label"] == "Incorrect" else 1
                sents = dataset[aid]["sents"]
                summary = " ".join([sents[str(i)]["text"] for i in range(len(sents))])
                clean_dataset.append({"filename": fn, "label": label, "document": document, "claim": summary, "cnndm_id": aid, "annotations": [label], "task": "cogensumm"})
        elif fn == "val_reranking.json":
            for aid in dataset:
                document = cnndm_aid2document(aid, cnndm_folder=cnndm_folder)
                for idx, data in dataset[aid].items():
                    label = 0 if data["label"] == "Incorrect" else 1
                    summary = " ".join([data["sents"][str(i)]["text"] for i in range(len(data["sents"]))])
                    clean_dataset.append({"filename": fn, "label": label, "document": document, "claim": summary, "cnndm_id": aid, "annotations": [label], "task": "cogensumm"})
        elif fn == "val_sentence_pairs.json":
            for d in dataset:
                document = cnndm_aid2document(d["article_id"], cnndm_folder=cnndm_folder)
                clean_dataset.append({"filename": fn, "label": 1, "document": document, "claim": d["correct_sent"], "cnndm_id": aid, "annotations": [label], "task": "cogensumm"})
                clean_dataset.append({"filename": fn, "label": 0, "document": document, "claim": d["incorrect_sent"], "cnndm_id": aid, "annotations": [label], "task": "cogensumm"})
    return clean_dataset

def load_frank(dataset_folder="/home/phillab/data/summac_benchmark/frank/", cut="val"):
    # FRANK: Factuality Evaluation Benchmark [https://aclanthology.org/2021.naacl-main.383.pdf]
    # Files must be downloaded from the Github repository: https://github.com/artidoro/frank
    raw_file = os.path.join(dataset_folder, "frank_human_annotations_sentence.json")
    val_hash_file = os.path.join(dataset_folder, "validation_split.txt")
    test_hash_file = os.path.join(dataset_folder, "test_split.txt")
    with open(val_hash_file if cut=="val" else test_hash_file, "r") as f:
        valid_hashes = set([line.strip() for line in f])

    with open(raw_file, "r") as f:
        raw_dataset = json.load(f)
    dataset = []
    for d in raw_dataset:
        article = d["article"]
        if d["hash"] not in valid_hashes:
            continue

        summ_labels = []

        annotator_labels = {}
        for annot in d["summary_sentences_annotations"]:
            annot_vals = [an for ans in annot.values() for an in ans]
            noerror_count = len([an for an in annot_vals if an=="NoE"])
            label = 1 if noerror_count >= 2 else 0
            summ_labels.append(label)
            for anno_name, anno in annot.items():
                if anno_name not in annotator_labels:
                    annotator_labels[anno_name] = []
                annotator_labels[anno_name] += anno

        annotations = [1 if all(a=="NoE" for a in annos) else 0 for annos in annotator_labels.values()]
        label = 0 if any(sl==0 for sl in summ_labels) else 1
        summary = d["summary"]
        dataset.append({"document": article, "claim": summary, "label": label, "cut": cut, "hash": d["hash"], "model_name": d["model_name"], "annotations": annotations, "task": "frank"})
    return dataset

def load_summeval(dataset_folder="/home/phillab/data/summac_benchmark/summeval/", cut="val"):
    # SummEval: Re-evaluating Summarization Evaluation [https://arxiv.org/abs/2007.12626]
    # Data files must be downloaded from the following Github repository: https://github.com/Yale-LILY/SummEval
    raw_dataset = []

    fn = os.path.join(dataset_folder, "summ_eval_model_annotations.aligned.scored.jsonl")
    with open(fn, "r") as f:
        for line in f:
            raw_dataset.append(json.loads(line))

    cnndm_dataset = load_dataset("cnn_dailymail", "3.0.0")
    cnndm_id2article = {d["id"]: d["article"] for d in cnndm_dataset["test"]}

    clean_dataset = []

    for i, d in enumerate(raw_dataset):
        cut = "val" if i % 2 == 0 else "test"
        _, _, article_id = d["id"].split("-")
        document = cnndm_id2article[article_id]
        annotations = d["expert_annotations"] # + d.get("turker_annotations", [])

        consistencies = [a["consistency"] for a in annotations]
        final_label = 1 if len([cons for cons in consistencies if cons==5]) > len(annotations)/2 else 0

        annotations = [1 if cons == 5 else 0 for cons in consistencies]
        clean_dataset.append({"document": document, "claim": d["decoded"], "label": final_label, "model_name": d["model_id"], "cnndm_id": d["id"], "cut": cut, "annotations": annotations, "task": "summeval"})
    final_dataset = [d for d in clean_dataset if d["cut"] == cut]
    return final_dataset

def load_xsumfaith(dataset_folder="/home/phillab/data/summac_benchmark/xsumfaith/", cut="val"):
    
    path_to_annotation = os.path.join(dataset_folder, "hallucination_annotations_xsum_summaries.csv")

    xsum = load_dataset("xsum")["test"]
    xsumid_2_article = {d["id"]: d["document"] for d in xsum}

    with open(path_to_annotation, "r") as f:
        raw_data = list(csv.reader(f))
        dataset = []
        keys = raw_data[0]
        for line in raw_data[1:]:
            dataset.append({k: v for k, v in zip(keys, line)})

    groups = {}
    for d in dataset:
        k = (d["bbcid"], d["system"])
        if k not in groups:
            groups[k] = []
        groups[k].append(d)

    clean_dataset = []
    for k, vs in groups.items():
        A = vs[0]
        document = xsumid_2_article[A["bbcid"]]
        labels = [v["hallucination_type"] for v in vs]
        annotations = [1 if label == "NULL" else 0 for label in labels]
        most_common_label = Counter(labels).most_common(1)[0][0]
        label = 1 if most_common_label == "NULL" else 0
        c = "val" if len(clean_dataset) % 2 == 0 else "test"

        clean_dataset.append({"document": document, "claim": A["summary"], "bbcid": A["bbcid"], "model_name": A["system"], "label": label, "cut": c, "annotations": annotations, "task": "xsumfaith"})
    final_dataset = [d for d in clean_dataset if d["cut"]==cut]
    return final_dataset

def choose_best_threshold(labels, scores):
    best_f1 = 0.0
    best_thresh = 0.0
    thresholds = [np.percentile(scores, p) for p in np.arange(0, 100, 0.2)]
    for thresh in thresholds:
        preds = [1 if score > thresh else 0 for score in scores]
        # f1_score = sklearn.metrics.f1_score(labels, preds, average='micro')
        f1_score = sklearn.metrics.balanced_accuracy_score(labels, preds)

        # print(sklearn.metrics.f1_score(labels, preds, average='micro'), sklearn.metrics.f1_score(labels, preds, average='macro'), )

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

def compute_doc_level(scorer_doc, dataset):
    documents = [d["document"] for d in dataset]
    summaries = [d["claim"] for d in dataset]
    doc_scores = scorer_doc(documents, summaries, progress=True)
    label_keys = [k for k in doc_scores.keys() if "_scores" in k]

    for label_key in label_keys:
        score_key = ("%s|doc" % (label_key)).replace("_scores", "")
        for d, score in zip(dataset, doc_scores[label_key]):
            d[score_key] = score
        from_score_to_pred(dataset, score_key)

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

class SummaCBenchmark:
    def __init__(self, benchmark_folder="/home/phillab/data/summac_benchmark/", cut="val"):
        assert cut in ["val", "test"], "Unrecognized cut for the Fact Checking Benchmark"
        self.cut = cut
        self.tasks = [
                      {"name": "cogensumm", "task": load_cogensumm(dataset_folder=os.path.join(benchmark_folder, "cogensumm"), cut=self.cut)},
                      {"name": "xsumfaith", "task": load_xsumfaith(dataset_folder=os.path.join(benchmark_folder, "xsumfaith"), cut=self.cut)},
                      {"name": "polytope", "task": load_polytope(dataset_folder=os.path.join(benchmark_folder, "polytope"), which_label="overall", cut=self.cut)},
                      {"name": "factcc", "task": load_factcc(dataset_folder=os.path.join(benchmark_folder, "factcc"), cut=self.cut)},
                      {"name": "summeval", "task": load_summeval(dataset_folder=os.path.join(benchmark_folder, "summeval"), cut=self.cut)},
                      {"name": "frank", "task": load_frank(dataset_folder=os.path.join(benchmark_folder, "frank"), cut=self.cut)},
                      ]
        self.model_majorities = {'frank_BERTS2S': 0, 'frank_TConvS2S': 0, 'frank_PtGen': 0, 'frank_TranS2S': 0, 'frank_bart': 1, 'frank_bert_sum': 1, 'frank_bus': 0, 'frank_pgn': 1, 'frank_s2s': 0, 'summeval_M13': 1, 'summeval_M14': 1,
        'summeval_M12': 1, 'summeval_M17': 1, 'summeval_M23': 1, 'summeval_M0': 1, 'summeval_M8': 1, 'summeval_M9': 1, 'summeval_M11': 1, 'summeval_M1': 1, 'summeval_M15': 1, 'summeval_M5': 1, 'summeval_M20': 1, 'summeval_M2': 1, 'summeval_M22': 1,
        'summeval_M10': 1, 'summeval_M23_dynamicmix': 1, 'xsumfaith_BERTS2S': 0, 'xsumfaith_TConvS2S': 0, 'xsumfaith_Gold': 0, 'xsumfaith_PtGen': 0, 'xsumfaith_TranS2S': 0}

        self.task_name_to_task = {task["name"]: task["task"] for task in self.tasks}

    def print_stats(self):
        dataset_stats = []
        for task in self.tasks:
            N_pos, N_neg = len([d for d in task["task"] if d["label"]==1]), len([d for d in task["task"] if d["label"]==0])
            dataset_stats.append({"name": task["name"], "N": len(task["task"]), "N_pos": N_pos, "N_neg": N_neg, "frac_pos": N_pos/(N_pos+N_neg)})
        print(pd.DataFrame(dataset_stats))

    def get_task(self, task_name):
        return self.task_name_to_task[task_name]

    def evaluate(self, scorer):
        benchmark = []

        for task in self.tasks:
            task_labels = [d["label"] for d in task["task"]]
            task_preds = scorer.score([d["document"] for d in task["task"]], [d["claim"] for d in task["task"]])["scores"]

            task_thresh, task_f1 = choose_best_threshold(task_labels, task_preds)
            benchmark.append({"name": task["name"], "score": task_f1, "threshold": task_thresh})
        return {"overall_score": np.mean([t["score"] for t in benchmark]), "benchmark": benchmark}


if __name__ == "__main__":
    summac_benchmark = SummaCBenchmark()

    # dataset_val = load_summeval()

    # dataset_val = load_frank(cut="val")
    # dataset_test = load_frank(cut="test")
    # print(len(dataset_val), len(dataset_test))
