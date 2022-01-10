import json, os, pandas as pd, numpy as np, csv
from datasets import load_dataset
from collections import Counter
import requests, zipfile, tarfile
import utils_scorer, utils_misc

# SummaC Benchmark
class SummaCBenchmark:

    def __init__(self, benchmark_folder="/home/phillab/data/summac_benchmark/", dataset_names=["cogensum", "xsumfaith", "polytope", "factcc", "summeval", "frank"], cut="val"):
        assert cut in ["val", "test"], "Unrecognized cut for the Fact Checking Benchmark"
        if not os.path.exists(benchmark_folder):
            os.makedirs(benchmark_folder)

        self.cut = cut
        self.benchmark_folder = benchmark_folder
        self.cnndm = None
        self.xsum = None

        self.datasets = []
        for dataset_name in dataset_names:
            if dataset_name == "cogensum":
                self.load_cogensumm()
            elif dataset_name == "xsumfaith":
                self.load_xsumfaith()
            elif dataset_name == "polytope":
                self.load_polytope()
            elif dataset_name == "factcc":
                self.load_factcc()
            elif dataset_name == "summeval":
                self.load_summeval()
            elif dataset_name == "frank":
                self.load_frank()
            else:
                raise ValueError("Unrecognized dataset name: %s" % (dataset_name))

    # Underlying dataset loader: CNN/DM and XSum
    def get_cnndm_document(self, aid):
        if self.cnndm is None:
            self.cnndm = load_dataset("cnn_dailymail", "3.0.0")
            self.cnndm_id2article = {}
            for cut in ["test", "validation"]:
                self.cnndm_id2article.update({d["id"]: d["article"] for d in self.cnndm[cut]})
        return self.cnndm_id2article[aid]

    def get_xsum_document(self, aid):
        if self.xsum is None:
            self.xsum = load_dataset("xsum")["test"]
            self.xsumid2article = {d["id"]: d["document"] for d in self.xsum}

        return self.xsumid2article[aid]

    # Individual dataset loaders
    def load_cogensumm(self):
        # Correctness of Generated Summaries: https://www.aclweb.org/anthology/P19-1213.pdf
        # CoGenSumm: https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2002

        dataset_folder = os.path.join(self.benchmark_folder, "cogensumm/")
        if not os.path.exists(dataset_folder):
            print("==== CoGenSumm dataset not found, downloading from scratch")
            os.makedirs(dataset_folder)
            data = requests.get("https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/2002/summary-correctness-v1.0.zip?sequence=3&isAllowed=y")
            zip_file = os.path.join(dataset_folder, "summary-correctness-v1.0.zip")
            with open(zip_file, "wb") as f:
                f.write(data.content)

            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(dataset_folder)
            os.remove(zip_file)

        clean_dataset = []
        for fn in os.listdir(dataset_folder):
            if self.cut not in fn:
                continue

            with open(os.path.join(dataset_folder, fn), "r") as f:
                dataset = json.load(f)

            if "_org" in fn or fn == "test_chen18_reranked.json":
                for aid in dataset:
                    document = self.get_cnndm_document(aid)
                    label = 0 if dataset[aid]["label"] == "Incorrect" else 1
                    sents = dataset[aid]["sents"]
                    summary = " ".join([sents[str(i)]["text"] for i in range(len(sents))])
                    clean_dataset.append({"filename": fn, "label": label, "document": document, "claim": summary, "cnndm_id": aid, "annotations": [label], "dataset": "cogensumm", "origin": "cnndm"})
            elif fn == "val_reranking.json":
                for aid in dataset:
                    document = self.get_cnndm_document(aid)
                    for idx, data in dataset[aid].items():
                        label = 0 if data["label"] == "Incorrect" else 1
                        summary = " ".join([data["sents"][str(i)]["text"] for i in range(len(data["sents"]))])
                        clean_dataset.append({"filename": fn, "label": label, "document": document, "claim": summary, "cnndm_id": aid, "annotations": [label], "dataset": "cogensumm", "origin": "cnndm"})
            elif fn == "val_sentence_pairs.json":
                for d in dataset:
                    aid = d["article_id"]
                    document = self.get_cnndm_document(aid)
                    clean_dataset.append({"filename": fn, "label": 1, "document": document, "claim": d["correct_sent"], "cnndm_id": aid, "annotations": [1], "dataset": "cogensumm", "origin": "cnndm"})
                    clean_dataset.append({"filename": fn, "label": 0, "document": document, "claim": d["incorrect_sent"], "cnndm_id": aid, "annotations": [0], "dataset": "cogensumm", "origin": "cnndm"})
        self.datasets.append({"name": "cogensumm", "dataset": clean_dataset})

    def load_xsumfaith(self):
        # On Faithfulness and Factuality in Abstractive Summarization - ACL 2020
        # https://github.com/google-research-datasets/xsum_hallucination_annotations
        # https://aclanthology.org/2020.acl-main.173.pdf

        dataset_folder = os.path.join(self.benchmark_folder, "xsumfaith/")
        if not os.path.exists(dataset_folder):
            print("==== XSum dataset not found, downloading from scratch")
            os.makedirs(dataset_folder)

            csv_file = requests.get("https://github.com/google-research-datasets/xsum_hallucination_annotations/raw/master/hallucination_annotations_xsum_summaries.csv")
            with open(os.path.join(dataset_folder, "hallucination_annotations_xsum_summaries.csv"), "wb") as f:
                f.write(csv_file.content)

        path_to_annotation = os.path.join(dataset_folder, "hallucination_annotations_xsum_summaries.csv")

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
            document = self.get_xsum_document(A["bbcid"])
            labels = [v["hallucination_type"] for v in vs]
            annotations = [1 if label == "NULL" else 0 for label in labels]
            most_common_label = Counter(labels).most_common(1)[0][0]
            label = 1 if most_common_label == "NULL" else 0
            c = "val" if len(clean_dataset) % 2 == 0 else "test"

            clean_dataset.append({"document": document, "claim": A["summary"], "bbcid": A["bbcid"], "model_name": A["system"], "label": label, "cut": c, "annotations": annotations, "dataset": "xsumfaith", "origin": "xsum"})
        final_dataset = [d for d in clean_dataset if d["cut"]==self.cut]
        self.datasets.append({"name": "xsumfaith", "dataset": final_dataset})

    def load_polytope(self, which_label="overall"):
        # What Have We Achieved on Text Summarization? [https://arxiv.org/abs/2010.04529]
        # Dataset must be downloaded from the Github repo: https://github.com/hddbang/polytope

        assert which_label in ["overall", "omission", "addition", "duplication", "inaccuracy"], "Unrecognized `which label`"

        dataset_folder = os.path.join(self.benchmark_folder, "polytope")
        if not os.path.exists(dataset_folder):
            print("==== Polytope dataset not found, downloading from scratch")
            os.makedirs(dataset_folder)

            for model_name in ["BART", "Bert_Ext", "Bert_Ext_Abs", "BottomUp", "PG", "PG_Coverage", "Summa", "TextRank", "seq2seq"]:
                url = "https://github.com/hddbang/PolyTope/raw/master/outputs_with_human_annotation/Human_Annotation_Summarization_%s.xlsm" % (model_name)
                r = requests.get(url)
                with open(os.path.join(dataset_folder, "Human_Annotation_Summarization_%s.xlsm" % (model_name)), "wb") as f:
                    f.write(r.content)

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
                d["dataset"] = "polytope"
                d["annotations"] = [d["label"]]
                d["origin"] = "cnndm"

                full_dataset.append(d)
        cut_dataset = [d for d in full_dataset if d["cut"]==self.cut]
        self.datasets.append({"name": "polytope", "dataset": cut_dataset})

    def load_factcc(self, max_entries=-1):
        # Evaluating the Factual Consistency of Abstractive Text Summarization [https://arxiv.org/abs/1910.12840]
        # Dataset for each split must be downloaded from the Github repo: https://github.com/salesforce/factCC

        dataset_folder = os.path.join(self.benchmark_folder, "factcc/")
        if not os.path.exists(dataset_folder):
            print("==== FactCC dataset not found, downloading from scratch")
            os.makedirs(dataset_folder)

            urls = ["https://storage.googleapis.com/sfr-factcc-data-research/unpaired_generated_data.tar.gz", "https://storage.googleapis.com/sfr-factcc-data-research/unpaired_annotated_data.tar.gz"]
            for url in urls:
                zip_name = url.split("/")[-1]
                r = requests.get(url)
                with open(os.path.join(dataset_folder, zip_name), "wb") as f:
                    f.write(r.content)
                
                with tarfile.open(os.path.join(dataset_folder, zip_name), "r:gz") as f:
                    f.extractall(dataset_folder)
                os.remove(os.path.join(dataset_folder, zip_name))

        if self.cut == "train":
            dataset = []
            with open(os.path.join(dataset_folder, "unpaired_generated_data/data-original/data-train.jsonl"), "r") as f:
                for i, line in enumerate(f):
                    if max_entries > 0 and i >= max_entries:
                        break
                    D = json.loads(line)
                    aid = D["filepath"].split("/")[-1].replace(".story", "")
                    full_text = self.get_cnndm_document(aid)

                    label = 1 if D["label"]=="CORRECT" else 0
                    datum = {"document": full_text, "claim": D["claim"], "cnndm_id": D["id"], "label": label, "dataset": "factcc", "origin": "cnndm"}
                    dataset.append(datum)

        if self.cut in ["val", "test"]:
            factcc_file = os.path.join(dataset_folder, "unpaired_annotated_data/%s/data-dev.jsonl" % (self.cut))
            dataset = []
            with open(factcc_file, "r") as f:
                for line in f:
                    dataset.append(json.loads(line))

            for d in dataset:
                aid = d["filepath"].split("/")[-1].replace(".story", "")
                d["document"] = self.get_cnndm_document(aid)
                d["label"] = 1 if d["label"] == "CORRECT" else 0
                d["annotations"] = [d["label"]]
                d["dataset"] = "factcc"
                d["origin"] = "cnndm"

        self.datasets.append({"name": "factcc", "dataset": dataset})

    def load_summeval(self, key_focus="consistency"):
        assert key_focus in ["consistency", "coherence", "fluency", "relevance"]
        # SummEval: Re-evaluating Summarization Evaluation [https://arxiv.org/abs/2007.12626]
        # Data files must be downloaded from the following Github repository: https://github.com/Yale-LILY/SummEval
        raw_dataset = []

        dataset_folder = os.path.join(self.benchmark_folder, "summeval/")
        fn = os.path.join(dataset_folder, "model_annotations.aligned.scored.jsonl")
        if not os.path.exists(dataset_folder):
            print("==== SummEval dataset not found, downloading from scratch")
            os.makedirs(dataset_folder)

            # From the 4/19/2020 update on the README: https://github.com/Yale-LILY/SummEval
            utils_misc.download_file_from_google_drive("1d2Iaz3jNraURP1i7CfTqPIj8REZMJ3tS", fn)

        with open(fn, "r") as f:
            for line in f:
                raw_dataset.append(json.loads(line))

        clean_dataset = []

        for i, d in enumerate(raw_dataset):
            c = "val" if i % 2 == 0 else "test"
            _, _, article_id = d["id"].split("-")
            document = self.get_cnndm_document(article_id)
            annotations = d["expert_annotations"]

            consistencies = [a[key_focus] for a in annotations]
            final_label = 1 if len([cons for cons in consistencies if cons==5]) > len(annotations)/2 else 0

            annotations = [1 if cons == 5 else 0 for cons in consistencies]
            error_type = "no error" if final_label == 1 else "error"

            clean_dataset.append({"document": document, "claim": d["decoded"], "label": final_label, "model_name": d["model_id"], "cnndm_id": d["id"], "cut": c, "annotations": annotations, "dataset": "summeval", "origin": "cnndm", "error_type": error_type})
        final_dataset = [d for d in clean_dataset if d["cut"] == self.cut]
        self.datasets.append({"name": "summeval", "dataset": final_dataset})

    def load_frank(self):
        # FRANK: Factuality Evaluation Benchmark [https://aclanthology.org/2021.naacl-main.383.pdf]
        # Files must be downloaded from the Github repository: https://github.com/artidoro/frank

        dataset_folder = os.path.join(self.benchmark_folder, "frank/")
        if not os.path.exists(dataset_folder):
            print("==== Frank dataset not found, downloading from scratch")
            os.makedirs(dataset_folder)

            fns = ["human_annotations_sentence.json", "validation_split.txt", "test_split.txt"]
            for fn in fns:
                data = requests.get("https://raw.githubusercontent.com/artidoro/frank/main/data/%s" % fn)
                with open(os.path.join(dataset_folder, fn), "w") as f:
                    f.write(data.text)

        raw_file = os.path.join(dataset_folder, "human_annotations_sentence.json")
        val_hash_file = os.path.join(dataset_folder, "validation_split.txt")
        test_hash_file = os.path.join(dataset_folder, "test_split.txt")
        with open(val_hash_file if self.cut=="val" else test_hash_file, "r") as f:
            valid_hashes = set([line.strip() for line in f])

        with open(raw_file, "r") as f:
            raw_dataset = json.load(f)
        dataset = []
        for d in raw_dataset:
            article = d["article"]
            origin = "cnndm" if len(d["hash"]) >= 40 else "xsum"

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

            error_type = "NoE"
            if label == 0:
                errors = [anno for annos in annotator_labels.values() for anno in annos if anno != "NoE"]
                error_type = Counter(errors).most_common(1)[0][0]

            summary = d["summary"]
            dataset.append({"document": article, "claim": summary, "label": label, "cut": self.cut, "hash": d["hash"], "model_name": d["model_name"], "annotations": annotations, "dataset": "frank", "origin": origin, "error_type": error_type})
        self.datasets.append({"name": "frank", "dataset": dataset})

    def get_dataset(self, dataset_name):
        for dataset in self.datasets:
            if dataset["name"] == dataset_name:
                return dataset["dataset"]
        raise ValueError("Unrecognized dataset name: %s" % (dataset_name))

    def print_stats(self):
        dataset_stats = []
        for dataset in self.datasets:
            N_pos, N_neg = len([d for d in dataset["dataset"] if d["label"]==1]), len([d for d in dataset["dataset"] if d["label"]==0])
            dataset_stats.append({"name": dataset["name"], "N": len(dataset["dataset"]), "N_pos": N_pos, "N_neg": N_neg, "frac_pos": N_pos/(N_pos+N_neg)})
        print(pd.DataFrame(dataset_stats))

    def evaluate(self, scorer):
        benchmark = []

        for dataset in self.datasets:
            dataset_labels = [d["label"] for d in dataset["dataset"]]
            dataset_preds = scorer.score([d["document"] for d in dataset["dataset"]], [d["claim"] for d in dataset["dataset"]])["scores"]

            dataset_thresh, dataset_f1 = utils_scorer.choose_best_threshold(dataset_labels, dataset_preds)
            benchmark.append({"name": dataset["name"], "score": dataset_f1, "threshold": dataset_thresh})
        return {"overall_score": np.mean([t["score"] for t in benchmark]), "benchmark": benchmark}


if __name__ == "__main__":
    import random

    for cut in ["val", "test"]:
        summac_benchmark = SummaCBenchmark(benchmark_folder="/home/tingu/data/summac_benchmark2/", cut=cut)
        print("============= SUMMAC %s ===============" % (cut.upper()))
        summac_benchmark.print_stats()
        for dataset in summac_benchmark.datasets:
            print("\n============= %s ===============" % (dataset["name"]))
            random.shuffle(dataset["dataset"])
            print(dataset["dataset"][0]["document"][:400])
            print("-------------")
            print(dataset["dataset"][0]["claim"])
            print("-------------")
            print(dataset["dataset"][0]["label"])
