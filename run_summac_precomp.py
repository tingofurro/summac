import utils_misc

utils_misc.select_freer_gpu()

from model_summac import SummaCConv, model_map
import utils_summac_benchmark
import argparse, json, tqdm, nltk

model_choices = list(model_map.keys()) + ["multi", "multi2"]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=model_choices, default="mnli")
parser.add_argument("--granularity", type=str, default="sentence") # , choices=["sentence", "paragraph", "mixed", "2sents"]
args = parser.parse_args()

def sent_tok(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    return [sent for sent in sentences if len(sent)>10]


if args.model == "multi":
    models = ["mnli", "anli", "vitc"]
elif args.model == "multi2":
    models = ["mnli", "vitc", "vitc-only", "vitc-base"]
else:
    models = [args.model]

model = SummaCConv(models=models, granularity=args.granularity)

dataset_fn = utils_misc.unique_file("/home/phillab/data/summac_cache/train_%s_%s.jsonl" % (args.model, args.granularity))
print(">> Will write to file: %s" % (dataset_fn))

d_train = utils_summac_benchmark.load_factcc(cut="train", max_entries=20000)
print("Dataset loaded")
for d in tqdm.tqdm(d_train):
    if len(sent_tok(d["document"])) == 0 or len(sent_tok(d["claim"])) == 0:
        continue
    d["image"] = model.build_image(d["document"], d["claim"])
    d["image"] = [img.tolist() for img in d["image"]]

    with open(dataset_fn, "a") as f:
        f.write(json.dumps(d)+"\n")
