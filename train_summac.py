import utils_misc

utils_misc.select_freer_gpu()
import torch, tqdm, nltk, numpy as np, argparse, json
from torch.utils.data import DataLoader, RandomSampler
import utils_optim, os, time
from utils_summac_benchmark import SummaCBenchmark, load_factcc
from model_summac import SummaCConv, model_map

def train(model="mnli", granularity="sentence", nli_labels="e", pre_file="", num_epochs=5, optimizer="adam", train_batch_size=32, learning_rate=0.1, bins="even50", silent=False, norm_histo=False):
    experiment = "%s_%s_%s_%s" % (model, granularity, bins, nli_labels)

    if not silent:
        print("Experiment name: %s" % (experiment))

    if len(pre_file) == 0:
        standard_pre_file = "/home/phillab/data/summac_cache/train_%s_%s.jsonl" % (model, granularity)
        if os.path.isfile(standard_pre_file):
            pre_file = standard_pre_file

    precomputed = len(pre_file) > 0
    device = "cpu" if precomputed else "cuda"

    if model == "multi":
        models = ["mnli", "anli", "vitc"]
    elif model == "multi2":
        models = ["mnli", "vitc", "vitc-only", "vitc-base"]
    else:
        models = [model]

    model = SummaCConv(models=models, granularity=granularity, nli_labels=nli_labels, device=device, bins=bins, norm_histo=norm_histo)

    optimizer = utils_optim.build_optimizer(model, learning_rate=learning_rate, optimizer_name=optimizer)
    if not silent:
        print("Model Loaded")

    def sent_tok(text):
        sentences = nltk.tokenize.sent_tokenize(text)
        return [sent for sent in sentences if len(sent)>10]

    def collate_func(inps):
        documents, claims, labels = [], [], []
        for inp in inps:
            if len(sent_tok(inp["claim"])) > 0 and len(sent_tok(inp["document"])) > 0:
                documents.append(inp["document"])
                claims.append(inp["claim"])
                labels.append(inp["label"])
        labels = torch.LongTensor(labels).to(device)
        return documents, claims, labels

    def collate_pre(inps):
        documents = [inp["document"] for inp in inps]
        claims = [inp["claim"] for inp in inps]
        # images = [[np.array(im) for im in inp["image"]] for inp in inps]
        images = [np.array(inp["image"]) for inp in inps]
        labels = torch.LongTensor([inp["label"] for inp in inps]).to(device)
        return documents, claims, images, labels

    if precomputed:
        d_train = []
        with open(pre_file, "r") as f:
            for line in f:
                d_train.append(json.loads(line))
        dl_train = DataLoader(dataset=d_train, batch_size=train_batch_size, sampler=RandomSampler(d_train), collate_fn=collate_pre)
    else:
        d_train = load_factcc(cut="train")
        dl_train = DataLoader(dataset=d_train, batch_size=train_batch_size, sampler=RandomSampler(d_train), collate_fn=collate_func)

    fcb = SummaCBenchmark(cut="val")

    if not silent:
        print("Length of dataset. [Training: %d]" % (len(d_train)))

    crit = torch.nn.CrossEntropyLoss()
    eval_every = 200
    best_val_score = 0.0
    best_file = ""

    for epi in range(num_epochs):
        ite = enumerate(dl_train)
        if not silent:
            ite = tqdm.tqdm(ite, total=len(dl_train))
        for ib, batch in ite:
            if precomputed:
                documents, claims, images, batch_labels = batch
                logits, _, _ = model(documents, claims, images=images)
            else:
                documents, claims, batch_labels = batch
                logits, _, _ = model(originals=documents, generateds=claims)
            loss = crit(logits, batch_labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # wandb.log({"loss": loss.item()})

            if ib % eval_every == eval_every-1:

                eval_time = time.time()
                benchmark = fcb.evaluate(model)
                val_score = benchmark["overall_score"]
                eval_time = time.time() - eval_time
                if eval_time > 10.0:
                    model.save_imager_cache()

                if not silent:
                    ite.set_description("[Benchmark Score: %.3f]" % (val_score))
                if val_score > best_val_score:
                    best_val_score = val_score
                    if len(best_file) > 0:
                        os.remove(best_file)
                    best_file = "/home/phillab/models/summac/%s_bacc%.3f.bin" % (experiment, best_val_score)
                    torch.save(model.state_dict(), best_file)
                    if not silent:
                        for t in benchmark["benchmark"]:
                            print("[%s] Score: %.3f (thresh: %.3f)" % (t["name"].ljust(10), t["score"], t["threshold"]))
    return best_val_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    model_choices = list(model_map.keys()) + ["multi", "multi2"]

    parser.add_argument("--model", type=str, choices=model_choices, default="mnli")
    parser.add_argument("--granularity", type=str, default="sentence") # , choices=["sentence", "paragraph", "mixed", "2sents"]
    parser.add_argument("--pre_file", type=str, default="", help="If not empty, will use the precomputed instead of computing images on the fly. (useful for hyper-param tuning)")
    parser.add_argument("--bins", type=str, default="percentile", help="How should the bins of the histograms be decided (even%d or percentile)")
    parser.add_argument("--nli_labels", type=str, default="e", choices=["e", "c", "n", "ec", "en", "cn", "ecn"], help="Which of the three labels should be used in the creation of the histogram")

    parser.add_argument("--num_epochs", type=int, default=5, help="Number of passes over the data.")
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam")
    parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Number of passes over the data.")
    parser.add_argument("--norm_histo", action="store_true", help="Normalize the histogram to be between 0 and 1, and include the explicit count")

    args = parser.parse_args()
    train(**args.__dict__)
