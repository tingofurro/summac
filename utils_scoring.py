import torch, time, numpy as np
import utils_misc

class ScorerWrapper:
    def __init__(self, scorers, scoring_method="logsum", max_batch_size=100, use_caching=False):
        assert scoring_method in ["sum", "product", "logsum"], "Unrecognized `scoring_method`"

        self.scorers = scorers
        self.scoring_method = scoring_method
        self.use_caching = use_caching
        self.cache = {}

        self.max_batch_size = max_batch_size
        if self.scoring_method == "logsum":
            self.score_func = logsum_score
        elif self.scoring_method == "product":
            self.score_func = product_score
        elif self.scoring_method == "sum":
            self.score_func = sum_score

    def get_score_names(self):
        return [s["name"] for s in self.scorers]

    def make_key(self, inp, gen):
        return "%s|||___|||%s" % (inp, gen)

    def score(self, inputs, generateds, partial=False, printing=False, timings=False, extras={}, progress=False):
        assert len(inputs) == len(generateds), "Input and output lengths don't match"

        if not self.use_caching:
            self.cache = {} # Reset the cache

        todo = []
        all_keys = []
        for inp, gen in zip(inputs, generateds):
            key = self.make_key(inp, gen)
            all_keys.append(key)
            if key not in self.cache:
                todo.append({"inp": inp, "gen": gen, "key": key})

        for d in todo:
            self.cache[d["key"]] = {}

        if self.use_caching and len(todo) < len(all_keys):
            print("With caching, only processing: %d / %d samples" % (len(todo), len(all_keys)))

        if len(todo) == 0:
            progress = False # Not needed, it's empty

        for batch_todo in utils_misc.batcher(todo, batch_size=self.max_batch_size, progress=progress):
            batch_inputs = [d["inp"] for d in batch_todo]
            batch_gens = [d["gen"] for d in batch_todo]

            batch_scores, timings_out = self.score_func(self.scorers, batch_inputs, batch_gens, partial=partial, printing=printing, extras=extras)

            for k, out in batch_scores.items():
                if type(out) in [torch.Tensor, np.array, np.ndarray]:
                    out = out.tolist()

                for i, d in enumerate(batch_todo):
                    self.cache[d["key"]][k] = out[i]

            if timings:
                print(timings_out)

        all_outputs = {}
        for k in self.cache[all_keys[0]].keys():
            all_outputs[k] = [self.cache[key][k] for key in all_keys]

        if printing:
            print("[total]", all_outputs["total_scores"])
        return all_outputs

    def __call__(self, inputs, generateds, **kwargs):
        return self.score(inputs, generateds, **kwargs)

def sum_score(scorers, paragraphs, generateds, partial=False, printing=False, extras={}):
    total_scores = np.zeros((len(paragraphs)))
    scorer_returns, timings = {}, {}
    T = time.time()

    for scorer in scorers:
        scores = scorer['model'].score(paragraphs, generateds, partial=partial, printing=printing, **extras)
        weight = scorer.get("weight", 1.0)
        total_scores += scorer["sign"]*weight*np.array(scores['scores'])

        scorer_returns.update({scorer['name']+"_"+k: v for k, v in scores.items()})
        timings[scorer["name"]] = time.time()-T
        T = time.time()

    scorer_returns['total_scores'] = total_scores
    return scorer_returns, timings

def product_score(scorers, paragraphs, generateds, partial=False, printing=False, extras={}):
    total_scores = np.ones((len(paragraphs)))
    scorer_returns, timings = {}, {}
    T = time.time()

    for scorer in scorers:
        scores = scorer['model'].score(paragraphs, generateds, partial=partial, printing=printing, **extras)
        if scorer['sign'] == 1:
            total_scores *= np.array(scores['scores'])
        else: # It's a binary penalty
            total_scores *= (1-np.array(scores['scores']))

        scorer_returns.update({scorer['name']+"_"+k: v for k, v in scores.items()})
        timings[scorer["name"]] = time.time()-T
        T = time.time()

    scorer_returns['total_scores'] = total_scores
    return scorer_returns, timings

def logsum_score(scorers, paragraphs, generateds, partial=False, printing=False, extras={}):
    total_scores = np.zeros((len(paragraphs)))
    scorer_returns, timings = {}, {}
    T = time.time()

    for scorer in scorers:
        scores = scorer['model'].score(paragraphs, generateds, partial=partial, printing=printing, **extras)
        weight = scorer.get("weight", 1.0)
        scores["scores"] = np.clip(scores["scores"], 0.0001, 0.9999)
        if scorer['sign'] == 1:
            total_scores += weight*np.log(np.array(scores['scores']))
        else: # It's a binary penalty
            total_scores += np.log(1-np.array(scores["scores"]))

        scorer_returns.update({scorer['name']+"_"+k: v for k, v in scores.items()})
        timings[scorer["name"]] = time.time()-T
        T = time.time()

    scorer_returns['total_scores'] = total_scores
    return scorer_returns, timings
