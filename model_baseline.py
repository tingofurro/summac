import datasets, nltk, numpy as np
import json, os, sys, argparse

class BaselineScorer:
    def __init__(self, model="questeval", do_weighter=False, load_cache=True):
        assert model in ["questeval", "feqa", "dae"], "Unrecognized baseline model"
        self.model = model
        self.do_weighter = do_weighter
        self.model_loaded = False
        self.cache = {}
        self.cache_file = "/export/share/plaban/summac_cache/cache_%s.json" % (self.model)
        if load_cache:
            self.load_cache()

    def load_model(self):
        if self.model == "questeval":
            from questeval.questeval_metric import QuestEval
            self.questeval = QuestEval(isCuda=True, do_weighter=self.do_weighter)
        elif self.model == "feqa":
            # import benepar, nltk
            # benepar.download('benepar_en2')
            # nltk.download('stopwords')
            from feqa import FEQA
            self.scorer = FEQA(use_gpu=True)
        elif self.model == "dae":
            sys.path.insert(0, "/home/phillab/dae-factuality/")
            from evaluate_factuality import MODEL_CLASSES, score_example_single_context

            parser = argparse.ArgumentParser()
            args = parser.parse_args()
            args.device = "cuda:0"
            args.per_gpu_eval_batch_size = 8
            args.max_seq_length = 128
            args.dependency_type =  "enhancedDependencies"
            self.args = args

            model_dir = "/home/phillab/models/dae_basic/"
            model_type = "electra_dae"
            config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]

            self.tokenizer = tokenizer_class.from_pretrained(model_dir)
            self.dae_model = model_class.from_pretrained(model_dir)
            self.dae_model.to(args.device)

        self.model_loaded = True

    def load_cache(self):
        if os.path.isfile(self.cache_file):
            with open(self.cache_file, "r") as f:
                self.cache = json.load(f)

    def save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f)

    def get_sample_key(self, document, generated):
        return "%s|%%%%|%%|%s" % (document, generated)

    def score_questeval(self, documents, generateds, **kwargs):
        scores = []
        for document, generated in zip(documents, generateds):
            score = self.questeval.compute_all(document, generated)
            scores.append(score["scores"]["fscore"])
        return {"scores": scores}

    def score_feqa(self, documents, generateds, **kwargs):
        scores = self.scorer.compute_score(documents, generateds, aggregate=False)
        self.save_cache()
        return {"scores": scores}

    def score_dae(self, documents, generateds, **kwargs):
        from evaluate_factuality import score_example_single_context

        scores = []
        for document, generated in zip(documents, generateds):
            document = " ".join(document.split(" ")[:250])
            score = score_example_single_context(generated, document, self.dae_model, self.tokenizer, self.args).item()
            scores.append(score)

        # self.save_cache()
        return {"scores": scores}

    def score(self, documents, generateds, **kwargs):
        new_samples = []
        for d, g in zip(documents, generateds):
            k = self.get_sample_key(d, g)
            if k not in self.cache:
                new_samples.append((k, d, g))

        if len(new_samples) > 0:
            if not self.model_loaded:
                self.load_model()

            if self.model == "questeval":
                new_scores = self.score_questeval([d[1] for d in new_samples], [d[2] for d in new_samples])
            elif self.model == "feqa":
                new_scores = self.score_feqa([d[1] for d in new_samples], [d[2] for d in new_samples])
            elif self.model == "dae":
                new_scores = self.score_dae([d[1] for d in new_samples], [d[2] for d in new_samples])

            for (k, d, g), score in zip(new_samples, new_scores["scores"]):
                self.cache[k] = score

        return {"scores": [self.cache[self.get_sample_key(d, g)] for d, g in zip(documents, generateds)]}


if __name__ == "__main__":
    hypothesis = """After wildfires consumed an entire town, students and teachers who had planned for remote classes found some comfort in staying connected amid the chaos."""

    source = """Ash fell from an apocalyptic orange sky as Jennifer Willin drove home last week from the only school in tiny Berry Creek, Calif., where she had picked up a pair of Wi-Fi hot spots for her daughters’ remote classes. Hours later,
    her cellphone erupted with an emergency alert: Evacuate immediately. By the next morning, what one official described as a “massive wall of fire” had swept through the entire Northern California town of about 1,200 people, killing nine residents,
    including a 16-year-old boy, and destroying the school and almost every home and business. Ms. Willin and her family escaped to a cramped hotel room 60 miles away.
    In her panic, she had forgotten to grab masks, but she had the hot spots, along with her daughters’ laptops and school books. On Monday, the two girls plan to meet with their teachers on Zoom, seeking some comfort amid the chaos.
    They’re still able to be in school,” Ms. Willin said, “even though the school burned to the ground.”
    As the worst wildfire season in decades scorches the West amid a still raging pandemic, families and educators who were already starting the strangest and most challenging school year of their lifetimes have been traumatized all over again.
    Tens of thousands of people have been forced to flee their homes, with some mourning the loss of their entire communities.
    But amid the twin disasters, the remote learning preparations that schools made for the coronavirus are providing a strange
    modicum of stability for teachers and students, letting many stay connected and take comfort in an unexpected form of virtual community."""

    qe_score = BaselineScorer(model="dae")
    print(qe_score.score([source], [hypothesis]))
