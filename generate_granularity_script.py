grans = ["paragraph-document", "paragraph-sentence", "2sents-document",  "2sents-sentence", "sentence-document"]
for model in ["mnli", "vitc"]:
    print("")
    for granularity in grans:
        print("python run_summac_precomp.py --model %s --granularity %s" % (model, granularity))

print("")
for model in ["mnli", "vitc"]:
    print("")
    for granularity in grans:
        print("python train_summac.py --model %s --granularity %s" % (model, granularity))
