from model_generator import Generator
from model_summac import SummaCConv
import nltk

model = Generator(model_card="gpt2-medium", device="cuda")
model.reload("/home/phillab/models/ACL2021/gpt2_med_keep_it_simple.bin")

summac = SummaCConv(models=["vitc"], bins="percentile", start_file="/home/phillab/models/summac/vitc_sentence_percentile_e_bacc0.746.bin")

document = """Jeff joined Microsoft in 1992 to lead corporate developer evangelism for Windows NT. He then served as a Group Program manager in Microsoft's Internet Business Unit.
In 1998, he led the creation of SharePoint Portal Server, which became one of Microsoft's fastest-growing businesses, exceeding $2 billion in revenues.
Jeff next served as Corporate Vice President for Program Management across Office 365 Services and Servers, which is the foundation of Microsoft's enterprise cloud leadership.
He then led Corporate Strategy supporting Satya Nadella and Amy Hood on Microsoft's mobile-first/cloud-first transformation and acquisitions.""" #  Prior to joining Microsoft, Jeff was vice president for software development for an investment firm in New York. He leads Office shared experiences and core applications, as well as OneDrive and SharePoint consumer and business services in Office 365. Jeff holds a Master of Business Administration degree from Harvard Business School and a Bachelor of Science degree in information systems and finance from New York University.

paul_summary = "Jeff joined Microsoft in 1992 to lead the company's corporate evangelism. He then served as a Group Manager in Microsoft's Internet Business Unit. In 1998, Jeff led Sharepoint Portal Server, which became the company's fastest-growing business, surpassing $3 million in revenue. Jeff next leads corporate strategy for SharePoint and Servers which is the basis of Microsoft's cloud-first strategy. He leads corporate strategy for Satya Nadella and Amy Hood on Microsoft's mobile-first."
generateds = model.generate([document], num_runs=16)[0]

generateds.append({"output_text": paul_summary})
print("=============== STANDARD GENERATION  ================")

for candidate in generateds:
    summac_score = summac.score(originals=[document], generateds=[candidate["output_text"]])["scores"][0]
    print("-----")
    print("[SummaC Score: %.3f] %s" % (summac_score, candidate["output_text"]))

print("=============== ONE SENTENCE AT A TIME  ================")

build_up = ""
for i in range(4):
    print("-------------- %d -------------" % (i))
    generateds = model.generate([document], force_start=build_up, num_runs=16)[0]
    candidate_sentences = []
    for candidate in generateds:
        candidate_sentences.append(nltk.tokenize.sent_tokenize(candidate["output_text"]))

    next_sentences = [cand_sent[i] for cand_sent in candidate_sentences if len(cand_sent) >= i+1]
    if len(next_sentences) == 0:
        break
    next_scores = summac.score([document] * len(next_sentences), next_sentences)["scores"]

    next_sents = [{"sentence": sent, "score": score} for sent, score in zip(next_sentences, next_scores)]
    next_sents = sorted(next_sents, key=lambda x: -x["score"])

    for next in next_sents:
        print("---------")
        print("[%.3f] %s" % (next["score"], next["sentence"]))

    build_up += " "+next_sents[0]["sentence"]
    build_up = build_up.strip()

print("--------------------------")
print("Final text:")
summac_score = summac.score(originals=[document], generateds=[build_up])["scores"][0]

print("[SummaC Score: %.3f] %s" % (summac_score, build_up))