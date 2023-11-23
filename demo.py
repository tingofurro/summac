import torch
from summac.model_summac import SummaCZS, SummaCConv

model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu") # If you have a GPU: switch to: device="cuda"
model_conv = SummaCConv(models=["vitc"], bins='percentile', granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")

document = """horrific cctv footage has emerged of a husband smashing his dancer wife 's head into concrete paving 11 times then shooting her in the face five times .\nhe later confessed that he 'd carried out the brutal act ` because he was jealous of men looking at her ' .\nsickening video captured on security cameras shows crazed milton vieira severiano , 32 , dragging the limp body of dancer cicera alves de sena , 29 , into the garden of their apartment in the city of rio de janeiro in eastern brazil .\nhorrific cctv footage has emerged of a husband smashing his dancer wife 's head into concrete paving 11 times then shooting her in the face five times\nmilton vieira severiano confessed that he battered and shot his wife because he was jealous of men looking at her\ndancer cicera alves de sena ( pictured ) was killed by her husband after he 'd yelled at her that she was ` a slut '\ncicera alves de sena belonged to a dance troop called jaula das gostozuedas\nhe then dumps her on the ground before grabbing her by the hair and savagely slamming the back of her head into the concrete 11 times before viciously punching her in the face another 10 times .\nleaving the battered and unconscious dancer on the ground he goes into the house and returns with a handgun , shooting her in the face five times at pointblank range .\nhorrified neighbour lelio maya puga , 33 , said : ` they had only been married four days when i heard a terrible argument coming from their apartment .\n` she belonged to a dance group called jaula das gostozuedas which i have been to see and am a big fan of .\n` he was yelling and accusing her of being a slut and a whore and screaming that she would be better off dead .\n` i knew something terrible was happening and then things went silent until i heard the gunshots . '\nafter shooting his wife , who went by the stage name of amanda bueno , severiano stole a neighbour 's car but was picked up by police after crashing it during his getaway .\na police spokesman said : ` in his car we found four handguns , a shotgun and a bullet proof vest .\n` he confessed to murdering his wife and said he had been jealous of men looking at her . '\nhe now faces life in jail .\ncrime scene : people gather outside the couple 's house after the shocking incident took place\npolice found four handguns , a shotgun and a bullet proof vest in severiano 's getaway car",
        "claim": "sickening video captured on security cameras shows crazed milton vieira severiano dragging the limp body of dancer cicera alves de sena into the garden of their apartment in the city of rio de janeiro in eastern brazil . \nhe then grabs her by the hair and savagely slams the back of her head into the concrete 11 times before viciously punching her in the face another 10 times . \nleaving the battered and unconscious dancer on the ground he goes into the house and returns with a handgun , shooting her five times at pointblank range . \nseveriano later confessed that he 'd carried out the brutal act ` because he was jealous of men looking at"""

summary1 = "- Milt Vieira Severiano, a Brazilian dancer, was killed by her husband after he yelled at her that she was a slut.\n        - He confessed to murdering her and said he was jealous of men looking at her.\n        - He now faces life in jail.\n        - Police found four handguns, a shotgun, and a bullet proof vest in his car."
score_zs1 = model_zs.score([document], [summary1])
score_conv1 = model_conv.score([document], [summary1])
print("[Summary 1] SummaCZS Score: %.3f; SummacConv score: %.3f" % (score_zs1["scores"][0], score_conv1["scores"][0])) # [Summary 1] SummaCZS Score: 0.582; SummacConv score: 0.536



summary2 = "There are strange shape patterns on Arcadia Planitia. The shapes could indicate the area might be made of glaciers."
score_zs2 = model_zs.score([document], [summary2])
score_conv2 = model_conv.score([document], [summary2])
print("[Summary 2] SummaCZS Score: %.3f; SummacConv score: %.3f" % (score_zs2["scores"][0], score_conv2["scores"][0])) # [Summary 2] SummaCZS Score: 0.877; SummacConv score: 0.709



quit()
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", cache_dir="llm_weights")


def get_model_tokenzier(model_name, prune_method="fullmodel"):
    if prune_method == "fullmodel":
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="llm_weights", trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True, cache_dir = "llm_weights")
    else:  
        short_name = str(model_name).split("/")[-1]
        model_name = f'pruned_model/{short_name}/{args.prune_method}'
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, tokenizer


text = "Paul Merson has restarted his row with Andros Townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team's 0-0 draw with Burnley on Sunday. 'Just been watching the game, did you miss the coach? #RubberDub #7minutes,' Merson put on Twitter. Merson initially angered Townsend for writing in his Sky Sports column that 'if Andros Townsend can get in (the England team) then it opens it up to anybody.' Paul Merson had another dig at Andros Townsend after his appearance for Tottenham against Burnley . Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley . Andros Townsend scores England's equaliser in their 1-1 friendly draw with Italy in Turin on Tuesday night . The former Arsenal man was proven wrong when Townsend hit a stunning equaliser for England against Italy and he duly admitted his mistake. 'It's not as though I was watching hoping he wouldn't score for England, I'm genuinely pleased for him and fair play to him \\u2013 it was a great goal,' Merson said. 'It's just a matter of opinion, and my opinion was that he got pulled off after half an hour at Manchester United in front of Roy Hodgson, so he shouldn't have been in the squad. 'When I'm wrong, I hold my hands up. I don't have a problem with doing that - I'll always be the first to admit when I'm wrong.' Townsend hit back at Merson on Twitter after scoring for England against Italy . Sky Sports pundit  Merson (centre) criticised Townsend's call-up to the England squad last week . Townsend hit back at Merson after netting for England in Turin on Wednesday, saying 'Not bad for a player that should be 'nowhere near the squad' ay @PaulMerse?' Any bad feeling between the pair seemed to have passed but Merson was unable to resist having another dig at Townsend after Tottenham drew at Turf Moor"

prompt_id="C"
document = generate_prompt(prompt_id, text)

print(f"==>> document: {document}")

def demo(model_name, document):


    print(' ')
    print("".center(50, "-"))
    print("".center(50, "-"))
    print(model_name)
    print("".center(50, "-"))
    
    model, tokenizer = get_model_tokenzier(model_name)
    original_len = len(tokenizer.encode(document, return_tensors="pt")[0])
    print(f"==>> original_len: \n {original_len}")
    generate_max_new_tokens = int(original_len*0.25)
    input_ids = tokenizer.encode(document, return_tensors="pt") 
    output = model.generate(input_ids.to(model.device), num_return_sequences=1,
                                    max_new_tokens=generate_max_new_tokens, 
                                    #device = "auto",
                                    )   # including one special token, origi len + 1
    output_text = tokenizer.decode(output[0][int(input_ids.shape[1]):], skip_special_tokens=True)
    print(f"==>> output: \n {output_text}")
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"==>> full output: \n {output_text}")
        
demo("llama-2-7b", document)
demo("NousResearch/Llama-2-7b-hf", document)