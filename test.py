import json
import os

from transformers import AutoTokenizer, AutoModelForCausalLM


model = AutoModelForCausalLM.from_pretrained("facebook/opt-iml-1.3b", cache_dir="llm_weights", trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
print("".center(50, "-"))
print("".center(50, "-"))
print(model.base_model.decoder.layers)



model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", cache_dir="llm_weights", trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
print(len(model.transformer.h))
print("222".center(50, "-"))
try: print(model.model.layers)
except: print(' pass 2222')

model = AutoModelForCausalLM.from_pretrained("NousResearch/Nous-Hermes-llama-2-7b", cache_dir="llm_weights", trust_remote_code=True, device_map="auto") # torch_dtype=torch.float16, low_cpu_mem_usage=True, 
try: print(len(model.model.layers))
except: pass
print(" llama model layer list".center(50, "-"))

quit()

save_path = os.path.join("generated_output", "Nous-Hermes-llama-2-7b", "wanda", "xsumfaith")


with open(save_path + f"/norepeated_result_promptNone.json", "r+") as json_file:
    generate_dict = json.load(json_file)
    print(generate_dict)

# Define the new dictionary to append
new_data = {"name": "Charlie", "age": 35}

# Open the existing JSON file for reading and appending
with open('test.json', 'r+') as json_file:
    # Load the existing data from the file
    data = json.load(json_file)

    # Append the new dictionary to the list
    data.append(new_data)

    # Move the file pointer to the beginning to overwrite the file
    json_file.seek(0)

    # Write the updated list of dictionaries back to the file
    json.dump(data, json_file, indent=4)

    # Truncate any remaining data in the file (if the new data is shorter)
    json_file.truncate()

# Close the file when you're done
json_file.close()


# # Open a file for writing in append mode ('a')
# with open('test.json', 'a') as json_file:
#     for i in range(5):  # Replace 5 with the number of iterations you need
#         result = [{"iteration": i, "data": i * 2}]  # Example data to be written

#         # Write the result as a JSON object to the file
#         json.dump(result, json_file)
#         json_file.write('\n')  # Add a newline to separate objects

# # Close the file when you're done
# json_file.close()

