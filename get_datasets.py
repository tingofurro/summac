from summac.benchmark import SummaCBenchmark
benchmark_val = SummaCBenchmark(benchmark_folder="data/", cut="val")
# for dataset in benchmark_val.datasets:
#     print(dataset["name"])


"""
benchmark_val.get_dataset("cogensumm") chose from cogensumm xsumfaith polytope factcc summeval frank
"""

cogensumm = benchmark_val.get_dataset("cogensumm") 
print(cogensumm[300].keys()) 
#dict_keys(['filename', 'label', 'document', 'claim', 
# 'cnndm_id', 'annotations', 'dataset', 'origin'])
print(cogensumm[300])

xsumfaith = benchmark_val.get_dataset("xsumfaith") 
print(xsumfaith[300].keys())
# dict_keys(['document', 'claim', 'bbcid', 
# 'model_name', 'label', 'cut', 'annotations', 'dataset', 'origin'])
print(xsumfaith[300])

# {"document: "A Darwin woman has become a TV [...]", 
# "claim": "natalia moon , 23 , has become a tv sensation [...]", 
# "label": 0, "cut": "val", "model_name": "s2s", "error_type": "LinkE"}