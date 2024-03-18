from llama_cpp import Llama
print("loading llama model....")
llm = Llama(model_path = "./models/BioMistral-7B.Q4_K_M.gguf" )
print('Model loaded')