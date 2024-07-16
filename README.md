#### Telugu LLM - Question and Answering System    

##### Introduction
This project utilizes the Telugu-Llama2-7B-v0-Instruct model from the Telugu-LLM-Labs to provide answers to questions in Telugu. The model is fine-tuned for causal language modeling tasks and can understand and generate Telugu text. 

##### Installation
To use the code, you need to install the required packages. This project requires Python and several libraries which can be installed using pip. 
1.pip install torch
2.pip install transformers  

##### Model Details
We are using the Telugu-Llama2-7B-v0-Instruct model for this project. This model is pre-trained for instruction-based tasks and fine-tuned to understand and generate Telugu language text.

##### Usage
The following code demonstrates how to load the model and tokenizer, and how to generate a response for a given question and context.

Load the Model and Tokenizer 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 

model_name = "Telugu-LLM-Labs/Telugu-Llama2-7B-v0-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)  

##### Provide Instruction and Input Text 
instruction = "vijay devarakonda hit movies emiti, vaati perlu enti?"
input_text = "విజయ్ దేవరకొండ ఇప్పటి వరకు పంతొమ్మిది సినిమాలు చేసాడు.అందులో పక్కా హీరోగా చేసిన చిత్రాలు పన్నెండు.హిట్ అయిన సినిమాలు పెళ్లి చూపులు, అర్జున్ రెడ్డి, గీత గోవిందం,టాక్సీ వాలా లేటెస్ట్ గా ఫ్యామిలీ స్టార్ తో డిజాస్టర్ ని అందుకున్నాడు.  దీంతో ఇప్పుడు తనకి అర్జెంట్ గా ఒక హిట్ కావాలి. ఈ నేపథ్యంలో వస్తున్న రెండు వార్తలు విజయ్ ఫ్యాన్స్ లో పండగ వాతావరణాన్ని తెస్తున్నాయి. పైగా ఈ సారి హిట్ ఖాయమని కూడా అంటున్నారు"
prompt = f"Instruction: {instruction}\nInput: {input_text}\nResponse:"   

##### Tokenize and Generate Response 

encodings = tokenizer(prompt, padding=True, return_tensors="pt")
with torch.inference_mode():
    outputs = model.generate(encodings.input_ids, do_sample=True, max_new_tokens=500, temperature=0.5, top_p=0.5)
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#Extract the part after "Response:" to get only the response
response = output_text.split("Response:")[-1].strip()
print(response) 

##### Expected Output
హిట్ అయిన సినిమాలు పెళ్లి చూపులు, అర్జున్ రెడ్డి, గీత గోవిందం, టాక్సీ వాలా. 

Pretrained model link: https://huggingface.co/Telugu-LLM-Labs/Telugu-Llama2-7B-v0-Instruct
