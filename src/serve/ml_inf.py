import torch
from transformers import pipeline
from flask import Flask, request

from utils import preprocess_instance


app = Flask(__name__)


model_path = "Universal-NER/UniNER-7B-type"
max_new_tokens = 256
generator = pipeline('text-generation', model=model_path, torch_dtype=torch.float16, device=0)


@app.route('/process_data', methods=['GET'])
def process_data():
    param1 = request.args.get('param1')
    param2 = request.args.get('param2')

    if param1 is not None and param2 is not None:
        # Perform some processing with the parameters
        text = param1
        entity_type = param2

        example = {"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}
        prompt = preprocess_instance(example['conversations'])
        outputs = generator(prompt, max_length=max_new_tokens, return_full_text=False)
        print(outputs[0]['generated_text'])
        result = outputs[0]['generated_text']
    
        #result = f"Received parameters: param1={param1}, param2={param2}"

        return result
    else:
        return "Error: Missing parameters", 400



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    # main()




