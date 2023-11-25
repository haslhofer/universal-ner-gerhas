import fire
from vllm import LLM
from transformers import LlamaTokenizer
from .inference import inference

from flask import Flask, request





model_path = "Universal-NER/UniNER-7B-type"
max_new_tokens = 256
tensor_parallel_size = 1
max_input_length = 512
llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
tokenizer = LlamaTokenizer.from_pretrained(model_path)

app = Flask(__name__)

@app.route('/process_data', methods=['GET'])
def process_data():
    param1 = request.args.get('param1')
    param2 = request.args.get('param2')

    if param1 is not None and param2 is not None:
        text = param1
        entity_type = param2

        if len(tokenizer(text + entity_type)['input_ids']) > max_input_length:
            print(f"Error: Input is too long. Maximum number of tokens for input and entity type is {max_input_length} tokens.")
    
        examples = [{"conversations": [{"from": "human", "value": f"Text: {text}"}, {"from": "gpt", "value": "I've read this text."}, {"from": "human", "value": f"What describes {entity_type} in the text?"}, {"from": "gpt", "value": "[]"}]}]
        output = inference(llm, examples, max_new_tokens=max_new_tokens)[0]
        print(output)
        # Perform some processing with the parameters
        result = output

        return result
    else:
        return "Error: Missing parameters", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
