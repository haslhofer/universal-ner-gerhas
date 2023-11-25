from flask import Flask, request


app = Flask(__name__)

@app.route('/process_data', methods=['GET'])
def process_data():
    param1 = request.args.get('param1')
    param2 = request.args.get('param2')

    if param1 is not None and param2 is not None:

        # Perform some processing with the parameters
        result = f"Received parameters: param1={param1}, param2={param2}"

        return result
    else:
        return "Error: Missing parameters", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
