from flask import Flask, request, jsonify,send_file
import pickle
from model_definition import ChittiQAModel

app = Flask(__name__)

# Load the model
try:
    with open('chitti_gold_qa_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)
    
@app.route('/')
def home():
    return send_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query']
        prediction = model.predict(query)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')