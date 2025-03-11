from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model
model = load_model('best_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define parameters
max_sequence_len = 30  # Adjust as per your training settings

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the partial sentence from the request
    data = request.json
    partial_sentence = data['sentence']

    # Convert the partial sentence to a sequence of indices
    sequence = tokenizer.texts_to_sequences([partial_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_len - 1)

    # Get predictions
    predicted_probs = model.predict(padded_sequence)
    predicted_word_index = np.argmax(predicted_probs, axis=-1)
    
    # Map index to word
    predicted_word = tokenizer.index_word[predicted_word_index[0]]
    
    return jsonify({'suggestion': predicted_word})

if __name__ == '__main__':
    app.run(debug=True)
