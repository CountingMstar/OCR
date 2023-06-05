from transformers import pipeline

# Load the pre-trained model for masked language modeling
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Define the input sentence with masked words
input_sentence = "I want to [MASK] NLP tasks using Python."

# Generate predictions for the masked word
predictions = fill_mask(input_sentence)

# Print the predicted words and their probabilities
for prediction in predictions:
    predicted_word = prediction["token_str"]
    probability = prediction["score"]
    print(f"Predicted word: {predicted_word}, Probability: {probability:.4f}")