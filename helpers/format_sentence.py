def format_sentence(predictions):
    unique_predictions = []
    previous_pred = None

    for pred in predictions:
        if pred != previous_pred:
            unique_predictions.append(pred)
        previous_pred = pred
    
    sentence = " ".join(unique_predictions)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:] + "."
    return sentence
