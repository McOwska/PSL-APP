import random

def random_action():
    actions = ["Hello!", "Welcome!", "Smile!", "Keep Going!", "Good Luck!"]
    return random.choice(actions)

def random_sentence():
    a = random.randint(1,2)
    if a==1:
        return "Losowe zdanie Jeden"
    else:
        return "Losowe zdanie 2"
    