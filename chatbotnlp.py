import random
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

intents = {
    'greeting': ['hello', 'hi', 'hey'],
    'goodbye': ['bye', 'see you', 'take care'],
    'thanks': ['thank you', 'thanks a lot', 'thanks'],
    'options': ['what can you do?', 'what are your features?', 'features']
}

responses = {
    'greeting': ['Hello!', 'Hi there!', 'Hey!'],
    'goodbye': ['Goodbye!', 'See you!', 'Take care!'],
    'thanks': ['You\'re welcome!', 'No problem!', 'My pleasure!'],
    'options': ['I can chat with you and answer some questions.']
}

def preprocess(text):
    # transforma em minúsculas
    text = text.lower()
    # remove caracteres especiais
    text = re.sub(r'[^\w\s]', '', text)
    return text

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
vectorizer.fit([' '.join(intent) for intent in intents.values()])

def get_intent(text, vectorizer):
    text_vect = vectorizer.transform([preprocess(text)])
    intention_vect = vectorizer.transform([' '.join(intent) for intent in intents.values()])
    sim_scores = cosine_similarity(text_vect.toarray(), intention_vect.toarray())
    intention_index = sim_scores.argmax()
    intention = list(intents.keys())[intention_index]
    return intention

print("Hi! I'm a simple chatbot. How can I help you today?")

while True:
    user_message = input('> ').lower()
    if user_message == 'quit':
        print('Bye!')
        break
    intention = get_intent(user_message, vectorizer)
    response = random.choice(responses[intention])
    print(response)
