import streamlit as st
import numpy as np
import nltk
import string
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the content file
with open('chatBot_content.txt', 'r', errors='ignore') as f:
    raw_doc = f.read().lower()

sent_tokens = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREET_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "hai")
GREET_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! you are talking to me"]

def greet(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)

def response(user_response):
    robo1_response = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo1_response = robo1_response + "I am sorry! I don't understand you"
        return robo1_response
    else:
        robo1_response = robo1_response + sent_tokens[idx]
        return robo1_response

def main():
    global word_tokens  # Declare word_tokens as a global variable

    st.title("GEAR GURU")

    st.write("BOT: My name is Gear Guru. Let's have a conversation! Also, if you want to exit anytime, just type bye!")

    user_response = st.text_input("You:")
    if st.button("Send"):
        if user_response.lower() != 'bye':
            if user_response.lower() in ['thanks', 'thank you']:
                st.write("BOT: You are welcome")
            else:
                if greet(user_response) is not None:
                    st.write(f"BOT: {greet(user_response)}")
                else:
                    sent_tokens.append(user_response)
                    word_tokens = word_tokens + nltk.word_tokenize(user_response)
                    final_words = list(set(word_tokens))
                    st.write("BOT: " + response(user_response))
                    sent_tokens.remove(user_response)
        else:
            st.write("BOT: Goodbye! Take care...")

if __name__ == "__main__":
    # Your main program logic goes here

    main()