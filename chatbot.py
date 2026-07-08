import nltk
nltk.download('punkt')

from faq_data import faqs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = list(faqs.keys())
import string

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
def get_answer(user_question):
    # Combine stored questions with user input
    all_questions = questions + [user_question]
    
    # Convert text into vectors
    vectorizer = CountVectorizer().fit_transform(all_questions)
    vectors = vectorizer.toarray()

    # Compare user question with all FAQs
    similarity = cosine_similarity([vectors[-1]], vectors[:-1])

    # Get best matching FAQ
    index = similarity.argmax()
    score = similarity[0][index]

    if score > 0.2:
        return faqs[questions[index]]
    else:
        return "Sorry, I couldn't find an answer for that."
if __name__  ==  "__main__":
    print("=== FAQ Chatbot ===")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        answer = get_answer(user_input)
        print("Bot:", answer)