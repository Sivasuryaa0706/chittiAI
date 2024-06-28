from sentence_transformers import SentenceTransformer
import numpy as np
import json
from scipy.spatial.distance import cosine

class ChittiQAModel:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.question_embeddings = np.load('question_embeddings.npy')
        self.category_embeddings = np.load('category_embeddings.npy')
        with open('expanded_dataset.json', 'r') as f:
            self.dataset = json.load(f)

    def find_best_match(self, query):
        query_embedding = self.model.encode([query])[0]
        similarity_threshold = 0.5
        question_similarities = [1 - cosine(query_embedding, qe) for qe in self.question_embeddings]
        category_similarities = [1 - cosine(query_embedding, ce) for ce in self.category_embeddings]

        combined_similarities = [0.7 * qs + 0.3 * cs for qs, cs in zip(question_similarities, category_similarities)]
        best_match_index = np.argmax(combined_similarities)
        best_match_similarity = combined_similarities[best_match_index]

        if best_match_similarity >= similarity_threshold:
            return self.dataset[best_match_index]['url']
        else:
            return None

    def predict(self, query):
        best_match_url = self.find_best_match(query)
        return best_match_url if best_match_url is not None else "I'm sorry, I couldn't find a specific resource for that question."