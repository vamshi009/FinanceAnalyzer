
from sentence_transformers import SentenceTransformer
from category_embedding import CLASS_EMBEDDINGS
import numpy as np


model = SentenceTransformer('all-MiniLM-L6-v2')



def cosine_similarity_numpy(vec1, vec2):
    """
    Calculates the cosine similarity between two 1D vectors using NumPy.
    """
    # Convert lists to numpy arrays if they are not already
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Check if the vectors are non-zero (division by zero would occur otherwise)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0 # Or handle as per specific requirements
        
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1) # L2 norm (magnitude)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    similarity = dot_product / (magnitude_vec1 * magnitude_vec2)
    return similarity



def get_confidence_score(description, category):

    description_embedding = model.encode(description)
    category_embedding = CLASS_EMBEDDINGS[category]
    similarity = cosine_similarity_numpy(description_embedding, category_embedding)
    normalized_score = (similarity + 1.0) / 2.0

    return normalized_score


