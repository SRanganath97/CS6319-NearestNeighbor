from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class LSH:
    def __init__(self, data):
        self.vectorizer = HashingVectorizer(norm='l2')
        self.hash_dict = {}
        self.num_buckets = None
        self.train(data)

    def train(self, data):
        self.num_buckets = 1000
        hashed_data = self.vectorizer.transform(data)
        for i, row in enumerate(hashed_data):
            for j, val in enumerate(row.data):
                h = hash((j, val)) % self.num_buckets
                if h not in self.hash_dict:
                    self.hash_dict[h] = []
                self.hash_dict[h].append(i)

    def query(self, query_vec, k):
        hashed_query = self.vectorizer.transform([query_vec])
        candidate_set = set()
        for j, val in enumerate(hashed_query.data):
            h = hash((j, val)) % self.num_buckets
            if h in self.hash_dict:
                candidate_set.update(self.hash_dict[h])
        candidates = [self.data[i] for i in candidate_set]
        similarities = cosine_similarity(candidates, [query_vec]).flatten()
        results = [(i, sim) for i, sim in zip(candidate_set, similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
