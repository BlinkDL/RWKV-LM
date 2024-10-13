import torch

class EmbeddingLookupTable:
    def __init__(self, weight, embedding_dim, vocab_size, cache_size=1000):
        self.embedding_table = weight       # w["emb.weight']
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.cache = {}
        self.cache_size = cache_size

    def get_embedding_table(self):
        return self.embedding_table

    def get_embedding(self, token_id):
        if token_id in self.cache.keys():
            return self.cache[token_id]

        embedding = self.embedding_table[token_id]

        if self.cache_size < len(self.cache):
            self.cache.pop(next(iter(self.cache)))
            print("Cache is full. Removing the oldest entry.")

        self.cache[token_id] = embedding
        return embedding

    def get_embeddings(self, token_ids):
        embeddings = [self.get_embedding(tid) for tid in token_ids]
        return torch.stack(embeddings, dim=0)

    def __repr__(self):
        return f"EmbeddingLookupTable(embedding_dim={self.embedding_dim}, vocab_size={self.vocab_size})"
