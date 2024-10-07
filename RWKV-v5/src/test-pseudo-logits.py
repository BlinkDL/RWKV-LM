# https://chatgpt.com/share/66f819e3-7528-8004-be95-f395c5c128cf
'''
I have 100 logits. I know the values of logit 1-50. I do not know values of logit
 51-100. 
Someone performed softmax over the 100 logits and get their probs. I don't know
each of the 100 probs. but I know the sum of the first 50 probs; also the sume
of the other 50 probs. 

how can I get values for logits 51-100? if the answer is not unique, it is okay
to have some "pseudo" values
'''

import numpy as np

# Softmax function
def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)

# Generate random known logits for each cluster
def generate_logits(clusters, size_per_cluster):
    return [np.random.randn(size) for size in size_per_cluster]

# Compute the sum of exponentials for each cluster of known logits
def compute_cluster_exponentials(clusters):
    return [np.sum(np.exp(cluster)) for cluster in clusters]

# Generalized function to handle m clusters of known logits and n clusters of unknown logits
def handle_known_unknown_logit_clusters(known_logit_clusters, known_prob_sums, unknown_prob_sums):
    # Step 1: Compute exponentials for known logits
    known_exponentials = compute_cluster_exponentials(known_logit_clusters)
    
    # Step 2: For each unknown cluster, solve for the sum of exponentials S_j
    unknown_exponentials = []
    for i in range(len(unknown_prob_sums)):
        # Q = sum of all exponentials from known logits
        Q = np.sum(known_exponentials)
        # S_j = Q * (unknown_prob_sum / sum of known_prob_sums)
        S_j = Q * (unknown_prob_sums[i] / np.sum(known_prob_sums))
        unknown_exponentials.append(S_j)
    
    # Step 3: Distribute unknown logits equally or use another assumption
    unknown_logit_clusters = [np.log(S_j / len(unknown_prob_sums)) * np.ones(len(unknown_prob_sums)) for S_j in unknown_exponentials]
    
    # Step 4: Concatenate all logits (known and unknown) for softmax computation
    all_logits = np.concatenate([*known_logit_clusters, *unknown_logit_clusters])
    
    # Compute softmax over all logits
    probs = softmax(all_logits)
    
    # Verify the sum of probabilities for each cluster
    results = {}
    start_idx = 0
    for i, cluster in enumerate(known_logit_clusters):
        cluster_size = len(cluster)
        cluster_probs = np.sum(probs[start_idx:start_idx + cluster_size])
        results[f"Known Cluster {i + 1}"] = cluster_probs
        start_idx += cluster_size

    for i, cluster in enumerate(unknown_logit_clusters):
        cluster_size = len(cluster)
        cluster_probs = np.sum(probs[start_idx:start_idx + cluster_size])
        results[f"Unknown Cluster {i + 1}"] = cluster_probs
        start_idx += cluster_size
    
    return results

# Example usage:

# m known clusters of logits with known probability sums
known_logit_clusters = generate_logits(2, [50, 40])  # 2 clusters with 50 and 40 logits respectively
known_prob_sums = [0.4, 0.3]  # Sum of softmax probabilities for known clusters

# n unknown clusters with known probability sums (but unknown logits)
unknown_prob_sums = [0.2, 0.1]  # Sum of softmax probabilities for unknown clusters

# Handle clusters and compute results
result = handle_known_unknown_logit_clusters(known_logit_clusters, known_prob_sums, unknown_prob_sums)

# Output the result
print("Probability sums for each cluster:")
for cluster, prob_sum in result.items():
    print(f"{cluster}: {prob_sum}")
