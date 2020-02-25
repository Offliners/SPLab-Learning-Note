import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.', 'Oh! Oh! Oh! ... The mouse is jumping on the bed!']

token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length = 10
results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))

print(results.shape)

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1.

for sample in samples:
    for word in sample.split():
        print(word)
    print(sample)
    
print(token_index)

print(results)
