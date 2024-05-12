from gensim.models import FastText
import numpy as np

# Initialize and train the FastText model
ft_model = FastText(sentences=res, min_count=1, window=5, negative=5, epochs=20)

# Extract the word vectors and store them in a list
ft_string = [[word, ft_model.wv.get_vector(word)] for word in ft_model.wv.index_to_key]

# Replace each word in the dataset with its corresponding FastText vector
for i in range(len(res)):
    for j in range(len(res[i])):
        res[i][j] = ','.join(map(str, ft_model.wv.get_vector(res[i][j])))

# Convert the list of word vectors to a numpy array for further processing
ft_string_array_new = np.array([entry[1] for entry in ft_string])

# Assuming `LSH` is a class for Locality-Sensitive Hashing, instantiate and train it
lsh_model = LSH(ft_string_array_new)
lsh_trained_model = lsh_model.train(30)

# Access and print the LSH model table for debugging purposes
# Adjust this part according to the LSH class implementation details
print(lsh_trained_model.model['table'])
