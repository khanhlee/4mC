from gensim.models import FastText
import pandas as pd

def generate_N_grams(text, ngram):
    """Generate n-grams of a given length from text."""
    temp = zip(*[text[i:] for i in range(0, ngram)])
    return [''.join(ngram) for ngram in temp]

# Set the n-gram length
n_gram = 3

# Load and process the positive training dataset
pos_file_path = 'data/F_vesca_Training_datasets/F_vesca_Pos_Training_3457.txt'
with open(pos_file_path) as f:
    lines = f.readlines()
filtered_lines = [lines[i].strip() for i in range(2, len(lines), 4)]

# Initialize lists for positive dataset
res = []
res2 = []

# Generate n-grams for positive dataset
for line in filtered_lines:
    res.append(generate_N_grams(line, n_gram))
    res2.append(generate_N_grams(line, n_gram))

# Load and process the negative training dataset
neg_file_path = 'data/F_vesca_Training_datasets/F_vesca_Neg_Training_3457.txt'
with open(neg_file_path) as f:
    lines = f.readlines()
filtered_lines_Neg = [lines[i].strip() for i in range(2, len(lines), 4)]

# Initialize lists for negative dataset
res_Neg = []
res2_Neg = []

# Generate n-grams for negative dataset
for line in filtered_lines_Neg:
    res_Neg.append(generate_N_grams(line, n_gram))
    res2_Neg.append(generate_N_grams(line, n_gram))

# Combine positive and negative datasets
res.extend(res_Neg)
res2.extend(res2_Neg)

# Train the FastText model
ft_model = FastText(sentences=res, min_count=1, window=5, negative=5, epochs=20)

# Create a list of word vectors for FastText
ft_string = [[word, ft_model.wv.get_vector(word)] for word in ft_model.wv.index_to_key]

# Replace words with their FastText vectors in the training data
for i in range(len(res)):
    for j in range(len(res[i])):
        res[i][j] = ','.join(map(str, ft_model.wv.get_vector(res[i][j])))

# Add labels to the training data
count = 1
for row in res:
    row.append(1 if count <= 3457 else 0)
    count += 1

# Convert the processed data to a DataFrame
df = pd.DataFrame(res)
# Save the DataFrame to a CSV file
df.to_csv('N4_ft_3mer_training.csv', index=False)

# Save the original data (without vector transformation) to another CSV file
df_origin = pd.DataFrame(res2)
df_origin.to_csv('N4_ft_3mer_training_DNA.csv', index=False)
