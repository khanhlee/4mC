from gensim.models import Word2Vec
import pandas as pd

def generate_N_grams(text, ngram):
    temp = zip(*[text[i:] for i in range(ngram)])
    ans = [''.join(ngram) for ngram in temp]
    return ans

n_gram = 6

# Load and process positive training dataset
with open('data/F_vesca_Training_datasets/F_vesca_Pos_Training_3457.txt') as f:
    lines = f.readlines()
    filtered_lines = [lines[i].strip() for i in range(2, len(lines), 4)]

res, res2 = [], []

for line in filtered_lines:
    res.append(generate_N_grams(line, n_gram))
    res2.append(generate_N_grams(line, n_gram))

# Load and process negative training dataset
with open('data/F_vesca_Training_datasets/F_vesca_Neg_Training_3457.txt') as f:
    lines = f.readlines()
    filtered_lines_Neg = [lines[i].strip() for i in range(2, len(lines), 4)]

res_Neg, res2_Neg = [], []

for line in filtered_lines_Neg:
    res_Neg.append(generate_N_grams(line, n_gram))
    res2_Neg.append(generate_N_grams(line, n_gram))

# Combine positive and negative datasets
res.extend(res_Neg)
res2.extend(res2_Neg)

# Train Word2Vec model
w2vmodel = Word2Vec(res, min_count=1, window=5, negative=5, epochs=20)

w2v_string = [[word, w2vmodel.wv.get_vector(word)] for word in w2vmodel.wv.index_to_key]

# Convert n-grams to word vectors and append labels
for i, row in enumerate(res):
    for j, gram in enumerate(row):
        res[i][j] = ','.join(map(str, w2vmodel.wv.get_vector(gram)))
    row.append(1 if i < 3457 else 0)

# Save training dataset
df = pd.DataFrame(res)
df.to_csv('N4_w2v_6mer_training.csv', index=False)

df_origin = pd.DataFrame(res2)
df_origin.to_csv('N4_w2v_6mer_training_DNA.csv', index=False)

# Load and process independent datasets
def load_filtered_lines(file_path, start=2, step=4):
    with open(file_path) as f:
        lines = f.readlines()
        return [lines[i].strip() for i in range(start, len(lines), step)]

filtered_lines_test_pos = load_filtered_lines('data/F_vesca_Independent_datasets/F_vesca_Pos_Ind_864.txt')
filtered_lines_test_neg_864 = load_filtered_lines('data/F_vesca_Independent_datasets/F_vesca_Neg_Ind_864.txt')
filtered_lines_test_neg_4320 = load_filtered_lines('data/F_vesca_Independent_datasets/F_vesca_Neg_Ind_4320.txt', start=1, step=2)
filtered_lines_test_neg_12960 = load_filtered_lines('data/F_vesca_Independent_datasets/F_vesca_Neg_Ind-12960.txt', start=1, step=2)

# Generate n-grams for positive and negative test datasets
def process_lines_to_ngrams(filtered_lines, n_gram):
    res = [generate_N_grams(line, n_gram) for line in filtered_lines]
    res_DNA = res.copy()
    return res, res_DNA

res_test_pos, res_test_pos_DNA = process_lines_to_ngrams(filtered_lines_test_pos, n_gram)
res_test_neg_864, res_test_neg_864_DNA = process_lines_to_ngrams(filtered_lines_test_neg_864, n_gram)
res_test_neg_4320, res_test_neg_4320_DNA = process_lines_to_ngrams(filtered_lines_test_neg_4320, n_gram)
res_test_neg_12960, res_test_neg_12960_DNA = process_lines_to_ngrams(filtered_lines_test_neg_12960, n_gram)

# Convert test n-grams to word vectors and handle exceptions
def convert_to_word_vectors_and_clean(res, res_DNA, w2vmodel):
    exceptions = []
    for i in range(len(res)):
        try:
            for j in range(len(res[0])):
                res[i][j] = ','.join(map(str, w2vmodel.wv.get_vector(res[i][j])))
        except KeyError:
            exceptions.append(i)
            break

    for i in reversed(exceptions):
        del res[i]
        del res_DNA[i]

convert_to_word_vectors_and_clean(res_test_pos, res_test_pos_DNA, w2vmodel)
convert_to_word_vectors_and_clean(res_test_neg_864, res_test_neg_864_DNA, w2vmodel)
convert_to_word_vectors_and_clean(res_test_neg_4320, res_test_neg_4320_DNA, w2vmodel)
convert_to_word_vectors_and_clean(res_test_neg_12960, res_test_neg_12960_DNA, w2vmodel)

# Append labels to test datasets
for row in res_test_pos:
    row.append(1)

for row in [res_test_neg_864, res_test_neg_4320, res_test_neg_12960]:
    for item in row:
        item.append(0)

# Save test datasets
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

save_to_csv(res_test_neg_864, 'N4_w2v_test_6mer_1to1.csv')
save_to_csv(res_test_neg_864_DNA, 'N4_w2v_test_6mer_1to1_DNA.csv')
save_to_csv(res_test_neg_4320, 'N4_w2v_test_6mer_1to5.csv')
save_to_csv(res_test_neg_4320_DNA, 'N4_w2v_test_6mer_1to5_DNA.csv')
save_to_csv(res_test_neg_12960, 'N4_w2v_test_6mer_1to15.csv')
save_to_csv(res_test_neg_12960_DNA, 'N4_w2v_test_6mer_1to15_DNA.csv')
