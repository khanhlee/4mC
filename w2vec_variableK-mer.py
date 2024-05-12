def generate_random_grams(text, kmin, kmax):
    """Generate random n-grams between kmin and kmax lengths."""
    import random
    n = random.randint(kmin, kmax)
    return [''.join(text[i:i + n]) for i in range(len(text) - n + 1)]

# Define k-min and k-max
kmin, kmax = 3, 5

# Load and process positive training dataset
file_path = 'data/F_vesca_Training_datasets/F_vesca_Pos_Training_3457.txt'
with open(file_path) as f:
    lines = f.readlines()

# Extract relevant lines at specified intervals
filtered_lines = [lines[i].strip() for i in range(2, len(lines), 4)]

# Initialize lists to hold results
res, res2 = [], []

# Generate random n-grams for each line
for line in filtered_lines:
    res.append(generate_random_grams(line, kmin, kmax))
    res2.append(generate_random_grams(line, kmin, kmax))
