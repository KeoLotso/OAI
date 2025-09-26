import os
from collections import Counter

# Get tokens (basicly rate youre data.txt)

def main():
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data.txt")

    if not os.path.exists(data_path):
        print("data.txt not found")
        return

    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens = text.split()
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    most_common = Counter(tokens).most_common(10)

    print(f"Total tokens: {total_tokens}")
    print(f"Unique tokens: {unique_tokens}")
    print("Top 10 most common tokens:")
    for word, count in most_common:
        print(f"{word}: {count}")

    #Simple vocab thoingy
    vocab_ratio = unique_tokens / total_tokens if total_tokens > 0 else 0
    if total_tokens < 1000:
        size_score = 1
    elif total_tokens < 10000:
        size_score = 2
    elif total_tokens < 100000:
        size_score = 3
    else:
        size_score = 4

    quality_score = round(vocab_ratio * 10 + size_score, 1)
    print(f"Dataset Score: {quality_score}/14")

if __name__ == "__main__":
    main()
