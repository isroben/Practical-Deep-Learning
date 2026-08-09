class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label  # only for leaves

    def is_leaf(self):
        return self.label is not None

def impurity(data):
    total = len(data)
    if total == 0:
        return 0
    counts = {}
    for row in data:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    gini = 1
    for c in counts.values():
        p = c / total
        gini -= p ** 2
    return gini

def weighted_impurity(left, right):
    n_left, n_right = len(left), len(right)
    n_total = n_left + n_right

    return (n_left/n_total) * impurity(left) + (n_right/n_total) * impurity(right)

def partition(data, j, t):
    left = [row for row in data if row[j] <= t]
    right = [row for row in data if row[j] > t]
    return left, right


def find_best_split(data, features):
    best_impurity = float('inf')
    best_j, best_t = None, None

    for j in features:
        values = sorted(set(row[j] for row in data)) # converts the set{} to list[]
        thresholds = [(values[i] + values[i+1]) / 2 for i in range(len(values)-1)]

        for t in thresholds:
            print(f"Finding best split among them: feat: {j}, thres: {t}:")

            left, right = partition(data, j, t)
            print(f"Partition: Left: {left}, Right: {right}")

            if not left or not right:
                continue
            w = weighted_impurity(left, right)
            print(f"Weighted impurity of child node: {w}\n")
            if w < best_impurity:
                best_impurity = w
                best_j, best_t = j, t

    return best_j, best_t, best_impurity


def majority_class(data):
    counts = {}
    for row in data:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1

    return max(counts, key=counts.get) # returns the most common class


def build_tree(data, features, depth=0, max_depth=5, min_samples=2):
    if depth >= max_depth or len(data) < min_samples or impurity(data) == 0:
        return Node(label=majority_class(data))

    j, t, best_impurity = find_best_split(data, features)
    print(f"\nbest feature: {j}, best_threshold: {t}, best_impurity: {best_impurity}")

    if j is None or impurity(data) - best_impurity <= 0:
        print("No better impurity")
        return Node(label=majority_class(data))

    left_data, right_data = partition(data, j, t)
    print(f"left_data: {left_data} \n Right_data: {right_data}\n")

    left_subtree = build_tree(left_data, features, depth+1, max_depth, min_samples)
    right_subtree = build_tree(right_data, features, depth+1, max_depth, min_samples)

    return Node(feature=j, threshold=t, left=left_subtree, right=right_subtree)


def predict(x, node):
    if node.is_leaf():
        return node.label
    if x[node.feature] <= node.threshold:
        return predict(x, node.left)
    else:
        return predict(x, node.right)

# usage
# feature_0 = Hours studied, feature_1 = Slept well (0/1), label = Pass (0/1)
# feature_0 = Income (k$), feature_1 = Credit score (hundreds, e.g. 6=600s), 
# feature_2 = Has existing debt (0/1), label = Loan approved (0/1)

train = [
    [25, 5, 1, 0],
    [30, 6, 1, 0],
    [45, 6, 0, 1],
    [20, 4, 1, 0],
    [60, 7, 0, 1],
    [35, 5, 1, 0],
    [50, 7, 1, 1],
    [22, 5, 0, 0],
    [70, 8, 0, 1],
    [40, 6, 1, 0],
    [55, 7, 0, 1],
    [28, 4, 1, 0],
    [65, 8, 1, 1],
    [33, 5, 0, 0],
    [48, 6, 0, 1],
    [38, 5, 1, 0],
    [58, 7, 1, 1],
    [26, 4, 0, 0],
    [42, 6, 1, 1],
    [31, 5, 1, 0],
]

test = [
    [27, 5, 1, 0],
    [52, 7, 0, 1],
    [44, 6, 1, 1],
    [24, 4, 0, 0],
    [63, 8, 0, 1],
    [36, 5, 1, 0],
]

# usage
num_features = len(train[0]) - 1   # exclude the label column
features = list(range(num_features))
tree = build_tree(train, features=features)

correct = 0
for row in test:
    x = row[:-1]
    true_label = row[-1]
    pred = predict(x, tree)
    correct += (pred == true_label)
    print(f"x={x}  true={true_label}  predicted={pred}")

print(f"Accuracy: {correct}/{len(test)}")