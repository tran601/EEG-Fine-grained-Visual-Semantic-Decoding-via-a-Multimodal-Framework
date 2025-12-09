import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

# load data file
path = "data_total.npy"
data = np.load(path, allow_pickle=True)

np.random.seed(42)


label_to_imgids = defaultdict(set)
imgid_to_items = defaultdict(list)

for idx, item in enumerate(data):
    label = item["label"]
    img_label = item["img_label"]
    label_to_imgids[label].add(img_label)
    imgid_to_items[(label, img_label)].append(idx)

# 8:2
train_indices = []
test_indices = []

for label, img_labels in label_to_imgids.items():
    img_labels = list(img_labels)
    train_img_labels, test_img_labels = train_test_split(
        img_labels, test_size=0.2, random_state=42
    )

    for img_lbl in train_img_labels:
        train_indices.extend(imgid_to_items[(label, img_lbl)])
    for img_lbl in test_img_labels:
        test_indices.extend(imgid_to_items[(label, img_lbl)])

train_data = data[train_indices]
test_data = data[test_indices]

print(f"Total samples: {len(data)}")
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

train_labels = set(item["label"] for item in train_data)
test_labels = set(item["label"] for item in test_data)
assert (
    train_labels == test_labels == set(range(4))
), "Some labels missing in train or test!"

np.save("train.npy",train_data)
np.save("test.npy",test_data)
print("划分完成！")
