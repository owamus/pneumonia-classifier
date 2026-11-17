from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

root = Path("data/chest_xray")

# count the images in each folder
for split in ["train", "val", "test"]:
    for cls in ["NORMAL", "PNEUMONIA"]:
        count = len(list((root / split / cls).glob("*.jpeg")))
        print(split, cls, count)

# show 4 random images for each class
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for row, cls in enumerate(["NORMAL", "PNEUMONIA"]):
    files = list((root / "train" / cls).glob("*.jpeg"))
    for col, f in enumerate(random.sample(files, 4)):
        img = Image.open(f)
        axes[row, col].imshow(img, cmap="gray")
        axes[row, col].set_title(f"{cls} {img.size}")
        axes[row, col].axis("off")
plt.tight_layout()
plt.savefig("samples.png")
plt.show()