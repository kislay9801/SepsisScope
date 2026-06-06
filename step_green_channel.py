import cv2
import matplotlib.pyplot as plt

# Load one DRIVE training image
img = cv2.imread(r"C:\SepsisScope\data\DRIVE\21_training.tif")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Split into red, green, blue channels
red   = img[:, :, 0]
green = img[:, :, 1]
blue  = img[:, :, 2]

# Display all four: original + three channels
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(img)
axes[0].set_title("Original colour image")
axes[0].axis("off")

axes[1].imshow(red, cmap="gray")
axes[1].set_title("Red channel")
axes[1].axis("off")

axes[2].imshow(green, cmap="gray")
axes[2].set_title("Green channel")
axes[2].axis("off")

axes[3].imshow(blue, cmap="gray")
axes[3].set_title("Blue channel")
axes[3].axis("off")

plt.suptitle("Green channel shows vessels most clearly", fontsize=13)
plt.tight_layout()
plt.show()