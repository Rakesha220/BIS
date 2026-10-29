
# ================================================
# CUCKOO SEARCH BASED IMAGE SEGMENTATION
# 2D MAXIMUM ENTROPY THRESHOLDING
# ================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from google.colab import files

# 1. UPLOAD IMAGE
print("Upload an image for segmentation:")
uploaded = files.upload()

IMAGE_PATH = next(iter(uploaded))

image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Unable to read image. Upload a valid image.")

print("Image uploaded:", IMAGE_PATH)
print("Image size:", image.shape)

# 2. INITIALIZE PARAMETERS
N = 50                 # Number of nests
Pa = 0.7               # Abandonment probability
MAX_ITER = 30          # Maximum iterations
B = 10                 # Local search radius
LIMIT = 255

np.random.seed(42)


# 3. CALCULATE 2D HISTOGRAM
def get_2d_histogram(image):

    gray = image.astype(np.uint8)

    # Mean intensity of 3x3 neighboring pixels
    mean_img = cv2.blur(
        gray.astype(np.float32), (3, 3)
    )

    mean_img = np.clip(
        np.rint(mean_img), 0, 255
    ).astype(np.uint8)

    hist, _, _ = np.histogram2d(
        gray.ravel(),
        mean_img.ravel(),
        bins=256,
        range=[[0, 256], [0, 256]]
    )

    P = hist / hist.sum()

    return P, mean_img


# 4. CALCULATE 2D MAXIMUM ENTROPY FITNESS
def fitness(nest, P):

    s, t = np.clip(
        np.rint(nest).astype(int), 1, 255
    )

    # Four regions of the 2D histogram
    regions = [
        P[:s, :t],
        P[s:, :t],
        P[:s, t:],
        P[s:, t:]
    ]

    entropy = 0.0

    for region in regions:

        total = region.sum()

        if total > 0:

            values = region[region > 0]

            entropy -= np.sum(
                values * np.log(values / total)
            )

    return entropy


# 5. LEVY FLIGHT
def levy_flight():

    beta = 1.5

    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (
            math.gamma((1 + beta) / 2)
            * beta
            * 2 ** ((beta - 1) / 2)
        )
    ) ** (1 / beta)

    u = np.random.normal(0, sigma, 2)
    v = np.random.normal(0, 1, 2)

    return u / (
        np.abs(v) ** (1 / beta) + 1e-10
    )


# 6. CUCKOO SEARCH ALGORITHM
def cuckoo_search(P):

    # Initialize nests with random threshold pairs
    nests = np.random.randint(
        1, 256, size=(N, 2)
    )

    scores = np.array([
        fitness(nest, P) for nest in nests
    ])

    best_index = np.argmax(scores)

    best_nest = nests[best_index].copy()
    best_score = scores[best_index]

    print("\nStarting Cuckoo Search...\n")

    for iteration in range(MAX_ITER):

        # Generate new solutions using Levy flights
        for i in range(N):

            new_nest = (
                nests[i] + levy_flight()
            )

            new_nest = np.clip(
                np.rint(new_nest), 1, 255
            ).astype(int)

            new_score = fitness(new_nest, P)

            # Compare with a randomly selected nest
            j = np.random.randint(N)

            if new_score > scores[j]:

                nests[j] = new_nest
                scores[j] = new_score

        # Abandon worst nests
        abandon_count = max(1, int(Pa * N))

        worst_indices = np.argsort(scores)[
            :abandon_count
        ]

        for i in worst_indices:

            nests[i] = np.random.randint(
                1, 256, size=2
            )

            scores[i] = fitness(nests[i], P)

        # Update best solution
        best_index = np.argmax(scores)

        if scores[best_index] > best_score:

            best_score = scores[best_index]
            best_nest = nests[best_index].copy()

        print(
            f"Iteration {iteration + 1:02d}/{MAX_ITER} | "
            f"Best entropy = {best_score:.6f} | "
            f"Thresholds = {best_nest}"
        )

    return best_nest


# 7. LOCAL SEARCH
def local_search(best, P):

    best_s, best_t = best

    best_score = fitness(best, P)
    result = best.copy()

    s_min = max(1, best_s - B)
    s_max = min(255, best_s + B)

    t_min = max(1, best_t - B)
    t_max = min(255, best_t + B)

    for s in range(s_min, s_max + 1):

        for t in range(t_min, t_max + 1):

            score = fitness([s, t], P)

            if score > best_score:

                best_score = score
                result = np.array([s, t])

    return result


# 8. RUN SEGMENTATION
P, mean_img = get_2d_histogram(image)

best = cuckoo_search(P)

best = local_search(best, P)

s, t = best

print("\n================================")
print("SEGMENTATION RESULTS")
print("Optimal threshold s =", s)
print("Optimal threshold t =", t)
print("Maximum entropy =", fitness(best, P))
print("================================")


# 9. CREATE BINARY SEGMENTED IMAGE
gray = image.astype(np.float32)
mean = mean_img.astype(np.float32)

segmented = (
    (gray >= s) & (mean >= t)
).astype(np.uint8) * 255


# 10. DISPLAY RESULTS
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mean_img, cmap="gray")
plt.title("3x3 Local Mean Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(segmented, cmap="gray")
plt.title(f"Cuckoo Search Segmentation\ns={s}, t={t}")
plt.axis("off")

plt.tight_layout()
plt.show()


# 11. SAVE AND DOWNLOAD SEGMENTED IMAGE
output_path = "segmented_image.png"

cv2.imwrite(output_path, segmented)

print("\nSegmented image saved successfully!")

files.download(output_path)
