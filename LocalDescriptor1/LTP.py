import numpy as np
import cv2

def ltp(image_gray, k=0.20):
    """
    Improved LTP:
    - Gaussian smoothing (removes noise, keeps texture)
    - Adaptive threshold = k * local mean gradient
    - Upper pattern -> 256-d feature

    returns:
    ltp_img (128x128)
    hist (256-d normalized vector)
    """

    # Smooth image for stable texture extraction
    img = cv2.GaussianBlur(image_gray, (3, 3), 0).astype(np.float32)


    # Compute adaptive threshold using local gradients
    # Sobel in LTP just helps choose the threshold
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    T = k * np.mean(grad)  # adaptive threshold

    # Pad to avoid border issues
    padded = np.pad(img, pad_width=1, mode='edge')

    neighbors = [
        (0, 1), (-1, 1), (-1, 0), (-1, -1),
        (0, -1), (1, -1), (1, 0), (1, 1)
    ]

    H, W = img.shape
    upper = np.zeros((H, W), dtype=np.uint8)

    for i, (dy, dx) in enumerate(neighbors):
        shifted = padded[1+dy:H+1+dy, 1+dx:W+1+dx]
        diff = shifted - img
        bit = (diff > T).astype(np.uint8)
        upper |= (bit << i)

    ltp_img = upper

    # Histogram (256 bins)
    hist, _ = np.histogram(ltp_img, bins=256, range=(0, 255))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)

    return ltp_img, hist
