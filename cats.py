import cv2
import numpy as np

# Load and preprocess input image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize image to 100x100 for example purposes
    image = cv2.resize(image, (100, 100))
    return image

# Placeholder function for boundary tracing
def boundary_tracing(Y_hat, Y):
    # Placeholder: Simply return the input as is
    return Y_hat

# Placeholder function for texture suppression
def texture_suppression(Y_hat, Y):
    # Placeholder: Simply return the input as is
    return Y_hat

# Placeholder function for weighted cross entropy
def weighted_cross_entropy(Y_hat, Y):
    Y_plus = np.where(Y == 255)  # Get indices of edge pixels
    Y_minus = np.where(Y == 0)   # Get indices of non-edge pixels
    epsilon = 1e-10  # Small epsilon value to avoid division by zero
    ce_loss = -np.sum(np.log(Y_hat[Y_plus] + epsilon)) - np.sum(np.log(1 - Y_hat[Y_minus] + epsilon))
    return alpha * ce_loss

# CoFusion Block
def co_fusion(Y_hats):
    Y_hats = np.array(Y_hats)
    k, h, w = Y_hats.shape
    Y_hats_flat = Y_hats.reshape((k, h * w))
    
    Q = Y_hats_flat
    K = Y_hats_flat
    V = Y_hats_flat

    attention = np.matmul(Q, K.T)
    attention = attention / np.sqrt(d_k)
    attention -= attention.max()  # Scale down to prevent overflow
    attention = np.exp(attention)
    attention /= np.maximum(attention.sum(axis=1, keepdims=True), 1e-9)  # Avoid division by zero
    attention[np.isnan(attention)] = 0  # Replace NaN with 0

    co_fused = np.matmul(attention, V)
    co_fused = co_fused.reshape((h, w))

    return co_fused

# Load and preprocess input images
input_image1 = load_image(r"A:\OneDrive - INSTITUTE OF ENGINEERING & MANAGEMENT\IEDC\DAEN\Codes\image1.jpeg")
input_image2 = load_image(r"A:\OneDrive - INSTITUTE OF ENGINEERING & MANAGEMENT\IEDC\DAEN\Codes\image2.jpg")

# Perform edge detection using CATS methodology for both images
alpha = 0.5
lambda1 = 0.1
lambda2 = 0.2
d_k = 64  # Dimension of key vectors

Y_hat1 = input_image1.astype(float) / 255.0  # Normalize image
Y1 = cv2.Canny(input_image1, 100, 200)  # Edge label using Canny edge detector
final_edge_prediction1 = co_fusion([Y_hat1])

Y_hat2 = input_image2.astype(float) / 255.0  # Normalize image
Y2 = cv2.Canny(input_image2, 100, 200)  # Edge label using Canny edge detector
final_edge_prediction2 = co_fusion([Y_hat2])

# Calculate areas of edges detected in each image
area1 = np.sum(final_edge_prediction1) / 255
area2 = np.sum(final_edge_prediction2) / 255

# Compare areas
if area1 < area2:
    print("Area has increased.")
elif area1 > area2:
    print("Area has decreased.")
else:
    print("Area remains the same.")

# Save final edge-detected images (optional)
output_image1 = (final_edge_prediction1 * 255).astype(np.uint8)
output_image2 = (final_edge_prediction2 * 255).astype(np.uint8)
cv2.imwrite('output1.jpg', output_image1)
cv2.imwrite('output2.jpg', output_image2)
