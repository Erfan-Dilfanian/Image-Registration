import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read data from Excel
excel_data = pd.read_excel('manualPoints.xlsx')
visible_points = excel_data[['Visible_X', 'Visible_Y']].values
thermal_points = excel_data[['Thermal_X', 'Thermal_Y']].values

print("Thermal Points Shape:", thermal_points.shape)
print("Visible Points Shape:", visible_points.shape)
# Read visible and thermal images
visible_image = cv2.imread('vs.png')
thermal_image = cv2.imread('ir.png')

# Plotting the manual points in a separate figure
plt.figure(figsize=(12, 6))

# Plotting points on visible image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(visible_image, cv2.COLOR_BGR2RGB))
plt.scatter(visible_points[:, 0], visible_points[:, 1], color='red', label='Manual Points')
plt.title('Visible Image with Manual Points')
plt.legend()

# Plotting points on thermal image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(thermal_image, cv2.COLOR_BGR2RGB))
plt.scatter(thermal_points[:, 0], thermal_points[:, 1], color='blue', label='Manual Points')
plt.title('Thermal Image with Manual Points')
plt.legend()

plt.tight_layout()
plt.show()

# Calculate transformation matrix using all points
M, _ = cv2.findHomography(np.float32(thermal_points), np.float32(visible_points))

# Invert the transformation matrix
M_inv = cv2.invert(M)[1]

# Warp thermal image based on transformation matrix
warped_thermal = cv2.warpPerspective(thermal_image, M, (visible_image.shape[1], visible_image.shape[0]))


# Fusion and alignment
alpha = 0.7  # Adjust alpha based on desired blending effect (0.0 for only thermal, 1.0 for only visible)
fusion_image = cv2.addWeighted(visible_image, alpha, warped_thermal, 1 - alpha, 0)

print('warped thermal',warped_thermal.shape)
print('infrared image shape:', thermal_image.shape)
print('rgb image shape:', visible_image.shape)
print('fusion image shape:', fusion_image.shape)

# Display or save the fused image
cv2.imshow('Fused Image', fusion_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the fused image
cv2.imwrite('fused_image.jpg', fusion_image)

# Display the homography matrix
print("Homography Matrix:")
print(M)
# Load new infrared and visible images
new_visible_image = cv2.imread('vs_test2.png')
new_thermal_image = cv2.imread('ir_test2.png')

# Warp new thermal image based on the transformation matrix
warped_new_thermal = cv2.warpPerspective(new_thermal_image, M, (new_visible_image.shape[1], new_visible_image.shape[0]))

# Fusion and alignment for the new set of images
alpha = 0.5  # Adjust alpha based on desired blending effect (0.0 for only thermal, 1.0 for only visible)
fusion_new_image = cv2.addWeighted(new_visible_image, alpha, warped_new_thermal, 1 - alpha, 0)

# Display or save the fused image for the new set of images
cv2.imshow('Fused New Image', fusion_new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Save the fused image
cv2.imwrite('fused_test_image.jpg', fusion_new_image)

'''
# Load new infrared and visible images
new_visible_image = cv2.imread('vs_test2.png')
new_thermal_image = cv2.imread('ir_test2.png')

# Warp new thermal image based on the transformation matrix
warped_new_thermal = cv2.warpPerspective(new_thermal_image, M, (new_visible_image.shape[1], new_visible_image.shape[0]))

# Fusion and alignment for the new set of images
alpha = 0.7  # Adjust alpha based on desired blending effect (0.0 for only thermal, 1.0 for only visible)
fusion_new_image = cv2.addWeighted(new_visible_image, alpha, warped_new_thermal, 1 - alpha, 0)

# Display or save the fused image for the new set of images
cv2.imshow('Fused New Image', fusion_new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
# Warp visible image using inverted transformation matrix
warped_visible = cv2.warpPerspective(visible_image, M_inv, (thermal_image.shape[1], thermal_image.shape[0]))

# Resize images to the same dimensions
warped_visible = cv2.resize(warped_visible, (warped_thermal.shape[1], warped_thermal.shape[0]))

# Fusion and alignment
alpha = 0.5  # Adjust alpha based on desired blending effect (0.0 for only thermal, 1.0 for only visible)
fusion_image = cv2.addWeighted(warped_visible, alpha, warped_thermal, 1 - alpha, 0)

print('infrared image shape:', thermal_image.shape)
print('rgb image shape:', visible_image.shape)
print('fusion image shape:', fusion_image.shape)

# Display or save the fused image
cv2.imshow('Fused Image', fusion_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the fused image
cv2.imwrite('fused_image.jpg', fusion_image)

'''