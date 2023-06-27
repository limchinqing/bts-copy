import datetime
import cv2
import numpy as np

def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a threshold to the image to create a binary image
    _, binary = cv2.threshold(gray, 164, 255, cv2.THRESH_BINARY_INV)

    # Define the structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Perform morphological operations to remove noise and fill gaps in the Braille dots
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours in the image
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    radius = 0
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        radius = int(max(w, h) / 2)
        
    # Loop through the contours and draw a circle around each Braille dot
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.circle(image, (x + int(w/2), y + int(h/2)), radius, (0, 0, 0), -6)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask
    _, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Invert the binary mask so that black areas become white and vice versa
    mask = cv2.bitwise_not(threshold)

    # Create a white background image of the same size as the original image
    white_background = np.ones_like(image) * 255

    # Bitwise AND operation to extract black areas on the white background
    result = cv2.bitwise_and(white_background, white_background, mask=mask)
    result = cv2.bitwise_not(result)

    output_filename = "output.jpg"

    # Save the result with the updated filename
    cv2.imwrite(output_filename, result)

    return output_filename
