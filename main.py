import cv2
import pytesseract
import gradio as gr
import numpy as np

# Configure the Tesseract path (update this to your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Apply dilation and erosion
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Deskew the image
    coords = np.column_stack(np.where(eroded > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    h, w = eroded.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(eroded, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Perform OCR on the preprocessed image
    extracted_text = pytesseract.image_to_string(rotated, lang='eng')

    return extracted_text


iface = gr.Interface(
    fn=preprocess_image,
    inputs=gr.inputs.Image(type="numpy", label="Upload an image"),
    outputs=gr.outputs.Textbox(label="Extracted Text"),
    live=True,
    capture_session=True,
    title="Improved Image Text Extraction",
    description="Upload an image to extract text using OpenCV, Tesseract, and preprocessing.",
)

if __name__ == "__main__":
    iface.launch()
