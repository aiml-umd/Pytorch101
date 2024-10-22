import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from streamlit_drawable_canvas import st_canvas
import io
import cv2

# Define the canvas size (28x28 grid for MNIST)
canvas_size = 28
canvas_pixel_size = 280  # Size in pixels for rendering the canvas

if "model" not in st.session_state:
    st.session_state.model = None

# Predict the digit from the grid using the uploaded PyTorch JIT model
def predict_digit(input_image):
    # Resize the canvas to 28x28 and normalize pixel values
    # Normalize between 0 and 1
    input_tensor = input_image.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    input_tensor = input_tensor.to(torch.float32)  # Ensure it's in float32 for the model
    model = st.session_state.model
    # Get model prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_digit = torch.argmax(probabilities, dim=1).item()

    return predicted_digit, probabilities[0,predicted_digit]

def sample_image(gray_image):
    result_image = np.zeros((28, 28))
    for i in range(28):
        for j in range(28):
            result_image[i, j] = np.mean(gray_image[i*10:(i+1)*10, j*10:(j+1)*10])
    
    return result_image

# Main Streamlit app function
def main():
    st.title("Draw a Digit on the Canvas (Upload Your PyTorch Model)")

    # Check if a model has been uploaded
    if st.session_state.model:
        # Step 1: Create a canvas for drawing
        st.write("Draw your digit in the box below:")
        
        # Use the streamlit-drawable-canvas component to create a drawing canvas
        canvas_result = st_canvas(
            fill_color="black",  # Fill color for drawing (black on white canvas)
            stroke_width=20,      # Set the thickness of drawing
            stroke_color="white",  # Background color of the canvas
            update_streamlit=True,
            height=canvas_pixel_size,
            width=canvas_pixel_size,
            drawing_mode="freedraw",
            display_toolbar=True,  # Remove the toolbar for a clean drawing experience
            key="canvas"
        )

        # Step 2: Process the drawing for prediction
        if canvas_result.image_data is not None:
            # The canvas returns an RGBA image, so we need to process it
            canvas = np.array(canvas_result.image_data)
            canvas = cv2.GaussianBlur(canvas, (51, 51), 0)  # Apply Gaussian blur to the image
            canvas_rgb = canvas[:, :, :3]
            np.set_printoptions(threshold=np.inf)
            
            canvas_gray = np.mean(canvas_rgb, axis=2) /255.0 # Convert RGB to grayscale # Normalize pixel values to 0-1
            canvas_gray = canvas_gray[..., np.newaxis]
            canvas_gray = np.squeeze(canvas_gray)
            canvas_gray = sample_image(canvas_gray)
            
            input_image = torch.Tensor(canvas_gray)

            prediction, probability = predict_digit(input_image)
            st.success(f"Predicted Digit: {prediction}, Probability: {probability.item()*100:.2f}%")
            
            enlarged_canvas = np.kron(canvas_gray, np.ones((10, 10)))
            st.image(enlarged_canvas, width=canvas_pixel_size, caption="Your Drawing (28x28)", use_column_width=True)
            
    else:
        uploaded_file = st.file_uploader("Upload your PyTorch model", type=["pt"])
        if uploaded_file:
            st.session_state.model = torch.jit.load(io.BytesIO(uploaded_file.read()))
            st.rerun()  # Reload the app to show the canvas

if __name__ == "__main__":
    main()
