import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from model import VAE_NET
from pathlib import Path

encoder = VAE_Encoder()
decoder = VAE_Decoder()

# Create VAE_NET instance
vae_net = VAE_NET(encoder, decoder)

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def load_model(model, model_path):
    # Load the model's state_dict
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, image):
    # Apply transformations to the input image
    image_tensor = transform(image).unsqueeze(0).to('cpu')
    # Make predictions using the model
    with torch.no_grad():
        reconstructed_image_, _, _ = model(image_tensor)
    reconstructed_image = reconstructed_image_.squeeze(0).cpu()  # Move to CPU and remove batch dimension
    reconstructed_image = transforms.ToPILImage()(reconstructed_image)
    return reconstructed_image

def main():
    st.title("Simple VAE application")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the file name only when uploaded
        st.write("Uploaded Image:", uploaded_image.name)

        # Load the model when the predict button is pressed
        if st.button("Predict"):
            image = Image.open(uploaded_image)

            # Display the uploaded image on the left
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)

            # Load the model
            model_path = Path('quantized_VAE_weights1.pth')
            model = load_model(vae_net, model_path)

            # Make prediction
            reconstructed_image = predict_image(model, image)

            # Display the reconstructed image on the right
            with col2:
                st.image(reconstructed_image, caption="Reconstructed Image", use_column_width=True)

            # Add a reupload button to perform prediction again without refreshing
            if st.button("Reupload"):
     
                # Reset the uploaded image
                uploaded_image = None

                # Trigger a rerun of the main function
                st.experimental_rerun()

if __name__ == "__main__":
    main()
