import streamlit as st
import matplotlib.pyplot as plt
import torch
from PIL import Image
from helpers import *
import concurrent.futures



st.set_page_config(layout="wide")
st.title("\U0001F600Stable Diffusion Model Comparer\U0001F600")

st.write("Welcome to the Stable Diffusion Comparer! This tool will let you tweak the output of two stable diffusion models and compare the results")
st.write("You will need to make 4 choices:")
st.write("1. Which stable diffusion model to use. Currently there are two choices due to memory constraints.")
st.write("2. What image you want the models to generate. This will be your 'prompt'.")
st.write("3. The number of inference steps. Inference steps are the amount of steps the model takes to denoisify and produce the image.")
st.write("4. The guidance scale. This determines how much the model adheres to the prompt when generating the image.")


col1, col2 = st.columns(2)

models = ["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5"]

with col1:
    option1 = st.selectbox(
        'Select a model',
        models,
        key= "box1")
    inference_steps1= st.slider("Select the number of inference steps to take", min_value=5, max_value=50, value=50, step=5, key= "slide1")
    guidance_scale1 = st.slider("Set the guidance scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5, key= "slide2")
    
    
with col2:
    option2 = st.selectbox(
        'Select a model',
        models,
        key='box2')
    inference_steps2= st.slider("Select the number of inference steps to take", min_value=5, max_value=50, value=50, step=5, key= "slide3")
    guidance_scale2 = st.slider("Set the guidance scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5, key= "slide4")
    

prompt = st.text_input('Please enter your prompt')
col3, col4 = st.columns(2)
if st.button('Generate'):
    
    with st.spinner("Generating images! This may take several minutes..."):
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(generate_image, option1, prompt, inference_steps1, guidance_scale1)
            future2 = executor.submit(generate_image, option2, prompt, inference_steps2, guidance_scale2)

            img1, intermediate1 = future1.result()
            img2, intermediate2  = future2.result()
            
            fig1 = build_grid(intermediate1)
            fig2 = build_grid(intermediate2)

            with col3:
                st.image(img1, caption=option1)
                st.pyplot(fig1)

            with col4:
                st.image(img2, caption=option2)
                st.pyplot(fig2)
                


        
