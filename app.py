import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
st.title("Ship Image Classification")
st.write("Please upload ship images to the application")
upload_file = st.sidebar.file_uploader("Upload Ship Images", type = 'jpg')
generate_pred = st.sidebar.button("PREDICT")
model = tf.keras.models.load_model('Ship-shape.h5')

st.write('Ship Image Classification is the deep learning project has been used in various applications including the classification of ship targets in inland waterways for enhancing intelligent transport systems. Various researchers introduced different classification algorithms, but they still face the problems of low accuracy and misclassification of other target objects. Hence, there is still a need to do more research on solving the above problems to prevent collisions in inland waterways. In this paper, we introduce a new convolutional neural network classification algorithm capable of classifying five classes of ships, including cargo, military, carrier, cruise and tanker ships, in inland waterways. The game of deep learning ship dataset, which is a public dataset originating from Kaggle, has been used for all experiments. Initially, the five pretrained models (which are AlexNet, VGG, Inception V3 ResNet and GoogleNet) were used on the dataset in order to select the best model based on its performance. Resnet-152 achieved the best model with an accuracy of 90.56%, and AlexNet achieved a lower accuracy of 63.42%. Furthermore, Resnet-152 was improved by adding a classification block which contained two fully connected layers, followed by ReLu for learning new characteristics of our training dataset and a dropout layer to resolve the problem of a diminishing gradient.')

st.write(
   """
   Classification Dataset and Hyperparameter Setting :
   -Dataset Description
   """)      
st.text(
"""
The dataset used in this experiment is public game of deep learning ship dataset
which can be found on Kaggle . The dataset consisted of five categories of ships: 
cargo, military, cruise, carrier and tanker ships, better distinguishing the 
classification capabilities of different neural networks for inland ships and thus
reflecting the impact of inland rivers when different classification networks were 
adjusted as the backbone network. The images of the ships were taken from different 
directions, in different weather conditions, at different shooting distances and 
angles and from different international and offshore harbors. The dataset consists 
of 8932 images, and there exist both RGB images and grayscale images with different 
image pixel sizes. In this dataset, the number of samples of all types is more than 
800, which can meet the needs of model training and testing. The dataset was divided 
at a ratio of 70:30, with 70% for training and 30% for testing. 
""")
st.write(
"""
-below shows sample images existing in the dataset.
""")

#opening the image

image = Image.open('information-12-00302-g002.webp')

#displaying the image on streamlit app

st.image(image, caption='Different types of ships in Dataset')


st.write("The dataset is collected from Kaggle website. Here is the link for the [dataset](https://www.kaggle.com/datasets/arpitjain007/game-of-deep-learning-ship-datasets) . The goal of this project is to classification of what the particular ship is depending on various factors.")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image:  url(https://images.ctfassets.net/92fo1e671z6m/4tWTF9X8UVXS8kdYNAYZ10/ea07ad4db17b764868aad57611653f4c/5_Predictions_to_Shape_Supply_Chains_in_2021_blog_12-07-20.jpg?w=1600&h=800&fl=progressive&q=50&fm=jpg);
background-size: cover;
}

[data-testid="stFileUploadDropzone"]{
background-image:  url(https://www.icolorpalette.com/download/solidcolorimage/24a0ed_solid_color_background_icolorpalette.png);
background-size: cover;
}


[data-testid="stHeader"] {
background-image:  url(https://images.ctfassets.net/92fo1e671z6m/4tWTF9X8UVXS8kdYNAYZ10/ea07ad4db17b764868aad57611653f4c/5_Predictions_to_Shape_Supply_Chains_in_2021_blog_12-07-20.jpg?w=1600&h=800&fl=progressive&q=50&fm=jpg);
background-size: cover;
}

[data-testid="stMarkdownContainer"]{
background-image:  url(https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRF7q4XxGV5MiBoJTval3JTLRUGBZeqX_DPVLE6Zln7jY1YF6FtwCPXtGMuPK9rcf5y018&usqp=CAU);
background-size: cover;
}


[data-testid="stSidebar"] {
background-image:  url(https://e7.pngegg.com/pngimages/264/87/png-clipart-assorted-medieval-ships-boat-ship-computer-file-ancient-sailing-collection-building-caravel.png);
background-size: cover;

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
<style>
""" 


st.markdown(page_bg_img, unsafe_allow_html=True)



def import_n_pred(image_data, model):
    size = (128,128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    reshape = img[np.newaxis,...]
    pred = model.predict(reshape)
    return pred

if generate_pred:
    image = Image.open(upload_file)
    with st.expander('image', expanded=True):
        st.image(image, use_column_width=True)
    pred = import_n_pred(image, model)
    labels = ['Cargo ship' , 'Military ship', 'Carrier ship', 'Cruise', 'Tanker ship']
    st.title("Prediction Of Ship Image Is {}".format(labels[np.argmax(pred)]))
    
st.markdown(""" 
<style>
.stButton > button {
background-color: blue;
border-radius: 50%;
font = "sans serif"
} 
<style> 
""", 
unsafe_allow_html=True
)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

