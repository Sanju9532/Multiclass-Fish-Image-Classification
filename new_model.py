import streamlit as st
from PIL import Image
import joblib
import numpy as np
import matplotlib.pyplot as plt

page = st.sidebar.radio('Navagition',['Home','visualization'])
if page == 'Home':
    st.markdown('<h1 style = "color:blue;"> MULTICLASS FISH IMAGE CLASSIFICATION</h1>',unsafe_allow_html=True)

    # Load your model
    model = joblib.load("C:/Users/megal/.conda/envs/myds/mobilenet_model.pkl")

    #defining class names
    class_names = [
        'fish sea_food trout',
        'fish sea_food striped_red_mullet',
        'fish sea_food shrimp',
        'fish sea_food sea_bass',
        'fish sea_food red_sea_bream',
        'fish sea_food red_mullet',
        'fish sea_food hourse_mackerel',
        'fish sea_food gilt_head_bream',
        'fish sea_food black_sea_sprat',
        'animal fish',
        'animal fish bass'
    ]

    st.markdown('<h2 style = "color:black;"> Upload An Image Of Fish To Classify Its Species </h2>',unsafe_allow_html = True)

    upload_image = st.file_uploader("**Choose An Image**", type=['jpg', 'jpeg', 'png'])

    if upload_image is not None:
        st.image(upload_image)
        a = st.button("**Click**")
        if a:
            image = Image.open(upload_image)
            # Resize the image to the expected input size
            image = image.resize((150,150))  # Resize to 150x150 pixels
            # Convert the image to a numpy array
            image_array = np.array(image)
            # Expand dimensions to match the input shape (1, 150, 150, 3)
            image_array = np.expand_dims(image_array, axis=0)
            # Rescale pixel values to [0,1]
            image_array = image_array / 255.0
            # Now you can use preprocessed_image for prediction
            prediction = model.predict(image_array)
            #st.write(prediction)  # Display the prediction
            #image_generator = data.flow(image_array,batch_size = 1)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

            a = st.write(f'Prediction : { predicted_class}')
            st.write(f'Confidence:{confidence:.2f}')

            
            st.success("Its Predeicted Correctly")
            st.balloons()

if page == 'visualization':
    st.markdown('<h2 style = "color:red;"> Accuracy And Loss Visualization </h2>',unsafe_allow_html = True)
    graph = st.selectbox('**Select Vlisual**',['Accuracy','Loss'])
    if graph == 'Accuracy':
        #Accuracy plot
        x = ['CNN', 'VGG16', 'ResNet', 'MobileNet', 'InceptionV3'] 
        y = [0.9319, 0.4226, 0.3324, 0.9856, 0.9583] 
        fig,ax = plt.subplots(figsize = (5,3))
        ax.bar(x,y)
        ax.set_title('Accuracy Graph')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy Score')
        st.pyplot(fig)

    if graph == 'Loss':
        #loss plot
        x = ['CNN','VGG16','ResNet','MobileNet','InceptionV3']
        y = [0.2915,0.4226,1.8524,0.0250,0.1277]

        fig,ax = plt.subplots(figsize = (5,3))
        ax.plot(x,y)
        ax.set_title('Loss Graph')
        ax.set_xlabel('Model')
        ax.set_ylabel('Loss_score')

        st.pyplot(fig)

       
        
        