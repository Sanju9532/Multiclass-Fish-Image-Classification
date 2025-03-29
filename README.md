Multiclass Fish Image Classification

Data Preprocessing:
    Images are loaded and rescaled to the range of 0 to 1. They are then randomly rotated, shifted 
horizontally and vertically, zoomed, and flipped. Missing pixels resulting from these transformations 
are filled. Finally, the image size is specified.

Model Training:
    Training various models, including CNN, VGG16, ResNet, MobileNet, and InceptionV3, resulted in MobileNet
demonstrating superior performance with an accuracy of 0.9926 and a loss of 0.0250. This trained MobileNet model 
was then saved to a pickle file.

Model Evaluation:
      Metrics, including accuracy, precision, recall, and F1-score, were compared across all models. 
Additionally, the training history (accuracy and loss) for each model was visualized.

Deployment in Streamlit:
      Build a Streamlit application that allows users to upload fish images, 
it predict  the class and displays the fish category, and confidence scores.
