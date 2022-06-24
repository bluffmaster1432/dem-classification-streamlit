"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
import torch
from torchvision import models, transforms
import streamlit as st
from PIL import Image
from model_config import get_model, get_model_8class

# set title of app
st.title("Dermoscopy Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = ["jpg","jpeg","png","bmp"])


def load_model():
    return get_model()

def load_model_multiclass():
    return get_model_8class()

resnet = load_model()
# model_multiclass = load_model_multiclass()


def predict(image):
    """Return top 5 predictions ranked by highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a ResNet model
    #resnet = get_model() #models.resnet101(pretrained = True)
    #torch.save(resnet.state_dict(),'model.pt', _use_new_zipfile_serialization=False)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image).convert('RGB')
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)[0]

    # model_multiclass.eval()
    # out_multiclass = model_multiclass(batch_t)[0]
    # print(out)

    classes = ['Benign', 'Malignant']
    classes_multiclass = ['ak', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

    # return the top 5 predictions ranked by highest probabilities
    # prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    prob = torch.nn.functional.softmax(out) * 100
    _, indices = torch.sort(out, descending = True)

    # prob_multiclass = torch.nn.functional.softmax(out_multiclass) * 100
    # _, indices_multiclass = torch.sort(out_multiclass, descending = True)
    # print(indices)
    # return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

    return [(classes[idx], prob[idx].item()) for idx in indices[:5]] #, [(classes_multiclass[idx], prob_multiclass[idx].item()) for idx in indices_multiclass[:5]]


if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up).convert('RGB')
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
    # st.info("Subclass result")
    # for i in labels_multiclass:
    #     st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
