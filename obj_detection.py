import streamlit as st
import cv2
import os
import sys 
import numpy as np
import tempfile
import math
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing import image
from skimage.transform import resize
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Sequential
#from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense,InputLayer
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions


# Load the model
model = VGG16(weights='imagenet')
#model =InceptionV3(include_top=True)


PAGE_CONFIG = {"page_title":"StColab.io", "page_icon":":smiley:","layout":"centered"}
st.set_page_config("Main Menu")

st.title("Object Detection with InceptionV3")
st.text("Choose UPLOAD on the dropdown menu to upload your own video and search for objects")

# Play video whilist the predictions are loading
video_file = open('./uploads/try.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

page = st.sidebar.selectbox("Choose action", ["Classify", "Upload"]) 
def main():
  st.sidebar.title("Dashboard")


  if page == "Classify":
    #Load a video from file
    video_input_file_path = ('./uploads/try.mp4')

   

    #Output file
    feature_output_file_path = ('./Frames/')

    # Capturing the video from path
    video = cv2.VideoCapture(video_input_file_path)
    frameRate = video.get(5)

    #Creating frames from the given video
    import math
    count = 0
    while(video.isOpened()):
        frameNum = video.get(1)
        ret, frame = video.read()
        if (ret != True):
            break
        if (frameNum % math.floor(frameRate) == 0):
            frameName = feature_output_file_path + "frame%d.jpg" % count;count+=1
            cv2.imwrite(frameName, frame)
    video.release()
    st.text("Frame Capturing complete!")

    frames = './Frames/*.jpg'

    # Converting the images to arrays
    images = []
    import glob
    for filename in glob.glob(frames): 
      frame = image.load_img(filename, target_size=(224,224,224)) 
      images.append(frame)

    
    for frame in images:
      frame_arr = image.img_to_array(frame)
    
    #frame_arr.shape
    frame_arr = np.expand_dims(frame_arr, axis = 0)

    # prepare the image for the VGG model
    img = preprocess_input(frame_arr)

    # predict the probability across all output classes
    yhat = model.predict(img)

    # convert the probabilities to class labels
    from keras.applications.vgg16 import decode_predictions
    label = decode_predictions(yhat)

    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]


    # print the classification
    with st.container():
      col1, col2 = st.columns([5,5])
      with col1:
        st.text("Object with highest classfication")
        st.write('%s (%.2f%%)' % (label[1], label[2]*100))
      with col2:
        st.image(yhat)

  ########################################################
  if page == "Upload":
    # If the user decides to upload own video
    st.text("Upload to search for objects in the video")
    def uploadFile():
      vid_file = st.file_uploader("Upload a video", type=["2mb only","mp4", "mov","avi"])
      tempVideo = tempfile.NamedTemporaryFile(delete=False) 
      if vid_file is not None: 
        tempVideo.write(vid_file.read())
      return tempVideo.name
    # Capturing video frames 
    def splitVideo(videoPath):
      import math
      count = 0
      cap = cv2.VideoCapture(videoPath)
      frameRate = cap.get(5) 
      tempImage = tempfile.NamedTemporaryFile(delete=False) 
      x=1
      # Splitting video frames into photos
      while(cap.isOpened()):
        frameId = cap.get(1) 
        ret, frame = cap.read()
        if (ret != True):
          break
        if (frameId % math.floor(frameRate) == 0):
          tempImage = videoPath.split('.')[0] +"_frame%d.jpg" % count;count+=1
          cv2.imwrite(tempImage, frame)
          frames.append(tempImage)
      cap.release() 
      return frames,count
    # Classifying the objects
    def classifyObjects():  
      model = VGG16()
      from keras.applications.vgg16 import decode_predictions
      classify = []
      frames,count = splitVideo(videoFile)

      for i in range(count):    
        image = load_img(frames[i], target_size=(224, 224)) 
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        # predict the probability across all output classes   
        img_pred = model.predict(image)
        # convert the probabilities to class labels
        label = decode_predictions(img_pred)    
        label = label[0][0]
        result =  label[1]
        classify.append(result)
      return classify

    def searchInFrames(object_):
      indeces = []
      classifications = classifyObjects()
      if object_ in classifications:
        for i in range(len(classifications)):
          if classifications[i] == object_:
            index = classifications.index(object_)
            indeces.append(index)
            filePath = frames[index]
            img = load_img(filePath, target_size = (224, 224))
            detected_paths.append(filePath)
        for i in range(len(indeces)):
          st.image(frames[i], width=224)
      else:
        st.write("Object not available in video!")

    videoFile = uploadFile()
    user_input = st.text_input("Enter object to search: ")

    if st.button('Search'):  
      frames =[]
      detected_paths = []
      searchInFrames(user_input)
      st.write("")


if __name__ == '__main__':
  main()
