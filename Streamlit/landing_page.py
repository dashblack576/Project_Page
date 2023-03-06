import streamlit as st

video_file = open('C:\\Users\\dashb\\Documents\\Streamlit\\videos\\LandingPage.mp4', 'rb')
video_bytes = video_file.read()


st.title('Welcome to the Dash Black machine learning webpage.') 
st.subheader('In this space you\'ll find everything machine learning.')

st.video(video_bytes)

st.write("My name is dash black and I've been doing computer science for about six yaers. I'm trying to keep all my projects in one place, where all of these projects can be easily found and run!")
st.write("To make any one project work, the github project will be linked. I will also have the Read Me avilable on the page to make things more easily accessible.")
st.write("What will happen: a page will appear for a given project. If the project is interactive you will need to download the github and all requirements. One downloaded just copy and paste the files with a given name inside of this streamlit folder. Once that's done you should be good! If the project isn't interactive all of the code can be read from that page, and github will still be linked.")
st.write("If you'd like an example, find your way to the summerization page.")