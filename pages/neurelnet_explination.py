import streamlit as st

video_file = open('videos\\NeurelNet.mp4', 'rb')
video_bytes = video_file.read()

st.header("Nuerel Network Explanation")

st.subheader("A quick explanation of a densly connected neurel network.")

st.markdown("The process of training and making predictions with dense networks is complicated. This is a quick visual of some of the math:")
st.video(video_bytes)
st.write("Each line here represents a weight and each neuron has a value. The current neuron is calculated by completing a summation of all previous neurons multiplied by all the corresponding weights (shown by the feedforward function).")
st.write("This becomes pretty massive when you take into account the size of each layer of the network. Most of the time they do different things, it all really depends on the task at hand.")