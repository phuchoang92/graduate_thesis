import cv2
import streamlit as st
from tempfile import NamedTemporaryFile
from pipeline import pipeline


st.set_page_config(
    page_title="Demo Action Monitoring",
)

st.sidebar.header("Home Page")


def read_video(video_file):
    video_capture = cv2.VideoCapture(video_file.name)

    ret, frame = video_capture.read()
    st.image(frame, channels="BGR")

st.title('Activity Monitoring')
st.sidebar.title("Upload Video")

temp_file = None
video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True


st.button('Process', on_click=click_button)

if video_file is not None:
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(video_file.read())

    read_video(temp_file)

    temp_file.close()

if st.session_state.clicked:
    if temp_file is not None:
        with st.spinner('Wait for it...'):
            action_bank, save_path, data = pipeline(temp_file.name)
        st.session_state.action_bank = action_bank
        st.session_state.save_path = r"F:\graduate_thesis\results\v61_anchor-free.mp4"
        st.session_state.data = data
        st.session_state['video_uploaded'] = True
        st.success("Video Detected successfully!")
    else:
        st.warning('Please upload a video first.', icon="⚠️")

# st.session_state.save_path = r"F:\graduate_thesis\results\v61_summ.mp4"

