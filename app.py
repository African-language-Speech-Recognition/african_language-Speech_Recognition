import logging
import logging.handlers
from pathlib import Path
import os
import streamlit as st
from pydub import AudioSegment
from recorder import record_to_file
#from asr_prediction import Prediction_Service
#from model_downloader import download_file
logger = logging.getLogger(__name__)

# Initialize session state for existence of model-file:
#if 'model' not in st.session_state:
   # st.session_state['model'] = False

# file+path-variables:
HERE = Path(__file__).parent
#MODEL_URL = "https://www.dropbox.com/s/w4sii4o7fa4mxds/RNN_mel2_last_vl35.6.h5?raw=1"
MODEL_LOCAL_PATH = HERE / "model/cnn-bi-rnn.h5"
RECORDED = HERE / "recordings/temp.wav"
UPLOADED = HERE / "recordings/uploaded.wav"
RESAMPLED = HERE / "recordings/resampled.wav"


SAMPLERATE = 16000

# page-config:
st.set_page_config(page_title="Speech Recognition app",
                   page_icon=":robot_face:")
st.title('Speech Recognition app')
st.markdown(
    """
This demo app is using a simplified end-to-end speech recognition engine similar to DeepSpeech2.
It was trained on the Amharic dataset and is able to recognize words and phrases from speech.
"""
)
# color "st.buttons" in main page light blue:
st.markdown("""
 <style>
 div.stButton > button:first-child {
     background-color: rgb(0, 131, 184);
 }
 </style>""", unsafe_allow_html=True)
# hide menu
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

# sidebar image
st.sidebar.image("./img/side.webp")

### functions #########################################################################
def read_audio(file):
    audio_file = open(str(file), 'rb')
    audio_bytes = audio_file.read()
    return audio_bytes

def main():
    # Download model-file if not existing and set session_state['model'] = True
    #download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=112500560) #expected_size= 112500560 337407816

    # set 3 pages to select in sidebar:
    page = st.sidebar.selectbox(
        "Choose an option:", ['Record speech', 'Open wav-file'])

    if page == 'Record speech':
        st.header('Record speech')
        record_to_file(RECORDED)
        if RECORDED.exists() == True:
            st.write("Recording found.")
            st. audio(read_audio(RECORDED), format='audio/wav')
            predict_rec = st.button('Transcribe recording')
            if predict_rec:
                prediction = ps.make_prediction(str(RECORDED))
                st.write(f"**Prediction:  '{prediction}'**")
                st.download_button(
                    "Save recorded file", read_audio(RECORDED), 'audio')

    elif page == 'Open wav-file':
        st.header('Open wav-file')
        # create upload button:
        uploaded_file = st.file_uploader("")
        if uploaded_file is not None:
            # Uploaded file is saved to the server as "uploaded.wav":
            bytes_data = uploaded_file.getvalue()
            with open(str(UPLOADED), 'wb') as f:
                f.write(bytes_data)

            # resample to 16000Hz and reduce channels to 1:
            audio_data = AudioSegment.from_wav(str(UPLOADED))
            audio_data = audio_data.set_channels(1)
            audio_data = audio_data.set_frame_rate(SAMPLERATE)
            audio_data.export(str(RESAMPLED), format="wav")

            # predict with PredictionService
            st.audio(read_audio(RESAMPLED), format='audio/wav')
            st.write("Transcribing...")
            #prediction = ps.make_prediction(str(RESAMPLED))
            st.write(f"**Prediction:  '{prediction}'**")

    else:
        # streamlit container-structure
        st.header('Examples')
        first = st.container()
        second = st.container()
        third = st.container()
  
        first.audio(read_audio(EXAMPLE_1), format='audio/wav')
        first.write("Text: 'destroy him my robots' (Impossible Mission)")
        example1 = first.button("Transcribe", key=1)

        second.markdown("""---""")
        second.audio(read_audio(EXAMPLE_2), format='audio/wav')
        second.write(
            "Text: 'there was a long hush for no single wolf cared to fight' (The Jungle Book)")
        example2 = second.button("Transcribe", key=2)

        third.markdown("""---""")
        third.audio(read_audio(EXAMPLE_3), format='audio/wav')
        third.write(
            "Text: 'and he grew very tired of saying the same thing over a hundred times' (The Jungle Book)")
        example3 = third.button("Transcribe", key=3)

        if example1:
            first.write(f"**Prediction:  '{ps.make_prediction(str(EXAMPLE_1))}'**")
        if example2:
            second.write(f"**Prediction:  '{ps.make_prediction(str(EXAMPLE_2))}'**")
        if example3:
            third.write(f"**Prediction:  '{ps.make_prediction(str(EXAMPLE_3))}'**")

#########################################################################
if __name__ == '__main__':
    DEBUG = os.environ.get("DEBUG", "false").lower() not in [
        "false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )
    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)
    main()
    
    #if st.session_state['model'] == False:
       #ps = Prediction_Service()
      # main()
