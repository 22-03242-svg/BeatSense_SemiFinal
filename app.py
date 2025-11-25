import streamlit as st
import BeatSense_Semi_Final

st.title("BeatSense – Arrhythmia Detection Through Signal Processing")
st.write("This Streamlit app runs your full Python arrhythmia detection pipeline.")

# Run your original file exactly as-is
# Your original file must have a main() block to avoid running on import
try:
    BeatSense_Semi_Final.main()
except AttributeError:
    st.error("""
    Your file BeatSense_Semi_Final.py does not have a main() function.
    I can generate one for you. Just type: **“Create main() function for me”**
    """)
