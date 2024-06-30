import streamlit as st
import pandas as pd
import numpy as np

st.title('Activity Monitoring')
st.file_uploader("Choose video",
                 type=None,
                 accept_multiple_files=False,
                 key=None, help=None,
                 on_change=None, args=None,
                 kwargs=None,
                 disabled=False, label_visibility="visible")

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

st.button('Process', on_click=click_button)


if st.session_state.clicked:
    # The message and nested widget will remain on the page
    st.write('Process start')
    st.slider('Select a value')


# def display_input_row(index):
#     left, middle, right = st.columns(3)
#     left.text_input('First', key=f'first_{index}')
#     middle.text_input('Middle', key=f'middle_{index}')
#     right.text_input('Last', key=f'last_{index}')
#
# if 'rows' not in st.session_state:
#     st.session_state['rows'] = 0
#
# def increase_rows():
#     st.session_state['rows'] += 1
#
# st.button('Add person', on_click=increase_rows)
#
# for i in range(st.session_state['rows']):
#     display_input_row(i)
#
# # Show the results
# st.subheader('People')
# for i in range(st.session_state['rows']):
#     st.write(
#         f'Person {i+1}:',
#         st.session_state[f'first_{i}'],
#         st.session_state[f'middle_{i}'],
#         st.session_state[f'last_{i}']
#     )