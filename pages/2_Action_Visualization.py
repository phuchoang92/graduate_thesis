import json
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Class names mapping
class_names = {
    0: "normal actions",
    1: "use smartphones",
    2: "sleep",
    3: "eat/drink",
    4: "communicate",
    5: "walk",
    6: "leave seat"
}

# Load action data from JSON
with open("action_data.json") as f:
    action_dict = json.load(f)

# Initialize variables
min_frame = float('inf')
max_frame = float('-inf')
person_trajectories = {}

# Process action data
for frame, actions in action_dict.items():
    frame_num = int(frame)
    min_frame = min(min_frame, frame_num)
    max_frame = max(max_frame, frame_num)

    for trajectory in actions:
        person_id, bounding_box, action = trajectory.split(', ')
        person_id = f"person_{person_id}"
        information = {"frame_id": frame_num, "bbox": bounding_box, "action": int(action)}

        if person_id not in person_trajectories:
            person_trajectories[person_id] = []
        person_trajectories[person_id].append(information)

data = person_trajectories

# Group frames with the same action
def group_continual_frames(person_data):
    grouped_frames = []
    current_group = []

    for i, frame in enumerate(person_data):
        if i == 0 or frame['action'] != person_data[i - 1]['action']:
            if current_group:
                grouped_frames.append({
                    'action': current_group[0]['action'],
                    'start_frame_id': current_group[0]['frame_id'],
                    'end_frame_id': current_group[-1]['frame_id'],
                    'frames': current_group
                })
                current_group = []
        current_group.append(frame)

    if current_group:
        grouped_frames.append({
            'action': current_group[0]['action'],
            'start_frame_id': current_group[0]['frame_id'],
            'end_frame_id': current_group[-1]['frame_id'],
            'frames': current_group
        })

    return grouped_frames

# Create timeline plot
def create_timeline_plot(grouped_frames, start_sec, end_sec):
    fig = go.Figure()
    unique_actions = sorted(set(group['action'] for group in grouped_frames))
    colormap = {
        0: '#808080',  # gray
        1: '#0000FF',  # blue
        2: '#008000',  # green
        3: '#FF0000',  # red
        4: '#800080',  # purple
        5: '#FFA500',  # orange
        6: '#00FFFF',  # cyan
    }

    for action in unique_actions:
        color = colormap.get(action, 'gray')
        action_frames = [group for group in grouped_frames if group['action'] == action]

        for group in action_frames:
            if start_sec * 16 <= group['start_frame_id'] <= end_sec * 16:
                start_time_sec = group['start_frame_id'] // 16
                end_time_sec = group['end_frame_id'] // 16
                start_time = timedelta(seconds=start_time_sec)
                end_time = timedelta(seconds=end_time_sec)

                fig.add_trace(go.Scatter(x=[start_time_sec, end_time_sec],
                                         y=[action, action],
                                         mode='lines',
                                         name='',
                                         line=dict(color=color, width=20),
                                         hovertemplate=f"{class_names[action]}<br>Time: {start_time} to {end_time}"))

    fig.update_layout(title='Timeline of Actions',
                      xaxis_title='Time',
                      yaxis_title='Action',
                      showlegend=False,
                      hovermode='x unified',
                      xaxis=dict(
                          tickmode='array',
                          tickvals=[i * 60 for i in range(int((end_sec - start_sec) // 60) + 2)],
                          ticktext=[str(timedelta(seconds=i * 60)) for i in range(int((end_sec - start_sec) // 60) + 2)]
                      ),
                      yaxis=dict(
                          tickvals=list(class_names.keys()),
                          ticktext=list(class_names.values()),
                          range=[-0.5, max(unique_actions) + 0.5])
                      )

    return fig

# Streamlit app
st.title('Timeline Visualization of Actions')

person_options = list(data.keys())
selected_person = st.selectbox('Select a person:', person_options)

if selected_person:
    st.subheader(f'Timeline for {selected_person}')
    grouped_frames = group_continual_frames(data[selected_person])
    # frame_range_sec = st.slider('Time Range (seconds)', min_value=min_frame // 16, max_value=max_frame // 16,
    #                             value=(min_frame // 16, max_frame // 16), step=1)
    start_sec, end_sec = min_frame // 16, max_frame // 16 #frame_range_sec
    fig = create_timeline_plot(grouped_frames, start_sec, end_sec)
    st.plotly_chart(fig)
