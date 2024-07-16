import streamlit as st
import plotly.graph_objects as go

video_path = r"D:\results\video6_edit.mp4"

data = [True, False, False, False, False, True, False, False, False, True, True, True, True, True, True,
        False, False, False, True, True, True, True, True, True, True, True, True, False, False,
        False, False, False, True, True, True, True, True, True, True, True, True, False, False,
        False, False, False, False, False, True, True, True, True, True, True, False, False, False,
        False, False, False, False, False, False, False, False, False, False, False, False, True,
        False, True, True, True, True, True, True, True, True, True, True]

def find_true_ranges(data):
    result = []
    start = None

    for i, value in enumerate(data):
        if value and start is None:
            start = i
        elif not value and start is not None:
            result.append((start, i - 1))
            start = None

    if start is not None:
        result.append((start, len(data) - 1))

    return result

ranges = find_true_ranges(data)

time_labels = []
for i in range(0, len(data), 10):
    minutes = i // 60
    seconds = i % 60
    time_labels.append(f'{minutes}m{seconds}s')

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[0, len(data)-1, len(data)-1, 0, 0],
    y=[0, 0, 1, 1, 0],
    fill='toself',
    fillcolor='lightgrey',
    line=dict(color='black', width=2),
    mode='lines',
    hoverinfo='none'
))

for i, (start, end) in enumerate(ranges):
    fig.add_trace(go.Scatter(
        x=[start, end, end, start, start],
        y=[0, 0, 1, 1, 0],
        fill='toself',
        fillcolor='red',
        line=dict(width=0),
        mode='lines',
        hoverinfo='text',
        hovertext=f'Start: {start}<br>End: {end}'
    ))

fig.update_layout(
    title='Segments of Video',
    xaxis_title='Time',
    xaxis=dict(
        tickmode='array',
        tickvals=list(range(0, len(data), 10)),
        ticktext=time_labels,
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        range=[0, 1]
    ),
    showlegend=False,
    height=150,
    margin=dict(l=0, r=0, t=30, b=0)
)

st.title("Visualization Time Ranges")

clicked_point = st.plotly_chart(fig)

width = 60
side = max((100 - width) / 2, 0.01)

_, container, _ = st.columns([side, width, side])
container.video(data=video_path)