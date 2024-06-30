import csv
import matplotlib.pyplot as plt
import numpy as np

def bar_show():
    csv_file = 'action_bank1.csv'

    data = []

    with open(csv_file, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')

        for row in csv_reader:
            data.append(row)

    data = np.array(data)

    data_to_plot = data[:, 1:]

    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    categories = ['use smartphones', 'sleep', 'eat/drink', 'communicate', 'walk', 'leave seat']


    fig, ax = plt.subplots(figsize=(10, 5))

    num_bars = data_to_plot.shape[1]

    bar_width = 0.8 / num_bars

    bar_positions = np.arange(data_to_plot.shape[0])

    for i in range(num_bars):
        ax.bar(bar_positions + i * bar_width, data_to_plot[:, i], width=bar_width, color=colors[i], label=categories[i])

    ax.set_xlabel('Data Points')
    ax.set_ylabel('Values')
    ax.set_title('Activity Distribution')

    ax.set_xticks(bar_positions + (num_bars - 1) * bar_width / 2)
    ax.set_xticklabels(['Data {}'.format(i+1) for i in range(data_to_plot.shape[0])])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig('activity_plot.png', dpi=300)  # Adjust dpi as needed
    plt.show()


def line_graph():


    # Your data
    csv_file = 'action_bank1.csv'

    data = []

    with open(csv_file, 'r', newline='') as file:
        csv_reader = csv.reader(file, delimiter=',')

        for row in csv_reader:
            data.append(row)

    # Convert data to numpy array for easier manipulation
    data = np.array(data)

    data = data[:, 1:]

    # Number of columns (points in each line)
    num_columns = data.shape[1]
    categories = ['use smartphones', 'sleep', 'eat/drink', 'communicate', 'walk', 'leave seat']
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each element across rows as a separate line
    for i in range(num_columns):
        ax.plot(range(1, data.shape[0] + 1), data[:, i], marker='o', label=categories[i])

    # Set labels and title
    ax.set_xlabel('Lines')
    ax.set_ylabel('Values')
    ax.set_title('Line Graph of Elements across Rows')

    # Set x-axis ticks and labels
    ax.set_xticks(range(1, data.shape[0] + 1))
    ax.set_xticklabels([f'Line {i + 1}' for i in range(data.shape[0])])

    # Add legend
    ax.legend()

    # Remove the surrounding box (frame)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Remove the grid
    ax.grid(False)

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.savefig('lines_from_rows_graph.png', dpi=300)  # Adjust dpi as needed
    plt.show()


line_graph()