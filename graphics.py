import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


# ----------------------------------------------------------------------------------
# Define your custom theme
theme_black = {
    'figure.facecolor': '#2b2d42',      # Background color of the figure
    'axes.facecolor': '#2b2d42',        # Background color of the axes
    'axes.edgecolor': 'white',        # Color of the axes edges
    'axes.labelcolor': 'white',       # Color of the axes labels
    'xtick.color': 'white',           # Color of the x-axis tick labels
    'ytick.color': 'white',           # Color of the y-axis tick labels
    'text.color': 'white',            # Color of the text
    'grid.color': 'gray',             # Color of the grid lines
    'grid.linestyle': ':',            # Style of the grid lines
    'lines.color': 'hotpink',         # Color of the lines
    'patch.edgecolor': 'white',       # Color of the patches
    'patch.facecolor': 'hotpink',     # Background color of the patches
    'axes.titlepad': 20,              # Padding of the title from the axes
    'axes.titlesize': 'x-large',      # Font size of the title
    'axes.titleweight': 'bold',       # Font weight of the title
    'axes.labelsize': 'large',        # Font size of the axes labels
    'font.size': 12,                  # Base font size
}
custom_theme = {
    'figure.facecolor': '#edf6f9',      # Background color of the figure
    'axes.facecolor': '#edf6f9',        # Background color of the axes
    'axes.edgecolor': '#2b2d42',        # Color of the axes edges
    'axes.labelcolor': '#2b2d42',       # Color of the axes labels
    'xtick.color': '#2b2d42',           # Color of the x-axis tick labels
    'ytick.color': '#2b2d42',           # Color of the y-axis tick labels
    'text.color': '#2b2d42',            # Color of the text
    'grid.color': 'gray',             # Color of the grid lines
    'grid.linestyle': ':',            # Style of the grid lines
    'lines.color': 'hotpink',         # Color of the lines
    'patch.edgecolor': '#2b2d42',       # Color of the patches
    'patch.facecolor': 'hotpink',     # Background color of the patches
    'axes.titlepad': 20,              # Padding of the title from the axes
    'axes.titlesize': 'x-large',      # Font size of the title
    'axes.titleweight': 'bold',       # Font weight of the title
    'axes.labelsize': 'large',        # Font size of the axes labels
    'font.size': 12,                  # Base font size
}
# ----------------------------------------------------------------------------------


def create_boxplot(data, title, layout):
    fig, ax = plt.subplots()
    ax.boxplot(data)
    ax.set_title(title)
    ax.set_xlabel(layout['xlabel'])
    ax.set_ylabel(layout['ylabel'])
    ax.set_xticklabels(layout['xticklabels'])
    plt.show()
# ----------------------------------------------------------------------------------


def create_line_plot(data, x, y, title, line_color='hotpink', text_color='black'):

    # Set the plot background color
    plt.style.use(custom_theme)

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the line with custom color
    sns.lineplot(x=x, y=y, data=data, ax=ax, color=line_color)

    # Set the title
    ax.set_title(title, color=text_color)

    # Set the axis labels color
    ax.set_xlabel(x, color=text_color)
    ax.set_ylabel(y, color=text_color)

    # Show the plot
    plt.show()

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------


# Example usage
data = [1, 2, 3, 4, 5]
title = "Boxplot Example"
layout = {
    'xlabel': "X-axis",
    'ylabel': "Y-axis",
    'xticklabels': ["Data"]
}
# create_boxplot(data, title, layout)


# Example usage
# Generate random data
x = np.linspace(-10, 10, 100)  # Create an array of 100 points from -10 to 10
y = np.random.rand(100) * np.sin(x)  # Generate random values multiplied by sine of x

data = {'Year': x,
        'Sales': y}
df = pd.DataFrame(data)

title = 'Sales Over Time'
x = 'Year'
y = 'Sales'
create_line_plot(df, x, y, title)
