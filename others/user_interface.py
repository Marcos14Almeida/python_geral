import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def show_dataframe():
    # Destroy the previous plot, if it exists
    if hasattr(root, 'canvas'):
        root.canvas.get_tk_widget().destroy()

    # Define the size of the DataFrame
    rows = 5
    cols = 3

    # Create a random NumPy array
    data = np.random.randn(rows, cols)

    # Create the DataFrame
    df = pd.DataFrame(data, columns=['A', 'B', 'C'])

    # Update the text widget with the DataFrame
    text_widget.delete(1.0, tk.END)  # Clear previous content
    text_widget.insert(tk.END, df.to_string())


def plot_data():
    # Destroy the previous plot, if it exists
    if hasattr(root, 'canvas'):
        root.canvas.get_tk_widget().destroy()

    # Create the plot
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]
    ax.plot(x, y)

    # Create a canvas to display the plot
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Store the canvas object in the root
    root.canvas = canvas


def select_file():
    file_path = filedialog.askopenfilename()
    print("Selected file:", file_path)


def close_window():
    root.destroy()


root = tk.Tk()

button_select = tk.Button(root, text="Select File", command=select_file)
button_select.pack()

button_close = tk.Button(root, text="Close", command=close_window)
button_close.pack()

button_plot = tk.Button(root, text="Plot Data", command=plot_data)
button_plot.pack()

button_plot = tk.Button(root, text="Dataframe", command=show_dataframe)
button_plot.pack()

# Create a text widget to display the DataFrame
text_widget = tk.Text(root, height=10, width=50)
text_widget.pack()

root.mainloop()
