import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Example lists of x and y values
x_values = [1, 2,  3, 4, 5, 6]
y_values = [0, -5, 0, 11, 21, 30]

# Create a smooth interpolation function
interp_func = interp1d(x_values, y_values, kind='cubic')

# Generate more points for a smoother line
x_smooth = np.linspace(min(x_values), max(x_values), 1000)
y_smooth = interp_func(x_smooth)

fig, ax = plt.subplots()

# Plotting the soft line
ax.plot(x_smooth, y_smooth, color='green', label='Smooth Line')

# Add text next to the dashed vertical line
ax.text(3, 33, 'ОКУПАЕМОСТЬ', ha='center', va='bottom', color='black', weight='bold')
ax.text(6.2, 30, 'NPV', ha='center', va='bottom', color='black', weight='bold')


# Remove the border on the top and right sides
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adding labels and title
ax.axhline(0, color='black', linestyle='--',)
ax.axvline(3, color='black', linestyle='--',)

# Display the plot
plt.show()
