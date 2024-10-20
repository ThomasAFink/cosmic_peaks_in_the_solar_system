import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from matplotlib.collections import LineCollection

# Mountain data with additional Martian mountains
mountains = {
    'Olympus Mons': 21900,
    'Ascraeus Mons': 18200,
    'Arsia Mons': 17700,
    'Pavonis Mons': 14100,
    'Elysium Mons': 12500,
    'Mount Everest': 8848.86,
    'K2': 8611,
    'Kangchenjunga': 8586,
    'Lhotse': 8516,
    'Makalu': 8485,
    'Cho Oyu': 8188,
    'Dhaulagiri I': 8167,
    'Manaslu': 8163,
    'Nanga Parbat': 8126,
    'Annapurna I': 8091,
    'Gasherbrum I': 8080,
    'Broad Peak': 8051,
    'Gasherbrum II': 8035,
    'Shishapangma': 8027
}

# Random seed and x values
np.random.seed(42)
x = np.linspace(0, 10, 500)

# Create the random rough terrain (cumulative random walk with noise)
rough_terrain = np.cumsum(np.random.normal(0, 2, size=x.size))

# Scale the roughness so that it doesn't dominate the mountain peaks
rough_terrain = rough_terrain / 2

# --- Modification starts here ---

# Adjust the roughness amplitude
x_transition = 4 # Transition point where roughness starts increasing
roughness_scaling = np.zeros_like(x)

# Decrease roughness from left to x_transition
indices_left = x <= x_transition
roughness_scaling[indices_left] = np.linspace(0.4, 0.15, np.sum(indices_left))

# Increase roughness from x_transition to the end
indices_right = x > x_transition
roughness_scaling[indices_right] = np.linspace(0.15, 0.3, np.sum(indices_right))

# Apply the adjusted roughness scaling
rough_terrain *= roughness_scaling

# Generate peaks without adding roughness yet
peaks_y = np.zeros_like(x)

# Mountain x positions and heights
mountain_x_positions = np.linspace(1, 9, len(mountains))
mountain_heights = np.array(list(mountains.values()))

# Scale the heights to fit the curve range
scaled_mountain_heights = mountain_heights / 1000  # Convert to km scale

# Add Gaussian-like peaks for each mountain
peak_width = 0.145  # Width of the peaks
for i, (mountain_name, height) in enumerate(mountains.items()):
    peak_height = scaled_mountain_heights[i]
    peak_position = mountain_x_positions[i]

    # Create a Gaussian peak for each mountain and add to peaks_y
    peaks_y += peak_height * np.exp(-((x - peak_position) ** 2) / (2 * peak_width ** 2))

# Add a peak at (0, 0)
peak_height_at_zero = 7  # Height of the peak at x=0
peak_width_at_zero = 1   # Width of the peak at x=0
peaks_y += peak_height_at_zero * np.exp(-((x - 0) ** 2) / (2 * peak_width_at_zero ** 2))

# Ensure a minimum height between x = 0 and x = 1
min_height = 7  # Minimum height between x = 0 and x = 1
indices = (x >= 0) & (x <= 1)
peaks_y[indices] = np.maximum(peaks_y[indices], min_height)

# Get the indices of the peak positions
peak_indices = [np.abs(x - px).argmin() for px in mountain_x_positions]

# Get the cumulative heights at the peak positions
cumulative_heights = peaks_y[peak_indices]

# Ensure the cumulative heights decrease from left to right
for i in range(1, len(cumulative_heights)):
    if cumulative_heights[i] > cumulative_heights[i - 1]:
        cumulative_heights[i] = cumulative_heights[i - 1]

# Create an envelope function by interpolating between the adjusted cumulative heights
envelope_func = interp1d(
    mountain_x_positions, cumulative_heights, kind='linear', fill_value='extrapolate'
)
envelope_y = envelope_func(x)

# Ensure that the envelope is not below the peaks
envelope_y = np.maximum(envelope_y, peaks_y)

# Calculate the available space between peaks_y and envelope_y
available_space = envelope_y - peaks_y

# Avoid division by zero
epsilon = 1e-6

# Scale the roughness so that it doesn't exceed the available space
scaling_factor = np.minimum(1, available_space / (np.abs(rough_terrain) + epsilon))

# Apply the scaling factor to the rough terrain
adjusted_rough_terrain = rough_terrain * scaling_factor

# Combine peaks and adjusted rough terrain
y = peaks_y + adjusted_rough_terrain

# --- New Modification to Avoid Flat Tops ---

# Identify where the terrain exceeds the local peaks
exceeds_peak = y >= peaks_y

# Generate a small random amount to subtract from the peak height at those points
delta = np.random.uniform(0.01, 0.05, size=y.shape)

# Adjust the terrain to be slightly below the local peaks
y[exceeds_peak] = peaks_y[exceeds_peak] - delta[exceeds_peak]

# --- Modification ends here ---

# Define a custom colormap with red dominating on the left and gray transitioning after x = 3
colors = [
    (231/255, 125/255, 17/255),  # Custom color #e77d11
    (231/255, 125/255, 17/255),  # Custom color to the transition
    (0.5, 0.5, 0.5),             # Gray
    (0.5, 0.5, 0.5),             # Gray continues
]

# Create the colormap with the transition at x = 3
cmap = LinearSegmentedColormap.from_list("red_to_gray", colors, N=500)

# Normalize the x values for the gradient transition
x_norm_transition = 5  # Transition point for colormap
x_normalized = np.clip((x - 0) / x_norm_transition, 0, 1)

# --- New Modifications Start Here ---

# Create an array for the threshold heights
threshold_y = np.full_like(x, 7.5)  # Default threshold is 7.5 km for Earth mountains

# Identify Martian mountains
martian_mountain_names = ['Olympus Mons', 'Ascraeus Mons', 'Arsia Mons', 'Pavonis Mons', 'Elysium Mons']

# Set threshold to 15 km for Martian mountains
for i, mountain_name in enumerate(mountains.keys()):
    if mountain_name in martian_mountain_names:
        mountain_x = mountain_x_positions[i]
        # Define a range around each Martian mountain to apply 15 km threshold
        mountain_indices = np.abs(x - mountain_x) < 0.4
        threshold_y[mountain_indices] = 15.0  # Set threshold to 15 km

# --- New Modifications End Here ---

# Compute y_lower (minimum of y and threshold_y) and y_upper
y_lower = np.minimum(y, threshold_y)
y_upper = y

# Create the plot without setting the figure facecolor
plt.figure(figsize=(16, 8))

# Get current axes
ax = plt.gca()

# Create the segments for the line
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a LineCollection
line_segments = LineCollection(segments, linewidths=1)

# Set the colors of the line segments
line_segments.set_color(cmap(x_normalized[:-1]))

# Add the LineCollection to the plot
ax.add_collection(line_segments)

# Get the minimum y value for plotting
y_min = min(y) - 2

# Plot the terrain with the gradient fill and white fill above thresholds
for i in range(len(x) - 1):
    # x segment
    x_segment = x[i:i + 2]
    # y values for the segment
    y_segment_lower = y_lower[i:i + 2]
    y_segment_upper = y_upper[i:i + 2]
    # Gradient color for the segment
    color_segment = cmap(x_normalized[i])
    # Fill between y_min and y_lower with gradient color
    plt.fill_between(
        x_segment,
        [y_min, y_min],
        y_segment_lower,
        color=color_segment
    )
    # If y_upper > y_lower, fill between y_lower and y_upper with white
    if np.any(y_segment_upper > y_segment_lower):
        plt.fill_between(
            x_segment,
            y_segment_lower,
            y_segment_upper,
            color='white'
        )

# Add mountain labels with heights in meters
for i, (mountain_name, height) in enumerate(mountains.items()):
    label_y_position = peaks_y[peak_indices[i]] + 0.75

    # Adjust the label position for Martian mountains to avoid overlap
    if mountain_name in martian_mountain_names:
        label_y_position -= 9.0  # Slightly higher for visibility

    # Add height in meters to the label
    plt.text(
        mountain_x_positions[i],
        label_y_position,
        f'{mountain_name} ({int(height)} m)',
        horizontalalignment='center',
        color='black',
        fontsize=8,
        rotation=90,
    )

# Place the title inside the plot area
ax.text(
    0.5,
    0.95,
    'Tallest Mountains on Earth Over 8000m and Martian Volcanoes',
    transform=ax.transAxes,
    horizontalalignment='center',
    verticalalignment='top',
    fontsize=16,
    color='black'
)

# Set y-axis minimum to 0 and x-axis limits
plt.ylim(bottom=0)
plt.xlim(left=0, right=10)

# Remove spines and ticks but keep the axes background
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

ax.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False,
    labelleft=False,
    labelbottom=False
)

# Show plot
#plt.show()
plt.savefig("img/mountains_earth_and_volcanoes_mars.png", dpi=1000)
