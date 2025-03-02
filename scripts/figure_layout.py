import matplotlib.pyplot as plt

# Update rcParams for global settings
plt.rcParams.update({
    #Figure properties
    'figure.figsize': (15, 7),  # Default figure size (width, height)
    'figure.autolayout': True, # Auto layout for adjusting subplots and legends
    'figure.dpi': 500,         # Resolution of the figure

    # Legend settings
    'legend.loc': 'upper right',     # Place legend near the top-right
    'legend.frameon': False,         # Remove the legend frame
    'legend.fontsize': 24,           # Font size for legend
    'legend.borderpad': 0.4,         # Padding around the legend content
    'legend.borderaxespad': 0,  # No padding between legend and axes
    'legend.columnspacing': 1.0,     # Spacing between columns in the legend
    'legend.handlelength': 1,      # Length of legend lines
    'legend.handleheight': 0.7,      # Height of legend handles
    'legend.labelspacing': 0.5,      # Spacing between legend entries

    # Axis properties
    'axes.labelcolor': 'black',      # Axis label color
    'axes.labelsize': 24, # Font size for axis labels
    'axes.linewidth': 1.2,      # Axis line width
    'axes.labelweight': 'normal',    # Axis label weight

    # Tick settings
    'xtick.labelsize': 24,           # Font size for x-axis ticks
    'ytick.labelsize': 24,           # Font size for y-axis ticks
    'xtick.color': 'black',          # Tick label color
    'ytick.color': 'black',          # Tick label color

    # Grid
    'axes.grid': False, # Do not show grid by default
    'grid.color': 'lightgray',       # Grid line color
    'grid.linestyle': '--',          # Grid line style
    'grid.linewidth': 0.5,           # Grid line width
    'grid.alpha': 0.6,

    # Other settings
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),  # Consistent color cycle

    # Font settings
    'font.family': 'Arial',          # Font family for all text
    'font.size': 24,                 # General font size
    'text.color': 'black',           # Default text color
})