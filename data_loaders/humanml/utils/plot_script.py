"""
3D Motion Visualization Utilities

This module provides functions for visualizing 3D human motion data with various
customization options including dataset-specific scaling, color mapping, and animation.
"""

import math
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import colorsys


# Color Utilities
def hex_to_hsv(hex_color):
    """
    Convert a hex color string to HSV color values.
    
    Args:
        hex_color (str): Hex color code (e.g., '#FF0000')
        
    Returns:
        tuple: HSV color values (hue, saturation, value)
    """
    hex_color = hex_color.strip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    hsv = colorsys.rgb_to_hsv(*(value / 255 for value in rgb))
    return hsv


def hsv_to_hex(hsv_color):
    """
    Convert HSV color values to a hex color string.
    
    Args:
        hsv_color (tuple): HSV color values (hue, saturation, value)
        
    Returns:
        str: Hex color code
    """
    rgb = colorsys.hsv_to_rgb(*hsv_color)
    hex_color = '#{:02x}{:02x}{:02x}'.format(*(int(value * 255) for value in rgb))
    return hex_color


def generate_similar_colors(base_hex, variance=0.1, num_variations=4):
    """
    Generate a list of colors similar to the base color.
    
    Args:
        base_hex (str): Base color in hex format
        variance (float): Amount of variation allowed (0-1)
        num_variations (int): Number of variations to generate
        
    Returns:
        list: List of similar colors in hex format
    """
    base_hsv = hex_to_hsv(base_hex)
    similar_colors = []
    
    for i in range(num_variations):
        # Slightly change the HSV values
        new_hue = (base_hsv[0] + variance * (0.5 - np.random.rand())) % 1.0
        new_saturation = min(max(base_hsv[1] + variance * (0.5 - np.random.rand()), 0), 1)
        new_value = min(max(base_hsv[2] + variance * (0.5 - np.random.rand()), 0), 1)
        
        # Convert it back to hex
        new_color_hex = hsv_to_hex((new_hue, new_saturation, new_value))
        similar_colors.append(new_color_hex)
    
    # Insert the base color in the middle of the list
    similar_colors.insert(2, base_hex)
    
    return similar_colors


def list_cut_average(data_list, intervals):
    """
    Cut a list into intervals and calculate the average for each interval.
    
    Args:
        data_list (list): Input list of values
        intervals (int): Number of intervals
        
    Returns:
        list: List of averaged values
    """
    if intervals == 1:
        return data_list

    bins = math.ceil(len(data_list) * 1.0 / intervals)
    averaged_list = []
    for i in range(bins):
        low_idx = intervals * i
        high_idx = low_idx + intervals
        high_idx = min(high_idx, len(data_list))
        averaged_list.append(np.mean(data_list[low_idx:high_idx]))
    return averaged_list


# Standard Color Palettes
COLOR_PALETTES = {
    'blue': ["#4D84AA", "#5B9965", "#61CEB9", "#34C1E2", "#80B79A"],      # GT color
    'orange': ["#DD5A37", "#D69E00", "#B75A39", "#FF6D00", "#DDB50E"],    # Generation color
    'grey': ["#BA7D3F", "#7D7D7D", "#373737", "#C79765", "#A3A3A3"],      
    'purple': ["#C76665", "#6F5B81", "#A69A8C", "#DDA3A2", "#9A8CA6"],    
    'pink': ["#BF4281", "#4244BF", "#D27A7C", "#D27AA6", "#7A7CD2"],      
    'green': ["#4E9F5B", "#896D52", "#526E89", "#6BAF92", "#C49C76"]      
}


def scale_data_for_dataset(data, dataset):
    """
    Apply dataset-specific scaling to motion data.
    
    Args:
        data (ndarray): Motion data
        dataset (str): Dataset name
        
    Returns:
        ndarray: Scaled data
    """
    data_scaled = data.copy()
    
    if dataset == 'kit':
        data_scaled *= 0.003
    elif dataset in ['humanml', 'humanml+']:
        data_scaled *= 1.3
    elif dataset in ['humanact12', 'uestc']:
        data_scaled *= -1.5
    
    return data_scaled


def plot_xz_plane(ax, minx, maxx, miny, minz, maxz):
    """
    Plot a plane on XZ axis in the 3D space.
    
    Args:
        ax: Matplotlib 3D axis
        minx, maxx: X-axis bounds
        miny: Y-axis value (height)
        minz, maxz: Z-axis bounds
    """
    verts = [
        [minx, miny, minz],
        [minx, miny, maxz],
        [maxx, miny, maxz],
        [maxx, miny, minz]
    ]
    xz_plane = Poly3DCollection([verts])
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    ax.add_collection3d(xz_plane)


def generate_color_mapping(indices, clip_embeddings, label_id_map):
    """
    Generate color mapping for different motion labels.
    
    Args:
        indices (list): List of indices
        clip_embeddings (ndarray): Clip embeddings
        label_id_map (str): Path to label ID mapping file
        
    Returns:
        tuple: Color map and hex color map
    """
    # Get unique indices
    unique_indices = list(set(indices))
    clip_embeddings_set = [clip_embeddings[i] for i in unique_indices]
    
    # Load label-id mapping
    with open(label_id_map, 'r') as file:
        texts_dict = json.load(file)
    inverse_texts_dict = {v: k for k, v in texts_dict.items()}
    
    # Calculate pairwise distances
    distances = pairwise_distances(clip_embeddings_set)
    scaler = MinMaxScaler()
    distances_normalized = scaler.fit_transform(distances)
    
    # Create color mapping
    colormap = cm.get_cmap('tab20')
    color_map = {}
    
    for i, label in enumerate(unique_indices):
        color = colormap(i / len(unique_indices))
        color_map[inverse_texts_dict[label]] = color
    
    # Generate similar colors for each label
    color_map_hex = {
        label: generate_similar_colors(matplotlib.colors.to_hex(rgba), variance=0.3) 
        for label, rgba in color_map.items()
    }
    
    return color_map, color_map_hex


def plot_3d_motion(save_path, kinematic_tree, joints, title, dataset, figsize=(3, 3), fps=120, radius=3,
                   vis_mode='default', gt_frames=None):
    """
    Plot and save a basic 3D motion animation.
    
    Args:
        save_path (str): Path to save the animation
        kinematic_tree (list): List of joint chains defining the skeleton
        joints (ndarray): Joint positions data
        title (str): Title for the visualization
        dataset (str): Dataset name ('kit', 'humanml', 'humanact12', 'uestc')
        figsize (tuple): Figure size (width, height)
        fps (int): Frames per second for the animation
        radius (float): Visualization radius
        vis_mode (str): Visualization mode ('default', 'gt', 'upper_body')
        gt_frames (list): List of ground truth frames
    """
    matplotlib.use('Agg')
    
    if gt_frames is None:
        gt_frames = []
    
    # Wrap title for better display
    title = '\n'.join(wrap(title, 20))
    
    # Reshape and scale data according to dataset
    data = joints.copy().reshape(len(joints), -1, 3)
    data = scale_data_for_dataset(data, dataset)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    
    # Set up initial view
    def init():
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)
    
    init()
    
    # Compute data bounds
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    
    # Select colors based on visualization mode
    colors_blue = COLOR_PALETTES['blue']    # GT color
    colors_orange = COLOR_PALETTES['orange']  # Generation color
    
    colors = colors_orange
    if vis_mode == 'upper_body':  # lower body taken fixed to input motion
        colors[0] = colors_blue[0]
        colors[1] = colors_blue[1]
    elif vis_mode == 'gt':
        colors = colors_blue
    
    # Get number of frames
    frame_number = data.shape[0]
    
    # Adjust data positioning
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajectory = data[:, 0, [0, 2]]
    
    # Center the motion
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    
    # Define update function for animation
    def update(index):
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        
        # Plot the ground plane
        plot_xz_plane(
            ax,
            MINS[0] - trajectory[index, 0], 
            MAXS[0] - trajectory[index, 0], 
            0, 
            MINS[2] - trajectory[index, 1], 
            MAXS[2] - trajectory[index, 1]
        )
        
        # Select colors based on whether the frame is in gt_frames
        used_colors = colors_blue if index in gt_frames else colors
        
        # Plot kinematic chains
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data[index, chain, 0], 
                data[index, chain, 1], 
                data[index, chain, 2], 
                linewidth=linewidth, 
                color=color
            )
        
        # Remove axis labels
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    # Create and save animation
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()


def plot_3d_motion_pred(save_path, kinematic_tree, joints, dataset, title, idx, clip_path, 
                        label_id_path, figsize=(10, 10), fps=120, radius=4, vis_mode='default', 
                        label_frames=None):
    """
    Plot and save an advanced 3D motion prediction animation with labeled frames.
    
    Args:
        save_path (str): Path to save the animation
        kinematic_tree (list): List of joint chains defining the skeleton
        joints (ndarray): Joint positions data
        dataset (str): Dataset name ('kit', 'humanml', 'humanact12', 'uestc')
        title (str): Title for the visualization
        idx (list): Indices for clip embeddings
        clip_path (str): Path to the clip embeddings file
        label_id_path (str): Path to the label ID mapping file
        figsize (tuple): Figure size (width, height)
        fps (int): Frames per second for the animation
        radius (float): Visualization radius
        vis_mode (str): Visualization mode
        label_frames (list): Frame labels, if None will be initialized with zeros
    """
    # Load clip embeddings
    clip_np = np.loadtxt(clip_path, delimiter='\t')
    
    # Generate color mapping
    color_map, color_map_hex = generate_color_mapping(idx, clip_np, label_id_path)
    
    # Prepare title
    title = '\n'.join(wrap(title, 62))
    
    # Reshape and scale data according to dataset
    data = joints.copy().reshape(len(joints), -1, 3)
    data = scale_data_for_dataset(data, dataset)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    ax = p3.Axes3D(fig)
    
    # Compute data bounds
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    
    # Get number of frames
    frame_number = data.shape[0]
    
    # Initialize frame labels if not provided
    if label_frames is None or label_frames == []:
        label_frames = np.zeros(frame_number)
        print('No frame labels provided, initializing with zeros')
    
    # Set up initial view
    def init():
        offset_x = 0.7 
        offset_y = -1
        offset_z = 0
        ax.set_xlim3d([(-radius / 2) + offset_x, (radius / 2) + offset_x])
        ax.set_ylim3d([0 + offset_y, radius + offset_y])
        ax.set_zlim3d([0 + offset_z, radius + offset_z])
        fig.suptitle(title, fontsize=20)
        ax.grid(False)
        ax.axis('off')
    
    init()
    
    # Adjust data positioning
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajectory = data[:, 0, [0, 2]]
    
    # Center the motion
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    
    # Add text element for displaying current frame's label
    text = fig.text(0.4, 0.22, '', ha='center', va='bottom')
    
    # Create secondary axes for time bar
    ax2 = fig.add_axes([0.02, 0.05, 0.65, 0.1], label='timebar')
    ax2.set_xlim(0, len(joints))
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_yticklabels([])
    
    # Define update function for animation
    def update(index):
        # Determine color dictionary to use
        unique_labels = list(set(label_frames))
        
        if color_map_hex is None:
            # Fallback to predefined colors if color_map_hex is None
            color_pre = {
                0: COLOR_PALETTES['blue'],
                1: COLOR_PALETTES['orange'],
                2: COLOR_PALETTES['grey'],
                3: COLOR_PALETTES['purple'],
                4: COLOR_PALETTES['pink'],
                5: COLOR_PALETTES['green']
            }
            color_dic = {label: color_pre[i % len(color_pre)] for i, label in enumerate(unique_labels)}
        else:
            color_dic = color_map_hex
        
        # Clear previous frame
        ax.lines = []
        ax.collections = []
        
        # Set camera view
        ax.view_init(elev=120, azim=-90)
        ax.dist = 7.5
        
        # Plot ground plane
        plot_xz_plane(
            ax,
            MINS[0] - trajectory[index, 0], 
            MAXS[0] - trajectory[index, 0], 
            0, 
            MINS[2] - trajectory[index, 1], 
            MAXS[2] - trajectory[index, 1]
        )
        
        # Get colors for current frame's label
        current_label = label_frames[index]
        used_colors = color_dic[current_label]
        
        # Plot kinematic chains
        for i, (chain, color) in enumerate(zip(kinematic_tree, used_colors)):
            linewidth = 4.0 if i < 5 else 2.0
            ax.plot3D(
                data[index, chain, 0], 
                data[index, chain, 1], 
                data[index, chain, 2], 
                linewidth=linewidth, 
                color=color
            )
        
        # Update text label
        text.set_text(current_label)
        text.set_color(used_colors[2])
        text.set_fontsize(25)
        
        # Remove axis labels
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
        # Update time bar
        ax2.cla()
        ax2.set_xlim(0, len(joints))
        ax2.set_ylim(0, 1)
        ax2.set_yticklabels([])
        
        # Draw color segments for each frame's label
        for i, label in enumerate(label_frames):
            rect = mpatches.Rectangle((i, 0), 1, 1, color=color_dic[label][2])
            ax2.add_patch(rect)
        
        # Vertical line for current frame
        ax2.axvline(x=index, color='k', linewidth=2)
    
    # Add legend if color mapping exists
    if color_map is not None:
        unique_labels = sorted(set(label_frames), key=label_frames.index)
        max_chars = 28
        
        # Wrap text for each label
        wrapped_labels = ['\n'.join(wrap(str(label), max_chars)) for label in unique_labels]
        
        # Create patches for legend
        handles = [
            mpatches.Patch(color=color_map[label], label=wrapped_label) 
            for label, wrapped_label in zip(unique_labels, wrapped_labels)
        ]
        
        # Add legend in separate axes
        legend_ax = fig.add_axes([0.73, 0.03, 0.2, 0.1], label='legend')
        legend_ax.axis('off')
        legend_ax.legend(handles=handles, fontsize='large', loc='center', frameon=False)
    
    # Create and save animation
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()