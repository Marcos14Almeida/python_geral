
# =============================================================================
# ================================= Libraries =================================
# =============================================================================

import numpy as np
import trimesh
import os
import matplotlib.pyplot as plt
from icp import icp

# =============================================================================
#                                   Functions
# =============================================================================


def plot(data, title="3D Scatter Plot", n_points=7000):
    """
    Create a 3D scatter plot for a given set of points.

    Parameters:
    - data (numpy.ndarray): The point cloud data.
    - title (str): The title for the plot (default is "3D Scatter Plot").
    - n_points (int): The number of random points to plot (default is 7000).

    Returns:
    None
    """
    # Sample a subset of points to visualize better the points
    random_indices = np.random.choice(data.shape[0], size=n_points, replace=False)

    # Select the subset of data using the random indices
    data = data[random_indices]

    # Extract the three columns for the scatter plot
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b', marker='o', s=7)

    # Customize the plot
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(title)

    # Show the plot
    plt.show()


def plot2(points1, points2, title="3D Scatter Plot", n_points=7000):
    """
    Create a 3D scatter plot for two sets of points.

    Parameters:
    - points1 (numpy.ndarray): The first point cloud data.
    - points2 (numpy.ndarray): The second point cloud data.
    - title (str): The title for the plot (default is "3D Scatter Plot").
    - n_points (int): The number of random points to plot (default is 7000).

    Returns:
    None
    """

    if points1.shape[0] < n_points:
        n_points = points1.shape[0]

    # Sample a subset of points to visualize better the points
    random_indices = np.random.choice(points1.shape[0], size=n_points, replace=False)
    points1 = points1[random_indices]
    points2 = points2[random_indices]

    # Create a Matplotlib 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first point cloud in blue
    ax.scatter(
            points1[:, 0], points1[:, 1], points1[:, 2],
            c='b', marker='o', s=6, label='Point Cloud 1')

    # Plot the second point cloud in red
    ax.scatter(
            points2[:, 0], points2[:, 1], points2[:, 2],
            c='r', marker='^', s=6, label='Point Cloud 2')

    # Customize the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(title)

    # Show the 3D plot
    plt.show()


def plot_trajectory(trajectory1, trajectory2, title=""):
    """
    Create a 3D plot to visualize trajectories.

    Parameters:
    - trajectory1 (list): List of (x, y, z) coordinates for trajectory 1.
    - trajectory2 (list): List of (x, y, z) coordinates for trajectory 2.
    - title (str): The title for the plot.

    Returns:
    None
    """

    # Extract X, Y, and Z coordinates from the array
    x1, y1, z1 = zip(*trajectory1)
    x2, y2, z2 = zip(*trajectory2)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the line by connecting the coordinates
    ax.plot(x1, y1, z1, label='Trajectory Calculated', marker='o', markersize=5)

    ax.plot(x2, y2, z2, label='Trajectory Truth', marker='s', markersize=5)

    # Add labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.legend()

    # Add title
    ax.set_title(title)

    # Show the 3D plot
    plt.show()


def list_files_in_directory(directory):
    """
    List all files in a directory and its subdirectories.

    Parameters:
    - directory (str): The directory path.

    Returns:
    list: A list of file paths.
    """
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))

    return file_list


def load_data_points(folder_path):
    """
    Load data points from a folder containing subdirectories of point cloud files.

    Parameters:
    - folder_path (str): The path to the main folder containing subdirectories.

    Returns:
    list: A list of point cloud data arrays.
    """
    print("\nLoading Data Points...\n")
    # Get a list of all subdirectories (folders) inside the "joica" folder
    subdirectories = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    # Print the list of subdirectories
    point_clouds = []
    for subdirectory in subdirectories:
        file_name = list_files_in_directory(subdirectory)[0]
        # print(file_name)
        point_cloud_obj = trimesh.load(file_name).vertices

        # Sample a portion of all points
        random_indices = np.random.choice(point_cloud_obj.shape[0], size=58553, replace=False)
        point_cloud_obj = point_cloud_obj[random_indices]

        point_clouds.append(point_cloud_obj)

    return point_clouds


def load_ground_truth_list(file_name):
    """
    Load a list of ground truth transformation matrices from a file.

    Parameters:
    - file_name (str): The name of the file containing ground truth data.

    Returns:
    list: A list of transformation matrices.
    """

    print("\nGROUND TRUTH - Transformation Matrix")
    all_ground_truth = np.load(file_name)
    ground_truth_list = [all_ground_truth[i:i+1, :, :] for i in range(30)]
    print(f"Qnt: {len(ground_truth_list)} / Shape: {ground_truth_list[0].shape}")

    return ground_truth_list


def car_trajectory(T, car_coordinates, coordinates_list):
    """
    Compute the trajectory of a car's position based on a transformation matrix.

    Args:
        T (numpy.ndarray): Transformation matrix representing the car's movement.
        car_coordinates (numpy.ndarray): Current coordinates of the car in homogeneous coordinates.
        coordinates_list (list): List to store the car's trajectory coordinates.

    Returns:
        Tuple[numpy.ndarray, list]: A tuple containing the updated car coordinates and the updated trajectory list.
    """

    # Apply the transformation matrix to the current position
    new_position = np.dot(T, car_coordinates)
    # Extract the coordinates (X, Y, Z) from the new position
    x, y, z, _ = new_position
    # Store the coordinates in the list
    coordinates_list.append([x, y, z])
    # Update the current position to the new position for the next iteration
    car_coordinates = new_position

    return car_coordinates, coordinates_list


# =============================================================================
#                                   Functions
# =============================================================================

print()

# Specify the path to the folder
folder_path = 'KITTI-Sequence'
point_clouds = load_data_points(folder_path)

print("-"*50)
print("Resume:")

print("Cloud Points")
print(f"Qnt: {len(point_clouds)} / Shape: {point_clouds[0].shape} -> (n points, dimensions)")
# plot(point_clouds[0])

# =============================================================================

# Ground Truth
file_name = "ground_truth.npy"
ground_truth_list = load_ground_truth_list(file_name)


# =============================================================================
# ICP

# Parameters
max_iterations = 50  # Maximum number of iterations
tolerance = 1e-6       # Convergence tolerance

print()
print("-"*50)
print("ICP ALGORITHM")

# Car coordinates
car_coordinates_icp = np.array([0, 0, 0, 1])
car_coordinates_truth = np.array([0, 0, 0, 1])
coordinates_icp_list = []
coordinates_truth_list = []

# Run the algorithm for each point cloud
for i in range(len(point_clouds)-1):
    print(f"\nCloud {i+1}")

    # Point Clouds
    cloud1 = point_clouds[0]
    cloud2 = point_clouds[i+1]

    # ICP Algorithm
    aligned_cloud, T_icp = icp(cloud1, cloud2, max_iterations, tolerance)

    # Result
    # plot2(cloud1, aligned_cloud, "Aligned Cloud " + str(i))
    # print(f"\nMatrix from ICP\n{T_icp}")

    # Compare with ground truth
    print("Error GROUND TRUTH")
    T_truth = ground_truth_list[0]
    sum_of_errors = np.sum(np.abs(T_icp - T_truth))
    print("Sum of Matrix errors:", sum_of_errors)
    aligned_cloud2_truth = np.dot(cloud2, T_truth[:3, :3].T) + T_truth[:3, 3]
    # plot2(aligned_cloud, aligned_cloud2_truth, "Aligned x Truth Cloud " + str(i))

    # Calculate car trajectory
    car_coordinates_icp, coordinates_icp_list = car_trajectory(
        T_icp, car_coordinates_icp, coordinates_icp_list)

    # Calculate true car trajectory
    car_coordinates_truth, coordinates_truth_list = car_trajectory(
        T_truth[0], car_coordinates_truth, coordinates_truth_list)


# =============================================================================
# Plot Car trajectory

print("\nCAR TRAJECTORY")
plot_trajectory(coordinates_icp_list, coordinates_truth_list, "Trajectory")

# =============================================================================
#                                     END
# =============================================================================
print()
print(f"{'-'*30}   END   {'-'*30}")
print()
