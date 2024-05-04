import numpy as np
from sklearn.neighbors import NearestNeighbors


def icp(A, B, max_iterations=20, tolerance=0.001):
    '''
    Iterative Closest Point (ICP) algorithm to align two point clouds.

    Parameters:
    - A: Nxm numpy array of source point cloud (mD points)
    - B: Nxm numpy array of destination point cloud (mD points)
    - max_iterations: Maximum number of iterations for convergence
    - tolerance: Convergence criteria

    Returns:
    - aligned_cloud: The merged point cloud
    - T: Final transformation matrix that aligns A with B
    '''

    assert A.shape == B.shape

    # Get the number of dimensions (m) and the number of points (N)
    m = A.shape[1]
    # N = A.shape[0]

    # Initialize the transformation matrix as an identity matrix
    T = np.identity(m + 1)

    prev_error = 0

    for iteration in range(max_iterations):
        # Find the nearest neighbors between the current source (A) and destination (B) points
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(B)
        distances, indices = neigh.kneighbors(A)

        # Extract the corresponding points
        correspondences_A = A
        correspondences_B = B[indices.flatten()]

        # Calculate the transformation matrix to align the correspondences_A to correspondences_B
        T_iteration = best_fit_transform(correspondences_A, correspondences_B)

        # Update the transformation matrix T
        T = np.dot(T_iteration, T)

        # Apply the transformation to the source cloud A
        A = np.dot(A, T_iteration[:m, :m].T) + T_iteration[:m, m]

        # Calculate the mean error (distance between corresponding points)
        mean_error = np.mean(distances)

        # Check for convergence
        if np.abs(prev_error - mean_error) < tolerance:
            break

        prev_error = mean_error

    # Apply the final transformation to the source cloud A to align it with the destination cloud B
    aligned_cloud = np.dot(A, T[:m, :m].T) + T[:m, m]

    return aligned_cloud, T


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions.

    Parameters:
    - A: Nxm numpy array of source points
    - B: Nxm numpy array of destination points

    Returns:
    - T: (m+1)x(m+1) homogeneous transformation matrix
    '''
    assert A.shape == B.shape

    # Get the number of dimensions (m)
    m = A.shape[1]

    # Calculate centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Subtract centroids to center the data
    AA = A - centroid_A
    BB = B - centroid_B

    # Singular Value Decomposition
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)

    # Calculate the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[m-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Calculate the translation vector
    t = centroid_B - np.dot(R, centroid_A)

    # Create the homogeneous transformation matrix
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def run():
    # Create two random point clouds A and B (Replace this with your point clouds)
    A = np.random.rand(100, 3)
    B = np.random.rand(100, 3)

    # Call the ICP function to align the point clouds
    aligned_cloud, transformation_matrix = icp(A, B)

    # Print the transformation matrix
    print("Transformation Matrix:")
    print(transformation_matrix)

# run()
