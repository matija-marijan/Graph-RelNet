import random
import numpy as np
from utils.utils import calculate_distances
import matplotlib.pyplot as plt

def normalize_points(points: np.ndarray, max_diameter: float = 300.0):
    """
    Normalize the coordinates of a microphone array to have zero mean and unit variance.

    Parameters:
    points (numpy.ndarray): A 2D array where each row represents the 3D coordinates of a microphone in an array.
    max_diameter (float): The maximum diameter of the microphone array in centimeters. Default is 300.0.

    Returns:
    numpy.ndarray: A 2D array where each row represents the normalized 3D coordinates of a microphone array.
    """
    
    for row in range(points.shape[0]):
        x = points[row, 0]
        y = points[row, 1]
        z = points[row, 2]

        # TODO: Normalize the coordinates

    return points

def add_perturbation(points: np.ndarray, max_perturbation: float = 0.1, min_distance: float = 0.5, max_diameter: float = 300.0):
    """
    Add a random perturbation to the coordinates of a microphone array.

    Parameters:
    points (numpy.ndarray): A 2D array where each row represents the 3D coordinates of a microphone in an array.
    max_perturbation (float): The maximum perturbation in centimeters. Default is 0.1.
    min_distance (float): The minimum distance between microphones in centimeters. Default is 0.5.
    max_diameter (float): The maximum diameter of the microphone array in centimeters. Default is 300.0.

    Returns:
    numpy.ndarray: A 2D array where each row represents the perturbed 3D coordinates of a microphone array.
    """
    num_points = points.shape[0]
    
    while True:
        tmp_points = points.copy()
        perturbation = np.random.uniform(-max_perturbation, max_perturbation, (num_points, 3))
        # Add perturbation to the Y and Z coordinates
        tmp_points[:, 1:3] += perturbation[:, 1:3]
        
        # Check if all distances are greater than min_distance and within max_size
        # TODO: Find a more efficient way to check this
        distances = calculate_distances(tmp_points, triu = True)
        if np.all(distances >= min_distance) and np.all(distances <= max_diameter):
            break
        # If not, revert the perturbation and try again
        print("Perturbation rejected")
    
    points = tmp_points
    return points

def generate_random_points(num_points: int = None, min_distance: float = 0.05, max_diameter: float = 3.0):
    """
    Generate random 2D microphone coordinates with a minimum distance between them and a maximum microphone array size. 
    This function assumes a polar coordinate system with the origin at the center of the microphone array. 

    Parameters:
    num_points (int): The number of microphones to generate. Default is None.
    min_distance (float): The minimum distance between microphones in meters. Default is 0.05
    max_diameter (float): The maximum diameter of the microphone array in meters. Default is 3.0
    """
    if num_points is None:
        num_points = random.randint(3, 25)
    
    points = []
    max_radius = max_diameter / 2

    count = 0
    points = np.zeros((num_points, 3))
    while count < num_points:

        # Generate random uniform points within the max_size using polar coordinates
        # TODO: Uniformly generate the points? (currently they are more dense at the center)

        x = 0
        # y = random.uniform(-max_radius, max_radius)
        # z = random.uniform(-max_radius, max_radius)

        r = random.uniform(0, max_radius)
        theta = random.uniform(0, 2 * np.pi)

        y = r * np.cos(theta)
        z = r * np.sin(theta)

        new_point = np.array([x, y, z])
        
        if all(np.linalg.norm(new_point - points[i, :]) >= min_distance for i in range(count)):
            points[count, :] = new_point
            count += 1
    
    return points

def plot_geometry(points: np.ndarray):
    """
    Plot the 2D coordinates of a microphone array.

    Parameters:
    points (numpy.ndarray): A 2D array where each row represents the 3D coordinates of a microphone in an array.
    """

    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 1], points[:, 2])
    plt.xlabel('Y coordinate')
    plt.ylabel('Z coordinate')
    plt.title('Microphone Array Coordinates')
    plt.grid(True)
    plt.axis('equal')
    plt.waitforbuttonpress()
    plt.close()
    
# Example usage
if __name__ == "__main__":

    num_points = 15
    min_distance = 1.0
    max_diameter = 20.0
    max_perturbation = 1.0

    points = generate_random_points(num_points = num_points, min_distance = min_distance, max_diameter = max_diameter)
    # print(points.shape)
    # print(points)
    distances = calculate_distances(points, triu = True)
    # print(distances)
    print(min(distances))

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(points[:, 1], points[:, 2])
    plt.xlabel('Y coordinate')
    plt.ylabel('Z coordinate')
    plt.title('Microphone Array Coordinates')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-max_diameter / 2, max_diameter / 2)
    plt.ylim(-max_diameter / 2, max_diameter / 2)

    points = add_perturbation(points, max_perturbation = max_perturbation, min_distance = min_distance, max_diameter = max_diameter)
    distances = calculate_distances(points, triu = True)
    # print(distances)
    print(min(distances))

    plt.subplot(122)
    plt.scatter(points[:, 1], points[:, 2])
    plt.xlabel('Y coordinate')
    plt.ylabel('Z coordinate')
    plt.title('Microphone Array Coordinates after Perturbation')
    plt.grid(True)
    plt.axis('equal')
    plt.xlim(-max_diameter / 2, max_diameter / 2)
    plt.ylim(-max_diameter / 2, max_diameter / 2)
    plt.show()

    