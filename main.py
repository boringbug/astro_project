import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from dataclasses import dataclass

from skimage import feature, measure
from scipy import ndimage

# DEFINITIONS

@dataclass
class Crater:
    name: str
    pixel_x: int
    pixel_y: int
    selenographic_latitude: float
    selenographic_longitude: float
    radius_km: float
    depth_km: float


@dataclass
class Moon:
    name: str
    center_pixel: tuple[float, float]
    radius_pixel: float
    tiltednes: float


def get_longest_contour(image_data, threshold: float = 1.2e4):
    contours = measure.find_contours(image_data, level=threshold)
    return max(contours, key=len)

def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], float]:
    M = np.column_stack([x, y, np.ones_like(x)])
    z = x**2 + y**2

    p, _, _, _ = np.linalg.lstsq(M, z, rcond=None)
    A, B, C = p

    x_center, y_center = A/2, B/2
    radius = np.sqrt(C + x_center**2 + y_center**2)

    return (x_center, y_center), radius

def get_moon_center_and_radius(image_data, x_contour_cutoff: int = 1_750):
    longest_contour = get_longest_contour(image_data)
    y_contour, x_contour = longest_contour[:, 0], longest_contour[:, 1]

    # Get the good parts of the contour
    filtered_x_contour, filtered_y_contour = zip(
        *[(x, y) for x, y in zip(x_contour, y_contour)
        if x > x_contour_cutoff]) or ([], [])

    sliced_x_contour = np.array(filtered_x_contour)
    sliced_y_contour = np.array(filtered_y_contour)

    center, radius = fit_circle(sliced_x_contour, sliced_y_contour)

    return center, radius

def circle_points(center: tuple[float, float], radius: float, number_points: int =100) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, number_points)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    return x, y

def plot_circle(center: tuple[float, float], radius: float, color: str, linewidth: float = 0.5) -> None:
    x_circle, y_circle = circle_points(center, radius)
    plt.plot(x_circle, y_circle, c=color, linewidth=linewidth)

def selenographic_to_pixel(latitude: np.ndarray, longitude: np.ndarray, image_data) -> tuple[np.ndarray, np.ndarray]:
    '''
    Latitude = phi
    Longitude = lambda 
    '''

    moon_center_pixel, moon_radius_pixel = get_moon_center_and_radius(image_data)

    latitude = np.deg2rad(latitude)
    longitude = np.deg2rad(longitude)

    x = moon_radius_pixel * np.sin(longitude) * np.cos(latitude) + moon_center_pixel[0]
    y = moon_radius_pixel * np.sin(latitude) + moon_center_pixel[1]

    return x, y

def plot_craters(file_name: str, size: float = 0.5, color: str = "Blue") -> None:
    crater_data_frame = pd.read_csv(file_name)
    real_lat, real_long = crater_data_frame['real_Phi'], crater_data_frame['real_lambda']
    pixel_x_coords, pixel_y_coords = selenographic_to_pixel(real_lat, real_long)
    plt.scatter(pixel_x_coords, pixel_y_coords, s=size, c=color)


def get_moon_data(file_name: str, blur: float = 1):
    # Open and read the FITS file
    hdul = fits.open(file_name)
    image_data = hdul[0].data  # Extract image data
    image_data = image_data[::-1, :] # reversing along the y axis
    hdul.close()
    blurred_image_data = ndimage.gaussian_filter(image_data, sigma=blur)

    return blurred_image_data

def plot_moon(image_data) -> None:
    plt.imshow(image_data, cmap='gray', origin='lower')
    plt.colorbar(label='Intensity (ADU/DN)')


def get_contours(image_data, threshold: float = 1.2e4):
    return measure.find_contours(image_data, level=threshold)

def plot_contour(contour, linewidth: float = 0.5) -> None:
    y_contour, x_contour = contour[:, 0], contour[:, 1]
    plt.plot(x_contour, y_contour, linewidth=linewidth)

def plot_all_contours(image_data, threshold):
    contours = get_contours(image_data, threshold)

    for contour in contours:
        y_contour, x_contour = contour[:, 0], contour[:, 1]
        plt.plot(x_contour, y_contour)




MOON_FILE_NAME = "LUNA-C-1, 2025-01-11, 1x5L, SkyWatcher, (C), SBIG ST-8300 CCD Camera_average_stacked.fits" 
CRATER_FILE_NAME = "creaters.csv"


def main() -> None:
    image_data = get_moon_data(MOON_FILE_NAME)

    plot_moon(image_data)

    plt.show()




if __name__ == "__main__":
    main()
