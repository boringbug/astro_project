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


def selenographic_to_pixel(latitude: float, longitude: float) -> tuple[float, float]:
    '''
    Latitude = phi
    Longitude = lambda 
    '''
    R_MOON_PIXEL = 750
    X_MOON_CENTER, Y_MOON_CENTER = (1_500, 1_500)

    latitude = np.deg2rad(latitude)
    longitude = np.deg2rad(longitude)

    x = R_MOON_PIXEL * np.sin(longitude) * np.cos(latitude) + X_MOON_CENTER
    y = R_MOON_PIXEL * np.sin(latitude) + Y_MOON_CENTER

    return x, y

def rotation_matrix(angle_deg: float) -> npt.NDArray[float]:
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    return np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])


def fit_circle(x: float, y: float) -> tuple[tuple[float, float], float]:
    M = np.column_stack([x, y, np.ones_like(x)])
    z = x**2 + y**2

    p, _, _, _ = np.linalg.lstsq(M, z, rcond=None)
    A, B, C = p

    x_center, y_center = A/2, B/2
    radius = np.sqrt(C + x_center**2 + y_center**2)

    return (x_center, y_center), radius


def circle_points(center: tuple[float, float], radius: float, number_points: int =100) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, number_points)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    return x, y


# PRINTING WITH MATPLOTLIB
crater_data_frame = pd.read_csv("creaters.csv")
real_lat, real_long = crater_data_frame['real_Phi'], crater_data_frame['real_lambda']
pixel_x_coords, pixel_y_coords = selenographic_to_pixel(real_lat, real_long)

moon_file_name = "LUNA-C-1, 2025-01-11, 1x5L, SkyWatcher, (C), SBIG ST-8300 CCD Camera_average_stacked.fits"

# Open and read the FITS file
hdul = fits.open(moon_file_name)
image_data = hdul[0].data  # Extract image data

# Close the file
hdul.close()

# Display with matplotlib
plt.figure(figsize=(10, 8))
plt.title('FITS Image')

blurred = ndimage.gaussian_filter(image_data, sigma=1)
plt.imshow(blurred, cmap='gray', origin='lower') # If its blured to simga=1 it look better
plt.colorbar(label='Intensity (ADU/DN)')


#plt.scatter(pixel_x_coords, pixel_y_coords)


contours = measure.find_contours(blurred, level=1.1e4)

contour = max(contours, key=len)
y_contour, x_contour = contour[:, 0], contour[:, 1]

plt.plot(x_contour, y_contour)

plt.show()
