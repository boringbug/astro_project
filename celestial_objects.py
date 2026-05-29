import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy._core.numeric import size
from scipy import ndimage
from skimage import measure


class Moon:
    """Represents the Moon with its image and derived properties."""

    def __init__(self, name: str, file_path: str, blur_sigma: float = 1.0):
        self.name = name
        self.file_path = file_path
        self.blur_sigma = blur_sigma
        # Load and process image (happens once)
        self._image_data = None
        self._center = None
        self._radius = None
        # Auto-load on creation
        self.load_image()

    def load_image(self) -> None:
        """Load and process the FITS image."""
        hdul = fits.open(self.file_path)
        image_data = hdul[0].data
        image_data = image_data[::-1, :]  # Reverse y-axis
        hdul.close()
        # Apply Gaussian blur
        self._image_data = ndimage.gaussian_filter(image_data,
                                                   sigma=self.blur_sigma)

    @property
    def image(self) -> np.ndarray:
        """Get the processed moon image."""
        if self._image_data is None:
            self.load_image()
        return self._image_data

    @property
    def center(self) -> tuple[float, float]:
        """Calculate or return cached moon center."""
        if self._center is None:
            self._calculate_center_and_radius()
        return self._center

    @property
    def radius(self) -> float:
        """Calculate or return cached moon radius."""
        if self._radius is None:
            self._calculate_center_and_radius()
        return self._radius

    def contours(self, threshold: float = 1.2e4) -> list:
        """Get all contours at given threshold."""
        return measure.find_contours(self.image, level=threshold)

    def longest_contour(self, threshold: float = 1.2e4) -> np.ndarray:
        """Get the longest contour (moon edge)."""
        contours = self.contours(threshold)
        return max(contours, key=len)

    def _fit_circle(self, x: np.ndarray, y: np.ndarray
                    ) -> tuple[tuple[float, float], float]:
        """Fit a circle to points."""
        m = np.column_stack([x, y, np.ones_like(x)])
        z = x**2 + y**2
        p, _, _, _ = np.linalg.lstsq(m, z, rcond=None)
        a, b, c = p
        x_center, y_center = a/2, b/2
        radius = np.sqrt(c + x_center**2 + y_center**2)
        return (x_center, y_center), radius

    def _calculate_center_and_radius(self,
                                     x_contour_cutoff: int = 1_750,
                                     threshold: float = 1.2e4
                                     ) -> None:
        """Find moon's center and radius from contour."""
        longest_contour = self.longest_contour(threshold)
        y_contour, x_contour = longest_contour[:, 0], longest_contour[:, 1]
        # Filter contour points
        filtered = [(x, y)
                    for x, y in zip(x_contour, y_contour)
                    if x > x_contour_cutoff]

        if not filtered:
            self._center = (0, 0)
            self._radius = 0
            return
        sliced_x = np.array([p[0] for p in filtered])
        sliced_y = np.array([p[1] for p in filtered])

        self._center, self._radius = self._fit_circle(sliced_x, sliced_y)

    # I need to add tildeness because
    # the selenographic coordinates are not quite right
    # and because the moon is slightl rotating even tho is tidaly locked

    def selenographic2pixel(self, latitude: np.ndarray, longitude: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
        '''
        Latitude = phi
        Longitude = lambda
        '''
        latitude = np.deg2rad(latitude)
        longitude = np.deg2rad(longitude)

        x_pixel = (self.radius*np.sin(longitude)*np.cos(latitude)
                   + self.center[0])
        y_pixel = self.radius*np.sin(latitude) + self.center[1]

        return x_pixel, y_pixel

    def pixel2selenographic(self, x_pixel: np.ndarray, y_pixel: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
        ...

    def plot(self,
             show_contours: bool = False,
             contour_threshold: float = 1.2e4
             ) -> None:
        """Plot the moon with optional contours."""
        plt.imshow(self.image, cmap='gray', origin='lower')
        plt.colorbar(label='Intensity (ADU/DN)')
        if show_contours:
            contours = self.contours(contour_threshold)
            for contour in contours:
                y_contour, x_contour = contour[:, 0], contour[:, 1]
                plt.plot(x_contour, y_contour, linewidth=0.5)
        plt.title(f"Moon: {self.name}")

    def plot_outline(self, color: str = 'red', linewidth: float = 1.0) -> None:
        """Plot a circle at the moon's center with its radius."""
        if self.radius > 0:
            theta = np.linspace(0, 2*np.pi, 100)
            x = self.center[0] + self.radius*np.cos(theta)
            y = self.center[1] + self.radius*np.sin(theta)
            plt.plot(x, y, c=color, linewidth=linewidth)

    @property
    def info(self) -> str:
        """Return a string with moon information."""
        return (f"Moon: {self.name}\n"
                f"Center: ({self.center[0]:.1f}, {self.center[1]:.1f}) pixels\n"
                f"Radius: {self.radius:.1f} pixels\n"
                f"Image shape: {self.image.shape}")

    def craters(self):
        ...

    def plot_craters(self) -> None:
        ...



class Crater:
    def __init__(self,
                 name: str,
                 selenographic_coords: tuple[float, float],
                 radius: float
                 ) -> None:

        self.name = name
        self.selenographic_coords = selenographic_coords
        self.radius = radius

    @property
    def info(self):
        return (f"Crater: {self.name}\n"
                f"Selenographic coordinates: {self.selenographic_coords}"
                f"Radius: {self.radius}")


MOON_FILE_PATH = ("LUNA-C-1, 2025-01-11, 1x5L, SkyWatcher, (C), "
                  "SBIG ST-8300 CCD Camera_average_stacked.fits")



def main() -> None:
    moon = Moon(
        name="Moon",
        file_path=MOON_FILE_PATH,
        blur_sigma=1.0
    )

    padding = 10

    x_min = int(moon.center[0] - moon.radius - padding)
    x_max = int(moon.center[0] + moon.radius + padding)
    y_min = int(moon.center[1] - moon.radius - padding)
    y_max = int(moon.center[1] + moon.radius + padding)


    croped_image = moon.image[y_min:y_max, x_min:x_max]

    plt.imshow(croped_image, cmap='gray', origin='lower')
    plt.show()






# TODO: Crop the image to show and calculate only on the actual moon
# TODO: Have the data only in the moon part, the other part of the matrix
# should be filled with zeros


if __name__ == "__main__":
    main()
