import matplotlib.pyplot as plt
import numpy as np
from skimage import filters, feature
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter

from time import time_ns

from celestial_objects import Moon

MOON_FILE_PATH = ("LUNA-C-1, 2025-01-11, 1x5L, SkyWatcher, (C), "
                  "SBIG ST-8300 CCD Camera_average_stacked.fits")
CRATER_FILE_PATH = "craters.csv"


def main() -> None:
    moon = Moon(
        name="Moon",
        file_path=MOON_FILE_PATH,
        blur_sigma=1.0
    )

    edges = feature.canny(moon.image, sigma=20)
    plt.imshow(edges, origin='lower')
    plt.show()

    sobel = filters.sobel(moon.image)
    plt.imshow(sobel, origin='lower')
    plt.show()

    print("Starting the hough_circle ...")

    start_time = time_ns()

    hough_radii = np.arange(10, 110, 1)
    hough_res = hough_circle(edges, hough_radii)

    print("Finn hough_circle")

    accums, cx, cy, radii = hough_circle_peaks(hough_res,
                                               hough_radii,
                                               total_num_peaks=100)
    print("Starting the loop")
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y,
                                        center_x,
                                        radius,
                                        shape=moon.image.shape)
        plt.plot(circx, circy)

    moon.plot()
    plt.show()

    end_time = time_ns()

    print(f"Elapsed time was: {(end_time - start_time)*1e-9}s")


if __name__ == "__main__":
    main()
