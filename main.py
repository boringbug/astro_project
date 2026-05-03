import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd

def selenographic_to_pixel(latitude, longitude):
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


crater_data_frame = pd.read_csv("creaters.csv")

real_lat, real_long = crater_data_frame['real_Phi'], crater_data_frame['real_lambda']

pixel_x_coords, pixel_y_coords = selenographic_to_pixel(real_lat, real_long)


moon_file_name = "LUNA-C-1, 2025-01-11, 1x5L, SkyWatcher, (C), SBIG ST-8300 CCD Camera_average_stacked.fits"

# Open and read the FITS file
hdul = fits.open(moon_file_name)
image_data = hdul[0].data  # Extract image data

print(hdul)

# Close the file
hdul.close()

# Display with matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(image_data, cmap='gray', origin='lower')
plt.colorbar(label='Intensity (ADU/DN)')
plt.title('FITS Image')

plt.scatter(pixel_x_coords, pixel_y_coords)

plt.show()



