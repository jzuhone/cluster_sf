import numpy as np
from astropy import wcs
from regions import RectangleSkyRegion, Regions
from astropy.coordinates import SkyCoord
import astropy.units as u


ra0 = 194.9436
dec0 = 27.9465

nx = 34
ny = 34

dx = 30.0 / 3600.0  # 30 arcseconds in degrees

w = wcs.WCS(naxis=2)
w.wcs.crpix = [0.5 * (nx + 1), 0.5 * (ny + 1)]
w.wcs.cdelt = np.array([-dx, dx])
w.wcs.crval = [ra0, dec0]
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

x, y = np.mgrid[0:nx, 0:ny] + 1

ra, dec = w.wcs_pix2world(x, y, 1)

regs = []
for i in range(nx):
    for j in range(ny):
        region = RectangleSkyRegion(
            center=SkyCoord(ra=ra[i, j], dec=dec[i, j], unit="deg"),
            width=dx * u.deg,
            height=dx * u.deg,
        )
        regs.append(region)

regs = Regions(regs)
regs.write("doppler_regions.reg", format="ds9", overwrite=True)
