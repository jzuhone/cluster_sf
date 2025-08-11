from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion, Regions 
from cluster_sf.constants import resolve_width, angular_scale
import astropy.units as u
import numpy as np


reg_c = RectangleSkyRegion(
    SkyCoord(ra=194.9436 * u.deg, dec=27.9465 * u.deg),
    resolve_width,
    resolve_width,
    angle=286.0 * u.deg,
)

reg_c1 = RectangleSkyRegion(
    SkyCoord(ra=194.9529258 * u.deg, dec=27.9306171 * u.deg),
    resolve_width * 0.5,
    resolve_width * 0.5,
    angle=286.0 * u.deg,
)

reg_c2 = RectangleSkyRegion(
    SkyCoord(ra=194.9256281 * u.deg, dec=27.9382628 * u.deg),
    resolve_width * 0.5,
    resolve_width * 0.5,
    angle=286.0 * u.deg,
)

reg_c3 = RectangleSkyRegion(
    SkyCoord(ra=194.9342726 * u.deg, dec=27.9615468 * u.deg),
    resolve_width * 0.5,
    resolve_width * 0.5,
    angle=286.0 * u.deg,
)

reg_c4 = RectangleSkyRegion(
    SkyCoord(ra=194.9615869 * u.deg, dec=27.9555657 * u.deg),
    resolve_width * 0.5,
    resolve_width * 0.5,
    angle=286.0 * u.deg,
)

reg_c = RectangleSkyRegion(
    SkyCoord(ra=194.9436 * u.deg, dec=27.9465 * u.deg),
    resolve_width,
    resolve_width,
    angle=286.0 * u.deg,
)

reg_s = RectangleSkyRegion(
    SkyCoord(ra=194.9412 * u.deg, dec=27.8469 * u.deg),
    resolve_width,
    resolve_width,
    angle=314.0 * u.deg,
)

reg_n = RectangleSkyRegion(
    SkyCoord(ra=194.94136 * u.deg, dec=28.04897 * u.deg),
    resolve_width,
    resolve_width,
    angle=105.84454 * u.deg,
)

if __name__ == "__main__":
    regs2 = Regions([reg_c1, reg_c2, reg_c3, reg_c4, reg_s])
    regs2.write("two_pts.reg", format="ds9", overwrite=True)
    regs3 = Regions([reg_c1, reg_c2, reg_c3, reg_c4, reg_s, reg_n])
    regs3.write("three_pts.reg", format="ds9", overwrite=True)
    
