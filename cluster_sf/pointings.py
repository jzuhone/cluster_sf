from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion, Regions 
from constants import resolve_width, angular_scale
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

reg_1Mpc = RectangleSkyRegion(
    SkyCoord(ra=194.6589254 * u.deg, dec=28.5318875 * u.deg),
    resolve_width,
    resolve_width,
    angle=0.0 * u.deg,
)

bin1_angles = np.array(
    [
        reg_c1.center.separation(reg_c2.center).to_value("arcmin"),
        reg_c1.center.separation(reg_c3.center).to_value("arcmin"),
        reg_c1.center.separation(reg_c4.center).to_value("arcmin"),
        reg_c2.center.separation(reg_c3.center).to_value("arcmin"),
        reg_c2.center.separation(reg_c4.center).to_value("arcmin"),
        reg_c3.center.separation(reg_c4.center).to_value("arcmin"),
    ]
)
bin2_angles = np.array(
    [
        reg_c1.center.separation(reg_s.center).to_value("arcmin"),
        reg_c2.center.separation(reg_s.center).to_value("arcmin"),
        reg_c3.center.separation(reg_s.center).to_value("arcmin"),
        reg_c4.center.separation(reg_s.center).to_value("arcmin"),
        reg_c1.center.separation(reg_n.center).to_value("arcmin"),
        reg_c2.center.separation(reg_n.center).to_value("arcmin"),
        reg_c3.center.separation(reg_n.center).to_value("arcmin"),
        reg_c4.center.separation(reg_n.center).to_value("arcmin"),
    ]
)
bin3_angles = np.array(
    [
        reg_n.center.separation(reg_s.center).to_value("arcmin"),
    ]
)


bin1_err = np.array([bin1_angles.min(), bin1_angles.max()])
bin1_mid = np.mean(bin1_err)
bin2_err = np.array([bin2_angles.min(), bin2_angles.max()])
bin2_mid = np.mean(bin2_err)
bin3_err = np.array([bin3_angles.min(), bin3_angles.max()])
bin3_mid = np.mean(bin3_err)
bin4_err = np.zeros(2)
bin4_mid = 1.0 / angular_scale.value # 1 Mpc

r_mid = np.array([bin1_mid, bin2_mid, bin3_mid, bin4_mid])
r_min, r_max = np.abs(
    [bin1_err - bin1_mid, bin2_err - bin2_mid, bin3_err - bin3_mid, bin4_err - bin4_mid]
).T

r_mid2 = np.array([bin1_mid, bin2_mid, bin3_mid, bin4_mid])
r_min2, r_max2 = np.abs(
    [bin1_err - bin1_mid, bin2_err - bin2_mid, bin3_err - bin3_mid, [0.0, 0.0]]
).T


if __name__ == "__main__":
    regs = Regions([reg_c1, reg_c2, reg_c3, reg_c4, reg_s, reg_n])
    regs.write("three_pts.reg", format="ds9", overwrite=True)
    #outlines = []
    #for reg in regs:
    #    outlines.append(reg.serialize(format="ds9"))
    #with open("three_pts.reg", "w") as f:
    #    f.writelines(outlines)