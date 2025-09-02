import astropy.units as u

sqrt2 = 2.0**0.5
zobs = 0.0231
angular_scale = (28.2 * u.kpc / u.arcmin).to("Mpc/arcmin")
resolve_width = 3.05 * u.arcmin
sigma_xrism = ((34.0 * u.arcsec) * angular_scale).to_value("Mpc")
pixel_width = ((1.5 * u.arcmin) * angular_scale).to_value("Mpc")

c_s = 1508.0  # in km/s

sf_stat_err = 30.0
sigma_stat_err = 30.0

r_c = 0.3

l_min0 = 0.001
l_max0 = 1.0
alpha0 = -11.0 / 3.0
