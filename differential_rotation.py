import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

dpsr = 3.3  # pulsar distance from Earth (kpc)
s = 0.25
psi = 50
vism_psi = -47

# J1603 coords
ra = '16 03 35.676751'
dec = '-72 02 32.73991'
c_icrs = SkyCoord('{0} {1}'.format(ra, dec), unit=(u.hourangle, u.deg),
                  frame='icrs')
c_gal = c_icrs.galactic

vscr, vlsr = 220, 220  # km/s
long = c_gal.l.radian
long_ = 2 * np.pi - long

d = (1 - s) * dpsr  # screen distance from Earth
r = 8  # Earth distance from galactic center (kpc)

# radii of screen and J1603 orbits
rscr = np.sqrt(d**2 + r**2 - (2 * d * r * np.cos(long_)))
rpsr = np.sqrt(dpsr**2 + r**2 - (2 * dpsr * r * np.cos(long_)))

# phi = angle between screen orbital velocity and transverse direction
costheta = r / rscr - (d * np.cos(long_) / rscr)
phi = long_ + np.arccos(costheta)

vtrans_scr = vscr * np.cos(phi)  # screen transverse velocity
vtrans_lsr = vlsr * np.cos(long_)  # LSR velocity along same direction

# determining velocity direction on the sky
c_new = SkyCoord(l=c_gal.l.degree + 1, b=c_gal.b.degree, unit=(u.deg, u.deg),
                 frame='galactic')
c_radec = c_new.icrs
ra_new = c_radec.ra.radian
dec_new = c_radec.dec.radian
ra_diff = ra_new - c_icrs.ra.radian
dec_diff = dec_new - c_icrs.dec.radian
angle = (np.pi / 2) - np.arctan(dec_diff / ra_diff)

psi = (psi / 180) * np.pi
diff_vel = (vtrans_scr - vtrans_lsr) * np.cos(angle - psi)

print('differential velocity =', diff_vel)
print('measured vism_psi     =', vism_psi)
print('difference            =', vism_psi - diff_vel)
