import math
import pymap3d as pm

# Constants as defined in the updated cookie_ekf.c
a = 6378137.0
b = 6356752.3142
e2 = 0.00669437999
ep2 = 0.00673949674

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

def ENU_to_LLA_C_Impl(enu, lat0, lon0, alt0):
    # 1. ENU to ECEF (Implementation from C file)
    phi0 = lat0 * DEG2RAD
    lam0 = lon0 * DEG2RAD
    sphi0 = math.sin(phi0)
    cphi0 = math.cos(phi0)
    slam0 = math.sin(lam0)
    clam0 = math.cos(lam0)

    N0 = a / math.sqrt(1.0 - e2 * sphi0 * sphi0)
    x0 = (N0 + alt0) * cphi0 * clam0
    y0 = (N0 + alt0) * cphi0 * slam0
    z0 = (N0 * (1.0 - e2) + alt0) * sphi0

    dx = -slam0 * enu[0] - sphi0 * clam0 * enu[1] + cphi0 * clam0 * enu[2]
    dy =  clam0 * enu[0] - sphi0 * slam0 * enu[1] + cphi0 * slam0 * enu[2]
    dz =                   cphi0 * enu[1] + sphi0 * enu[2]

    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    # 2. ECEF to LLA (Bowring's method as in C file)
    p = math.sqrt(x*x + y*y)
    # Note: simple atan2(z*a, p*b)
    theta = math.atan2(z * a, p * b)
    sintheta = math.sin(theta)
    costheta = math.cos(theta)
    sintheta3 = sintheta * sintheta * sintheta
    costheta3 = costheta * costheta * costheta

    phi = math.atan2(z + ep2 * b * sintheta3, p - e2 * a * costheta3)
    lam = math.atan2(y, x)

    sphi = math.sin(phi)
    N = a / math.sqrt(1.0 - e2 * sphi * sphi)
    
    # Note: The C code uses p / cos(phi) - N for altitude
    calc_alt = p / math.cos(phi) - N
    
    return [phi * RAD2DEG, lam * RAD2DEG, calc_alt]

def run_tests():
    # Reference point
    lat0, lon0, alt0 = 45.0, 10.0, 100.0
    
    test_cases = [
        [0, 0, 0],         # Origin
        [100, 0, 0],       # 100m East
        [0, 100, 0],       # 100m North
        [0, 0, 100],       # 100m Up
        [1000, 1000, 50],  # Mixed
        [-500, 200, -20],  # Mixed negative
        [10000, 0, 0],     # 10km East
    ]

    print(f"{'ENU Input':<20} | {'C Impl (LLA)':<30} | {'pymap3d (LLA)':<30} | {'Diff (L,L,A)':<30}")
    print("-" * 120)

    for enu in test_cases:
        # Our C implementation
        lla_c = ENU_to_LLA_C_Impl(enu, lat0, lon0, alt0)
        
        # pymap3d implementation
        # pymap3d.enu2geodetic returns lat, lon, alt
        lla_py = pm.enu2geodetic(enu[0], enu[1], enu[2], lat0, lon0, alt0)
        
        diff = [lla_c[0]-lla_py[0], lla_c[1]-lla_py[1], lla_c[2]-lla_py[2]]
        
        enu_str = f"[{enu[0]}, {enu[1]}, {enu[2]}]"
        c_str = f"{lla_c[0]:.6f}, {lla_c[1]:.6f}, {lla_c[2]:.3f}"
        py_str = f"{lla_py[0]:.6f}, {lla_py[1]:.6f}, {lla_py[2]:.3f}"
        diff_str = f"{diff[0]:.2e}, {diff[1]:.2e}, {diff[2]:.2e}"
        
        print(f"{enu_str:<20} | {c_str:<30} | {py_str:<30} | {diff_str:<30}")

if __name__ == "__main__":
    run_tests()
