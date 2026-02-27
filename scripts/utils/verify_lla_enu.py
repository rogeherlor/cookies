import math
import pymap3d as pm

# Constants as defined in the updated cookie_ekf.c
a = 6378137.0
b = 6356752.3142
e2 = 0.00669437999
ep2 = 0.00673949674

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

def LLA_to_ENU_C_Impl(lat, lon, alt, lat0, lon0, alt0):
    # 1. LLA to ECEF
    phi = lat * DEG2RAD
    lam = lon * DEG2RAD
    sphi = math.sin(phi)
    cphi = math.cos(phi)
    slam = math.sin(lam)
    clam = math.cos(lam)
    
    N = a / math.sqrt(1.0 - e2 * sphi * sphi)
    x = (N + alt) * cphi * clam
    y = (N + alt) * cphi * slam
    z = (N * (1.0 - e2) + alt) * sphi

    # Reference ECEF
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

    # 2. ECEF Delta
    dx = x - x0
    dy = y - y0
    dz = z - z0

    # 3. ECEF to ENU Rotation
    e = -slam0 * dx + clam0 * dy
    n = -sphi0 * clam0 * dx - sphi0 * slam0 * dy + cphi0 * dz
    u = cphi0 * clam0 * dx + cphi0 * slam0 * dy + sphi0 * dz
    
    return [e, n, u]

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
    
    print("\n--- Testing ENU to LLA (Bowring) ---")
    test_cases_enu = [
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

    for enu in test_cases_enu:
        # Our C implementation
        lla_c = ENU_to_LLA_C_Impl(enu, lat0, lon0, alt0)
        # pymap3d implementation
        lla_py = pm.enu2geodetic(enu[0], enu[1], enu[2], lat0, lon0, alt0)
        
        diff = [lla_c[0]-lla_py[0], lla_c[1]-lla_py[1], lla_c[2]-lla_py[2]]
        
        enu_str = f"[{enu[0]}, {enu[1]}, {enu[2]}]"
        c_str = f"{lla_c[0]:.6f}, {lla_c[1]:.6f}, {lla_c[2]:.3f}"
        py_str = f"{lla_py[0]:.6f}, {lla_py[1]:.6f}, {lla_py[2]:.3f}"
        diff_str = f"{diff[0]:.2e}, {diff[1]:.2e}, {diff[2]:.2e}"
        print(f"{enu_str:<20} | {c_str:<30} | {py_str:<30} | {diff_str:<30}")
    
    print("\n--- Testing LLA to ENU ---")
    # Generate test cases using pymap3d from ENU points above to get valid LLAs
    test_cases_lla = []
    # Add origin
    test_cases_lla.append([lat0, lon0, alt0])
    # Add other points
    offset_lla = pm.enu2geodetic(100, 100, 100, lat0, lon0, alt0)
    test_cases_lla.append([offset_lla[0], offset_lla[1], offset_lla[2]])
    offset_lla2 = pm.enu2geodetic(-100, -100, -50, lat0, lon0, alt0)
    test_cases_lla.append([offset_lla2[0], offset_lla2[1], offset_lla2[2]])

    print(f"{'LLA Input':<35} | {'C Impl (ENU)':<30} | {'pymap3d (ENU)':<30} | {'Diff (E,N,U)':<30}")
    print("-" * 129)
    
    for lla in test_cases_lla:
        lat, lon, alt = lla[0], lla[1], lla[2]
        
        # Our C implementation
        enu_c = LLA_to_ENU_C_Impl(lat, lon, alt, lat0, lon0, alt0)
        
        # pymap3d implementation
        enu_py = pm.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
        
        diff = [enu_c[0]-enu_py[0], enu_c[1]-enu_py[1], enu_c[2]-enu_py[2]]
        
        lla_str = f"[{lat:.5f}, {lon:.5f}, {alt:.1f}]"
        c_str = f"{enu_c[0]:.3f}, {enu_c[1]:.3f}, {enu_c[2]:.3f}"
        py_str = f"{enu_py[0]:.3f}, {enu_py[1]:.3f}, {enu_py[2]:.3f}"
        diff_str = f"{diff[0]:.2e}, {diff[1]:.2e}, {diff[2]:.2e}"
        
        print(f"{lla_str:<35} | {c_str:<30} | {py_str:<30} | {diff_str:<30}")

if __name__ == "__main__":
    run_tests()
