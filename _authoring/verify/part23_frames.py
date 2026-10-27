"""Part 23 verification: ROS 2 (ENU / FLU) <-> PX4 (NED / FRD) conversions.
PX4 docs (ROS 2 user guide): ENU->NED = rotation of pi/2 about Z (up) followed by pi about X;
FLU->FRD = pi about X.  Here we check that these equal the component swaps used in code.
Run: python3 -I part23_frames.py
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

# A vector expressed in ENU; the same physical vector expressed in NED is (n, e, d) = (y, x, -z)
v_enu = np.array([1.0, 2.0, 3.0])            # east=1, north=2, up=3
v_ned_swap = np.array([v_enu[1], v_enu[0], -v_enu[2]])
# "pi/2 about Z (up), then pi about X" is an intrinsic sequence (second rotation about the
# already-rotated X axis), so the matrix is R_z(pi/2) R_x(pi)
R_enu2ned = Rz(np.pi/2) @ Rx(np.pi)
v_ned_rot = R_enu2ned @ v_enu
print("ENU", v_enu, "-> NED by swap", v_ned_swap, " by rotation", v_ned_rot, " equal:", np.allclose(v_ned_swap, v_ned_rot))
print("R_enu2ned =\n", np.round(R_enu2ned, 3), "\n involutory (R^2 = I):", np.allclose(R_enu2ned @ R_enu2ned, np.eye(3)))

v_flu = np.array([1.0, 2.0, 3.0])            # forward=1, left=2, up=3
v_frd_swap = np.array([v_flu[0], -v_flu[1], -v_flu[2]])
v_frd_rot = Rx(np.pi) @ v_flu
print("FLU", v_flu, "-> FRD by swap", v_frd_swap, " by rotation", v_frd_rot, " equal:", np.allclose(v_frd_swap, v_frd_rot))

# Yaw: ENU yaw psi_enu measured CCW from east (x_ENU); NED heading psi_ned measured CW from north.
for psi_enu_deg in (0.0, 30.0, 90.0, 180.0):
    psi_enu = np.radians(psi_enu_deg)
    fwd_enu = np.array([np.cos(psi_enu), np.sin(psi_enu), 0.0])
    fwd_ned = R_enu2ned @ fwd_enu
    psi_ned = np.arctan2(fwd_ned[1], fwd_ned[0])
    print("yaw ENU %6.1f deg -> heading NED %6.1f deg   (pi/2 - yaw_enu = %6.1f deg)"
          % (psi_enu_deg, np.degrees(psi_ned), np.degrees((np.pi/2 - psi_enu + np.pi) % (2*np.pi) - np.pi)))

# Course from NED velocity (as in the lesson node): chi = atan2(v_e, v_n)
vn, ve = 20.0, 15.0
print("v = (%.0f N, %.0f E) m/s -> V_g = %.2f m/s, chi = %.1f deg" % (vn, ve, np.hypot(vn, ve), np.degrees(np.arctan2(ve, vn))))
