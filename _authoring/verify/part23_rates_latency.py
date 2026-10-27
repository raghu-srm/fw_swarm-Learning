"""Part 23 verification: rates, latencies, message sizes and bandwidth for the swarm bus.
Run: python3 -I part23_rates_latency.py
"""
import numpy as np

print("=== distance flown during one latency (m) ===")
speeds = [20.0, 60.0, 150.0, 250.0]
lats = [0.01, 0.05, 0.1, 0.5, 1.0]
print("V [m/s] |" + "".join(" %6.0f ms" % (l*1e3) for l in lats))
for V in speeds:
    print("%7.0f |" % V + "".join(" %9.1f" % (V*l) for l in lats))

print("\n=== SwarmState message payload (bytes, before DDS/UDP/IP overhead) ===")
fields = [("uav_id", 1), ("timestamp_us", 8), ("seq", 2), ("lat_deg", 8), ("lon_deg", 8), ("alt_amsl_m", 4),
          ("p_n,p_e,p_d (swarm NED)", 12), ("v_n,v_e,v_d", 12), ("course_rad", 4), ("heading_rad", 4),
          ("v_g", 4), ("v_a", 4), ("gamma_rad", 4), ("mode", 1), ("health", 1)]
payload = sum(b for _, b in fields)
for n, b in fields: print("  %-26s %2d B" % (n, b))
print("  payload total = %d B" % payload)
overhead = 100   # assumed RTPS + UDP + IP header budget per packet (documented as an assumption)
for N in (3, 10, 30):
    for rate in (5.0, 10.0, 20.0):
        msgs = N * rate
        bytes_s = msgs * (payload + overhead)
        print("N=%2d UAVs @ %4.1f Hz: %5.0f msg/s, %7.1f kB/s = %6.1f kbit/s (payload %d B + %d B overhead)"
              % (N, rate, msgs, bytes_s/1e3, bytes_s*8/1e3, payload, overhead))

print("\n=== proof-of-life margin: PX4 needs OffboardControlMode at > 2 Hz ===")
for f in (5.0, 10.0, 20.0, 50.0):
    print("publish at %4.1f Hz: period %5.0f ms, margin factor %.0f over 2 Hz, lost packets tolerated before 500 ms gap: %d"
          % (f, 1e3/f, f/2.0, int(0.5*f) - 1))

print("\n=== loop rates used in the lesson (design choice) ===")
for name, hz in (("swarm state broadcast", 10), ("coordination (consensus/formation)", 10),
                 ("guidance + safety filter + PX4 setpoint", 20), ("task allocation (CBBA)", 1)):
    print("  %-40s %3d Hz  (%5.0f ms)" % (name, hz, 1e3/hz))

print("\n=== timestamp arithmetic ===")
ns = 1_700_000_000_123_456_789
print("ROS clock ns = %d -> PX4-style us = %d" % (ns, ns // 1000))
age_us = 250_000
print("message age %d us = %.3f s" % (age_us, age_us/1e6))
