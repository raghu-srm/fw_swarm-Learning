import numpy as np
g = 9.81

def envelope(Vg, phi_max=None, nz_max=None, V_stall=None):
    """Turn envelope of a coordinated level turn at ground speed Vg.
    Limits: bank phi_max (comfort / autopilot), structure nz_max, and the lift limit
    n_aero = (Vg/V_stall)^2.  Returns a dict with the binding limit."""
    cands = {}
    if phi_max is not None:
        cands['bank'] = np.tan(phi_max)
    if nz_max is not None:
        cands['structure'] = np.sqrt(nz_max**2 - 1.0)
    if V_stall is not None:
        n_aero = (Vg / V_stall)**2
        cands['lift'] = np.sqrt(max(n_aero**2 - 1.0, 0.0))     # tan(phi) = sqrt(nz^2 - 1)
    limit = min(cands, key=cands.get)                        # smallest tan(phi) wins
    tan_phi = cands[limit]
    if tan_phi <= 0.0:                                        # below stall: no turn possible
        return dict(limit='stall', tan_phi=0.0, nz=1.0, omega_max=0.0, R_min=np.inf, a_lat=0.0)
    return dict(limit=limit, tan_phi=tan_phi, phi=np.arctan(tan_phi),
                nz=np.sqrt(1.0 + tan_phi**2), omega_max=g*tan_phi/Vg,
                R_min=Vg**2/(g*tan_phi), a_lat=g*tan_phi)

def corner_speed(V_stall, nz_max):
    return V_stall*np.sqrt(nz_max)

def schedule_gains(Vg, env, kappa=0.5, sep=3.0):
    """Formation gain k_p = kappa*omega_max, course-loop gain b_chi = sep*k_p (time-scale separation)."""
    k_p = kappa*env['omega_max']
    return dict(k_p=k_p, b_chi=sep*k_p, tau_formation=1.0/k_p,
                e_linear=env['R_min']/kappa**2)              # cross-track error that saturates the turn rate

if __name__ == '__main__':
    print("--- brief check values")
    for Vg, phi in [(25, 30), (100, 30)]:
        e = envelope(Vg, phi_max=np.radians(phi))
        print(f"Vg={Vg:4d} phi={phi}: R={e['R_min']:7.1f} m  omega={np.degrees(e['omega_max']):5.2f} deg/s  nz={e['nz']:.3f}")
    e = envelope(200, nz_max=4.0)
    print(f"Vg=200 nz_max=4: R_min={e['R_min']:.1f} m  omega_max={np.degrees(e['omega_max']):.2f} deg/s  phi_max={np.degrees(e['phi']):.1f} deg")
    print("--- high-speed aircraft: V_stall=60, nz_max=4  (corner speed %.1f m/s)" % corner_speed(60.0, 4.0))
    print("   Vg   limit      nz    omega_max[deg/s]   R_min[m]   k_p[1/s]  tau_f[s]  e_lin[m]")
    for Vg in [70, 100, 120, 150, 200, 250]:
        e = envelope(Vg, nz_max=4.0, V_stall=60.0); s = schedule_gains(Vg, e)
        print(f"  {Vg:4d}  {e['limit']:9s}  {e['nz']:5.2f}  {np.degrees(e['omega_max']):10.2f}  {e['R_min']:12.1f}"
              f"  {s['k_p']:9.4f}  {s['tau_formation']:7.2f}  {s['e_linear']:8.0f}")
    print("--- latency x speed [m]")
    taus = [0.02, 0.1, 0.5, 1.0]
    for Vg in [20, 60, 150, 250]:
        print(f"  V={Vg:4d}: " + "  ".join(f"tau={t:4.2f}s -> {Vg*t:6.1f}" for t in taus))
