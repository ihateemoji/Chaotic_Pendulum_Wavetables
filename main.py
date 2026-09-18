import numpy as np
from scipy.integrate import solve_ivp
import struct
import os

def save_wt(filename, frames, use_full_16bit=False):
    wave_size = len(frames[0])
    wave_count = len(frames)
    flags = 0x0004  # int16
    if use_full_16bit:
        flags |= 0x0008
    with open(filename, "wb") as f:
        f.write(b'vawt')
        f.write(struct.pack('<I', wave_size))
        f.write(struct.pack('<H', wave_count))
        f.write(struct.pack('<H', flags))
        for frame in frames:
            if use_full_16bit:
                scaled = np.clip(frame, -1.0, 1.0) * 32767.0
            else:
                scaled = np.clip(frame, -1.0, 1.0) * 16383.0
            f.write(scaled.astype(np.int16).tobytes())

def normalize(wave):
    peak = np.max(np.abs(wave))
    if peak > 1e-9:
        return wave / peak
    return wave

def soft_clip(x, drive=1.15):
    return np.tanh(x * drive)

def double_pendulum(t, y):
    theta1, dtheta1, theta2, dtheta2 = y
    s1 = np.sin(theta1)
    s12 = np.sin(theta1 - theta2)
    c12 = np.cos(theta1 - theta2)
    den = -16.0 + 9.0 * c12 * c12
    ddtheta1 = (1.0 / den) * (
        -9.0 * (-2.0 + c12) * s1 + \
                3.0 * (2.0 * dtheta2**2 + 3.0 * dtheta1**2 * c12) * s12
    )
    ddtheta2 = (-1.0 / den) * (
        3.0 * ((-8.0 + 9.0 * c12) * s1 + \
                (8.0 * dtheta1**2 + 3.0 * dtheta2**2 * c12) * s12)
    )
    return [dtheta1, ddtheta1, dtheta2, ddtheta2]

def simulate(y0, t_span=(0.0, 80.0), dt=0.005):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        double_pendulum,
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )
    return sol.t, sol.y  # y shape (4, N)

def make_cycle_from_signal(sig, size=2048):
    """Take a 1D signal segment and resample / window it into a single cycle."""
    n = len(sig)
    if n < 4:
        return np.zeros(size, dtype=np.float32)
    # Linear resample to exact size
    x_old = np.linspace(0, 1, n, endpoint=False)
    x_new = np.linspace(0, 1, size, endpoint=False)
    cycle = np.interp(x_new, x_old, sig)
    # Remove DC and normalize
    cycle = cycle - np.mean(cycle)
    cycle = normalize(cycle)
    # Gentle fade at ends to reduce clicks when looping (optional but nice)
    fade = min(32, size // 16)
    if fade > 0:
        env = np.ones(size)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        cycle = cycle * env
        cycle = normalize(cycle)
    return cycle.astype(np.float32)

def extract_frames_from_trajectory(t, state, n_frames=64, size=2048,
                                                            mode="tip_x"):
    """
    Slice the long chaotic trajectory into successive windows and turn each
    into a single-cycle wavetable frame.
    """
    theta1, dth1, theta2, dth2 = state
    # Tip position
    x_tip = np.sin(theta1) + np.sin(theta2)
    y_tip = -np.cos(theta1) - np.cos(theta2)
    # Angular velocities
    # Energy-ish proxy
    frames = []
    N = len(t)
    # Use overlapping or sequential windows after a short transient
    start = int(0.05 * N)  # skip very early transient
    usable = N - start
    win = max(size // 2, usable // (n_frames + 2))  # window length in samples

    for i in range(n_frames):
        # Center of window advances through the trajectory
        center = start + int((i + 0.5) * usable / n_frames)
        a = max(0, center - win // 2)
        b = min(N, a + win)
        if mode == "tip_x":
            sig = x_tip[a:b]
        elif mode == "tip_y":
            sig = y_tip[a:b]
        elif mode == "theta1":
            sig = np.unwrap(theta1[a:b])
        elif mode == "theta2":
            sig = np.unwrap(theta2[a:b])
        elif mode == "dtheta1":
            sig = dth1[a:b]
        elif mode == "dtheta2":
            sig = dth2[a:b]
        elif mode == "combo":
            # Interesting mix: tip radius modulated by angle difference
            r = np.sqrt(x_tip[a:b]**2 + y_tip[a:b]**2)
            dth = theta1[a:b] - theta2[a:b]
            sig = r * np.sin(dth) + 0.4 * dth2[a:b]
        elif mode == "potential":
            sig = -(3*np.cos(theta1[a:b]) + np.cos(theta2[a:b]))
        else:
            sig = x_tip[a:b]

        cycle = make_cycle_from_signal(sig, size=size)
        # Mild soft clip for character
        cycle = soft_clip(cycle, 1.25)
        cycle = normalize(cycle)
        frames.append(cycle)
    return frames

def main():
    out_dir = "Chaotic_Pendulum_Wavetables"
    os.makedirs(out_dir, exist_ok=True)

    # Different initial conditions → different chaotic flavours
    # (angles in radians, velocities zero to start)
    ics = [
        ("Chaos_A", np.array([np.random.uniform(-1, 1)*np.pi, 0.0, \
                                    np.random.uniform(-1, 1)*np.pi, 0.0])),
        ("Chaos_B", np.array([np.random.uniform(-1, 1)*np.pi, 0.0, \
                                    np.random.uniform(-1, 1)*np.pi, 0.0])),
        ("Chaos_C", np.array([np.random.uniform(-1, 1)*np.pi, 0.0, \
                                    np.random.uniform(-1, 1)*np.pi, 0.0])),
        ("Chaos_D", np.array([np.random.uniform(-1, 1)*np.pi, 0.0, \
                                    np.random.uniform(-1, 1)*np.pi, 0.0])),
        ("Chaos_E", np.array([np.random.uniform(-1, 1)*np.pi, 0.0, \
                                    np.random.uniform(-1, 1)*np.pi, 0.0])),
    ]

    modes = [
        ("TipX", "tip_x"),
        ("TipY", "tip_y"),
        ("Theta1", "theta1"),
        ("AngVel", "dtheta2"),
        ("Combo", "combo"),
    ]

    print("Simulating chaotic double pendulums and building wavetables...")
    tables = []
    for ic_name, y0 in ics:
        print(f"  Integrating {ic_name} ...")
        t, state = simulate(y0, t_span=(0.0, 90.0), dt=0.004)
        for mode_name, mode in modes:
            # Only generate a selection to keep the pack focused (~10 tables)
            pass

    # Curated set of 10 interesting tables
    curated = [
        ("01_Chaos_TipX_A",        ics[0][1], "tip_x",   64, 2048),
        ("02_Chaos_TipY_A",        ics[0][1], "tip_y",   64, 2048),
        ("03_Chaos_Combo_A",       ics[0][1], "combo",   64, 2048),
        ("04_Chaos_AngVel_B",      ics[1][1], "dtheta2", 48, 2048),
        ("05_Chaos_Theta1_B",      ics[1][1], "theta1",  64, 2048),
        ("06_Chaos_TipX_C",        ics[2][1], "tip_x",   64, 2048),
        ("07_Chaos_Combo_C",       ics[2][1], "combo",   48, 2048),
        ("08_Chaos_TipY_D",        ics[3][1], "tip_y",   64, 2048),
        ("09_Chaos_Energy_E",      ics[4][1], "potential", 48, 2048),
        ("10_Chaos_AngVel_E",      ics[4][1], "dtheta1", 64, 2048),
    ]

    for name, y0, mode, n_frames, size in curated:
        print(f"  Building {name} ...")
        t, state = simulate(y0, t_span=(0.0, 100.0), dt=0.004)
        frames = extract_frames_from_trajectory(t, state, \
                            n_frames=n_frames, size=size, mode=mode)
        path = os.path.join(out_dir, f"{name}.wt")
        save_wt(path, frames)
        print(f"    → {n_frames} frames × {size} samples")

    readme = """Chaotic Double-Pendulum Wavetables for Surge XT / Cardinal
=========================================================

Generated from the classic chaotic double pendulum system.

Each wavetable is built by:
1. Integrating the double pendulum from a chosen initial condition
2. Taking successive windows of a derived signal (tip position, angle,
   angular velocity, energy proxy, or a combination)
3. Resampling each window into a single-cycle waveform
4. Assembling the cycles into a morphable .wt table

Because the system is chaotic, later frames explore very different regions
of state space from earlier ones → rich, unpredictable morphing behaviour.

Tables
------
01_Chaos_TipX_A.wt      Tip X-coordinate trajectory (IC set A)
02_Chaos_TipY_A.wt      Tip Y-coordinate trajectory (IC set A)
03_Chaos_Combo_A.wt     Radius × sin(Δθ) + angular velocity mix
04_Chaos_AngVel_B.wt    Angular velocity of second pendulum
05_Chaos_Theta1_B.wt    Unwrapped angle of first pendulum
06_Chaos_TipX_C.wt      Tip X from a different energetic IC
07_Chaos_Combo_C.wt     Combo signal, energetic IC
08_Chaos_TipY_D.wt      Tip Y, near-inverted start
09_Chaos_Energy_E.wt    Kinetic + potential energy proxy
10_Chaos_AngVel_E.wt    First pendulum angular velocity

Format: native Surge .wt (16-bit, ~-6 dBFS, power-of-2 frame sizes)

Usage tips
----------
- Morph slowly — chaos makes small morph movements dramatic
- Great for industrial / experimental / glitch / ambient chaos pads
- Try Formant + mild Saturate for extra aggression
- Layer under more conventional oscillators for controlled chaos

Enjoy the butterfly effect in your oscillators.
"""
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write(readme)

    print("\nDone →", out_dir)

if __name__ == "__main__":
    main()
