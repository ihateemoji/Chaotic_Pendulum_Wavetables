import numpy as np
from scipy.integrate import solve_ivp
import struct
import os

def save_wt(filename, frames, use_full_16bit=False):
    """Write normalised waveform frames to a binary ``.wt`` file.

    The file begins with a 12-byte little-endian header:

    * ``b"vawt"`` — file signature
    * ``uint32`` — number of samples per frame
    * ``uint16`` — number of frames
    * ``uint16`` — format flags (``0x0004`` for signed 16-bit samples,
      plus ``0x0008`` when full 16-bit scaling is enabled)

    Each frame is expected to contain floating-point samples in the
    ``[-1.0, 1.0]`` range. Samples outside that range are clipped before
    conversion to signed little-endian 16-bit integers. By default, samples
    are scaled by ``16383``; with ``use_full_16bit=True``, they are scaled by
    ``32767`` to use the full positive int16 range.

    Args:
        filename: Destination path for the binary waveform file.
        frames: Non-empty sequence of equally sized, float-like waveform
            frames. Each frame should contain normalised samples.
        use_full_16bit: If ``True``, use full-range int16 scaling and set the
            corresponding format flag. If ``False``, use half-range scaling.
    """
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
    """Scale a waveform so its largest absolute sample reaches unit amplitude.

    The waveform is divided by its peak absolute value, preserving its shape
    and relative sample amplitudes while mapping the largest magnitude to
    approximately ``1.0``. Near-silent waveforms are returned unchanged to
    avoid amplifying numerical noise or dividing by an effectively zero value.

    Args:
        wave: Array-like waveform samples. Typically represented as a NumPy
            array of numeric values.

    Returns:
        A waveform with peak absolute amplitude of approximately ``1.0``,
        or the original waveform unchanged when its peak is at most
        ``1e-9``.
    """
    peak = np.max(np.abs(wave))
    if peak > 1e-9:
        return wave / peak
    return wave

def soft_clip(x, drive=1.15):
    """Apply smooth, nonlinear saturation to a signal.

    The input is multiplied by ``drive`` and passed through the hyperbolic
    tangent function. Small signals remain approximately linear, while larger
    amplitudes are progressively compressed toward the bounded range
    ``(-1.0, 1.0)``. This produces a softer alternative to hard clipping and
    can help create mild analogue-style overdrive or saturation.

    Args:
        x: Scalar or array-like signal values.
        drive: Input gain applied before saturation. Larger values produce
            stronger compression and more pronounced distortion. Must be
            finite for predictable results.

    Returns:
        A NumPy array or scalar containing the softly clipped signal, bounded
        to ``(-1.0, 1.0)``.
    """
    return np.tanh(x * drive)

def double_pendulum(t, y):
    """Compute the state derivatives for a normalised double pendulum.

    This function models a planar double pendulum using angular coordinates
    and angular velocities. The state vector is ordered as::

        y = [theta1, dtheta1, theta2, dtheta2]

    where ``theta1`` and ``theta2`` are the angles of the two pendulum arms
    and ``dtheta1`` and ``dtheta2`` are their corresponding angular
    velocities. The returned vector has the same ordering and contains:

        [dtheta1, ddtheta1, dtheta2, ddtheta2]

    The equations include gravitational coupling, relative-angle coupling,
    and velocity-dependent terms between the two pendulum arms. The
    formulation assumes fixed, normalised physical parameters; masses,
    lengths, and gravitational acceleration are incorporated into the
    constants in the equations.

    Args:
        t: Current integration time. Included for compatibility with ODE
            solvers; the system is autonomous and does not explicitly
            depend on time.
        y: Four-element state vector containing the two angular positions
            and their angular velocities, in the order
            ``[theta1, dtheta1, theta2, dtheta2]``.

    Returns:
        list: The time derivative of the state vector in the order
        ``[dtheta1, ddtheta1, dtheta2, ddtheta2]``.
    """
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
    """Numerically integrate the double-pendulum equations over time.

    The system is integrated with SciPy's adaptive ``RK45`` solver, while
    results are sampled at uniformly spaced times separated by ``dt``.
    Although the solver chooses its own internal step sizes, the returned
    arrays contain only the requested evaluation times.

    Args:
        y0: Initial four-element state vector in the order
            ``[theta1, dtheta1, theta2, dtheta2]``.
        t_span: Two-element tuple ``(t_start, t_end)`` defining the
            integration interval.
        dt: Spacing between consecutive output times. This controls the
            sampling resolution, not the adaptive solver's internal step
            size.

    Returns:
        tuple:
            ``t``: One-dimensional array of sampled time values with shape
                ``(N,)``.
            ``y``: State history with shape ``(4, N)``. Each row contains
                one state variable, and each column corresponds to a time
                value in ``t``.
    """
    print("\nSimulating trajectory with initial state:")
    print(y0)
    print("\n")
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
    """Convert a one-dimensional signal segment into a loopable waveform cycle.

    The input segment is linearly resampled to exactly ``size`` samples,
    centred by removing its DC offset, and peak-normalised. A short linear
    fade is then applied at both ends to reduce discontinuities and audible
    clicks when the resulting cycle is looped. The cycle is normalised again
    after fading because the envelope may reduce its peak amplitude.

    Signals shorter than four samples cannot provide a useful cycle and
    produce a zero-filled waveform instead.

    Args:
        sig: One-dimensional, array-like signal segment. Its samples should
            be numeric and arranged in the order in which they should appear
            within the cycle.
        size: Number of samples in the returned cycle. Defaults to ``2048``.

    Returns:
        numpy.ndarray: A ``float32`` waveform containing exactly ``size``
        samples. For a valid input, the result is approximately centred
        around zero and peak-normalised; for an input shorter than four
        samples, it contains only zeros.
    """
    n = len(sig)
    if n < 4:
        return np.zeros(size, dtype=np.float32)
    # Linear resample to exact size
    x_old = np.linspace(0, 1, n, endpoint=False)
    x_new = np.linspace(0, 1, size, endpoint=False)
    cycle = np.interp(x_new, x_old, sig)
    # Remove DC and normalise
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
    """Extract normalised wavetable frames from a simulated trajectory.

    The trajectory is divided into windows after skipping the first 5% of
    samples to avoid early transients. Each window is converted into one
    loopable cycle with ``make_cycle_from_signal``, gently soft-clipped for
    additional character, and peak-normalised before being added to the
    output.

    Args:
        t: One-dimensional time array. Only its length is used to determine
            the trajectory length; the time values themselves are not
            consulted.
        state: State history with shape ``(4, N)`` and rows ordered as
            ``[theta1, dtheta1, theta2, dtheta2]``.
        n_frames: Number of wavetable frames to extract.
        size: Number of samples in each output frame.
        mode: Signal derived from the trajectory for each window. Supported
            values are:

            * ``"tip_x"`` — horizontal position of the pendulum tip.
            * ``"tip_y"`` — vertical position of the pendulum tip.
            * ``"theta1"`` — unwrapped angle of the first arm.
            * ``"theta2"`` — unwrapped angle of the second arm.
            * ``"dtheta1"`` — angular velocity of the first arm.
            * ``"dtheta2"`` — angular velocity of the second arm.
            * ``"combo"`` — a nonlinear combination of tip radius, angle
              difference, and second-arm angular velocity.
            * ``"potential"`` — a cosine-based potential-energy proxy.

            Unrecognised values fall back to ``"tip_x"``.

    Returns:
        list: A list containing ``n_frames`` NumPy ``float32`` arrays, each
        with shape ``(size,)``. Frames are centred, normalised, softly
        clipped, and suitable for use as wavetable cycles.
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
        # Centre of window advances through the trajectory
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
    # Set of 10 interesting tables
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
    print("\nDone →", out_dir)

if __name__ == "__main__":
    main()
