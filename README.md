# Chaotic Double Pendulum Wavetables

Generates morphing `.wt` wavetables from the chaotic double pendulum system for use in Surge XT, Cardinal, and other wavetable synths.

## What it does

- Simulates the classic chaotic double pendulum
- Extracts different signals from the trajectory (tip position, angles, velocities, energy)
- Turns sliding windows of those signals into single-cycle waveforms
- Packs them into native Surge `.wt` files (64 or 48 frames × 2048 samples)

Because the system is chaotic, later frames explore completely different regions of state space than earlier ones. This produces rich, unpredictable morphing behavior.

## Generated tables

| File                        | Signal source                  | Notes                              |
|----------------------------|--------------------------------|------------------------------------|
| `01_Chaos_TipX_A.wt`       | Tip X coordinate               | IC set A                           |
| `02_Chaos_TipY_A.wt`       | Tip Y coordinate               | IC set A                           |
| `03_Chaos_Combo_A.wt`      | Radius × sin(Δθ) + velocity    | Interesting rhythmic character     |
| `04_Chaos_AngVel_B.wt`     | Angular velocity (pendulum 2)  | Fast, aggressive motion            |
| `05_Chaos_Theta1_B.wt`     | Unwrapped angle (pendulum 1)   | Smooth, evolving                   |
| `06_Chaos_TipX_C.wt`       | Tip X (different IC)           | More energetic trajectory          |
| `07_Chaos_Combo_C.wt`      | Combo signal (energetic IC)    | Complex morphing                   |
| `08_Chaos_TipY_D.wt`       | Tip Y (near-inverted start)    | Wild swings                        |
| `09_Chaos_Energy_E.wt`     | Potential energy proxy         | Organic, breathing motion          |
| `10_Chaos_AngVel_E.wt`     | Angular velocity (pendulum 1)  | Sharp, percussive edges            |

## Requirements

- Python 3
- numpy
- scipy

```bash
pip install numpy scipy
python main.py
```

Output goes to `Chaotic_Pendulum_Wavetables/`.

## Usage tips

- Morph slowly — chaos makes small morph movements dramatic
- Excellent for industrial, experimental, glitch, and ambient pads
- Try Formant + mild Saturate for extra aggression
- Layer under more conventional oscillators for controlled chaos

Enjoy the butterfly effect in your oscillators.