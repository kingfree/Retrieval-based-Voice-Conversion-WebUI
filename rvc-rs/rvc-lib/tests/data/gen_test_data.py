import numpy as np, pyworld

fs = 16000

def save_case(name, signal):
    f0, _ = pyworld.harvest(signal.astype(np.float64), fs=fs, f0_floor=50, f0_ceil=1100, frame_period=10)
    np.save(f"{name}_signal.npy", signal.astype(np.float32))
    np.save(f"{name}_f0.npy", f0.astype(np.float32))

# Case 1: zero signal
save_case("zero", np.zeros(160))

# Case 2: 100 Hz sine for 0.1 second
samples = np.arange(0, int(0.1*fs))
wave = 0.5*np.sin(2*np.pi*100*samples/fs)
save_case("sine100", wave)
