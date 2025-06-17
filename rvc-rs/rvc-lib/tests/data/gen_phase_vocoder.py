import torch, json

a = [
    0.84442186, 0.7579544, 0.42057157, 0.25891674,
    0.5112747, 0.40493414, 0.7837986, 0.30331272,
]
b = [
    0.47659695, 0.58338207, 0.9081129, 0.50468683,
    0.28183785, 0.7558042, 0.618369, 0.25050634,
]
fade_out = [
    0.9097463, 0.98278546, 0.81021726, 0.90216595,
    0.31014758, 0.72983176, 0.8988383, 0.6839839,
]
fade_in = [
    0.4721427, 0.10070121, 0.43417183, 0.61088693,
    0.9130111, 0.9666064, 0.47700977, 0.86530995,
]

n = len(a)
a_t = torch.tensor(a)
b_t = torch.tensor(b)
fo_t = torch.tensor(fade_out)
fi_t = torch.tensor(fade_in)
window = torch.sqrt(fo_t * fi_t)
fa = torch.fft.rfft(a_t * window)
fb = torch.fft.rfft(b_t * window)
absab = fa.abs() + fb.abs()
if n % 2 == 0:
    absab[1:-1] *= 2
else:
    absab[1:] *= 2
phia = fa.angle()
phib = fb.angle()
deltaphase = phib - phia
deltaphase -= 2 * torch.pi * torch.floor(deltaphase / (2 * torch.pi) + 0.5)
w = 2 * torch.pi * torch.arange(absab.shape[0]) + deltaphase

def compute():
    result = []
    for i in range(n):
        t = i / n
        summ = torch.sum(absab * torch.cos(w * t + phia))
        val = a_t[i] * fo_t[i] ** 2 + b_t[i] * fi_t[i] ** 2 + summ * window[i] / n
        result.append(float(val))
    return result

case = {
    "a": a,
    "b": b,
    "fade_out": fade_out,
    "fade_in": fade_in,
    "expected": compute(),
}
with open('phase_vocoder_case.json', 'w') as f:
    json.dump(case, f)
