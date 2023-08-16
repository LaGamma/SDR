import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# error formula for 2nd order Costas Loop
def costas_BPSK_err(sample):
    return np.real(sample) * np.imag(sample)

# error formula for 4nd order Costas Loop
def costas_QPSK_err(sample):
    test_res = [-1.0, 1.0]
    a = test_res[sample.real > 0]
    b = test_res[sample.imag > 0]
    return a * sample.imag - b * sample.real

# Costas loop
def costas(freq, phase, samples):
    # These next two params is what to adjust, to make the feedback loop faster or slower (impacts stability)
    alpha = 0.05
    beta = 0.01
    out = np.zeros(len(samples), dtype=complex)
    for i in range(len(out)):
        out[i] = samples[i] * np.exp(-1j*phase) # adjust the input sample by inverse of estimated phase offset
        error = costas_QPSK_err(out[i])
        # Advance the loop (recalc phase and freq offset)
        freq += (beta * error)
        phase += freq + (alpha * error)
        while phase > 2*np.pi: phase -= 2*np.pi
        while phase < 0: phase += 2*np.pi
    return freq, phase, out

class SimPLL():
    def __init__(self, lf_bandwidth):
        self.freq_offset = 0.0
        self.phase_out = 0.0
        self.freq_out = 0.0
        self.vco = np.exp(1j*self.phase_out)
        self.phase_difference = 0.0
        self.bw = lf_bandwidth
        self.beta = np.sqrt(lf_bandwidth)

    def update_phase_estimate(self):
        self.vco = np.exp(1j*self.phase_out)

    def update_phase_difference(self, in_sig):
        self.phase_difference = np.angle(in_sig*np.conj(self.vco))

    def step(self, in_sig):
        # Takes an instantaneous sample of a signal and updates the PLL's inner state
        self.update_phase_difference(in_sig)
        self.freq_out += self.bw * self.phase_difference
        self.phase_out += self.beta * self.phase_difference + self.freq_out
        self.update_phase_estimate()
        return self.vco, self.freq_out #self.phase_difference

AMP = [1, 1, 1, 1]
C_FREQ = [500, 500, 500, 500] #100000] # in Hz
PHASE = [0, np.pi, 3*np.pi/2, np.pi/2]#3*np.pi/2] #np.pi, 3*np.pi/2]

samp_r = 1e6
num_samp = 2048 # also fft size
Ts = np.arange(num_samp) / samp_r
# simulate sinusoids
signals = [AMP[i]*np.exp(1j*(2*np.pi*C_FREQ[i]*Ts+PHASE[i])) for i in range(4)]
# complex noise with unity power
noise = (np.random.randn(num_samp) + 1j*np.random.randn(num_samp)) / np.sqrt(2)
noise_power = 2
r_signal = sum(signals) + noise * np.sqrt(noise_power)

pll = SimPLL(0.002)
other_sig = [pll.step(samp) for samp in r_signal]
freq,phase,other_sig = costas(0,0,r_signal)

BIG_SIZE = 10
total_x = np.zeros(len(Ts) * BIG_SIZE)
total_y = np.zeros(len(r_signal) * BIG_SIZE)

total_z = np.zeros(len(r_signal) * BIG_SIZE)
ttz = np.zeros(len(r_signal) * BIG_SIZE)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
def animus(i):
    global total_y, total_x, total_z, ttz, signals, phase, freq
    Ts = np.arange(num_samp*i, num_samp*(i+1)) / samp_r
    # simulate sinusoids
    signals = [AMP[i]*np.exp(1j*(2*np.pi*C_FREQ[i]*Ts+PHASE[i])) for i in range(len(signals))]
    # complex noise with unity power
    noise = (np.random.randn(num_samp) + 1j*np.random.randn(num_samp)) / np.sqrt(2)
    noise_power = 0.1

    r_signal = signals[(i//3)%len(signals)] + noise * np.sqrt(noise_power)
    other_sig = [pll.step(samp) for samp in r_signal]
    #other_sig0 = [other_sig[i][0] for i in range(len(other_sig))]
    #other_sig1 = [other_sig[i][1] for i in range(len(other_sig))]

    # Create our raised-cosine filter
    sps = 8
    beta = 0.35
    t = np.arange(-51, 52) # remember it's not inclusive of final number
    h = np.sinc(t/sps) * np.cos(np.pi*beta*t/sps) / (1 - (2*beta*t/sps)**2)
    # Filter our signal, in order to apply the pulse shaping
    r_signal = np.convolve(r_signal, h, mode='same')

    freq,phase,other_sig0 = costas(freq,phase,r_signal)

    total_x = np.hstack((total_x, Ts))[num_samp:]
    total_y = np.hstack((total_y, r_signal))[num_samp:]
    total_z = np.hstack((total_z, other_sig0))[num_samp:]
    #ttz = np.hstack((ttz, other_sig1))[num_samp:]
    ax1.clear()
    ax1.plot(total_x, total_y)
    ax1.set_xlabel("Time [s]")

    ax3.clear()
    ax3.plot(np.real(other_sig0), np.imag(other_sig0), '.')
    ax3.axis([-20, 20, -20, 20])
    ax3.set_ylabel('Q')
    ax3.set_xlabel('I')

    ax4.clear()
    ax4.plot(total_x, total_z)
    #ax4.plot(total_x, ttz)

    # now do the frequency domain
    PSD = np.abs(np.fft.fft(r_signal))**2 / (num_samp * samp_r)
    PSD_log = 10.0*np.log10(PSD)
    PSD_shifted = np.fft.fftshift(PSD_log)
    f = np.arange(samp_r/-2.0, samp_r/2.0, samp_r/num_samp)
    ax2.clear()
    ax2.plot(f, PSD_shifted)
    ax2.set_xlabel("Frequency [Hz]")

ani = animation.FuncAnimation(fig, animus, interval=10)
plt.show()
