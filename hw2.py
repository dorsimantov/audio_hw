import numpy as np
import os
import random
import librosa
from metrics import pesq, stoi, si_sdr
from rir_generator import generate
from scipy.signal import convolve
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
import librosa.display

# Constants
C = 343  # Speed of sound (m/s)
ROOM_DIMENSIONS = [4, 5, 3]  # Room dimensions (meters)
MICROPHONE_POSITION = [2, 1, 1.7]  # Center of the microphone array
MIC_DISTANCE = 0.05  # 5 cm between microphones
NUM_MICROPHONES = 5
SOURCE_ANGLE = 30  # Direction of desired source speaker (degrees)
INTERFERENCE_ANGLE = 150  # Direction of interfering speaker (degrees)
T60_VALUES = [0.15, 0.3]  # Reverberation times in seconds
FS = 16000  # Sampling frequency (Hz)
WAVELENGTH = C / FS  # Approximate wavelength of sound (meter / sample)
SNR_VALUES = [0, 10]  # Signal-to-noise ratio values in dB
OUTPUT_DIR = "output/"
DATA_DIR = "data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load random 20 FLAC files from data directory
flac_files = []
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".flac"):
            flac_files.append(os.path.join(root, file))

selected_files = random.sample(flac_files, 20)
clean_signals = []
for file in selected_files:
    signal, _ = librosa.load(file, sr=FS)
    clean_signals.append(signal)

# Generate RIRs for source
mic_positions = np.zeros((NUM_MICROPHONES, 3))
for i in range(NUM_MICROPHONES):
    mic_positions[i, :] = [
        MICROPHONE_POSITION[0] + (i - (NUM_MICROPHONES - 1) / 2) * MIC_DISTANCE,
        MICROPHONE_POSITION[1],
        MICROPHONE_POSITION[2],
    ]

source_position = [
    MICROPHONE_POSITION[0] + 1.5 * np.cos(np.radians(SOURCE_ANGLE)),
    MICROPHONE_POSITION[1] + 1.5 * np.sin(np.radians(SOURCE_ANGLE)),
    MICROPHONE_POSITION[2],
]

rir_list = []
for t60 in T60_VALUES:
    rir = generate(
        c=343,
        fs=FS,
        r=mic_positions,
        s=source_position,
        L=ROOM_DIMENSIONS,
        reverberation_time=t60,
        nsample=int(t60 * FS)
    )
    rir_list.append(rir)

# Generate RIRs for interfering signal
interfering_position = [
    MICROPHONE_POSITION[0] + 2.0 * np.cos(np.radians(INTERFERENCE_ANGLE)),
    MICROPHONE_POSITION[1] + 2.0 * np.sin(np.radians(INTERFERENCE_ANGLE)),
    MICROPHONE_POSITION[2],
]

interfering_rir_list = []
for t60 in T60_VALUES:
    rir = generate(
        c=343,
        fs=FS,
        r=mic_positions,
        s=interfering_position,
        L=ROOM_DIMENSIONS,
        reverberation_time=t60,
        nsample=int(t60 * FS)
    )
    interfering_rir_list.append(rir)

# List of instances, one for each speaker
instances = []

for clean_signal_index, clean_signal in enumerate(clean_signals):
    instance_dict = {"clean": {}, "noisy": {"gaussian": {}, "interfering": {}}}
    instance_dir = os.path.join(OUTPUT_DIR, f"instance_{clean_signal_index}")
    os.makedirs(instance_dir, exist_ok=True)

    for t60_index, rir in enumerate(rir_list):
        t60 = T60_VALUES[t60_index]
        mic_signals = np.array([
            convolve(clean_signal, rir[:, i])[: len(clean_signal)] for i in range(NUM_MICROPHONES)
        ])
        instance_dict["clean"][f"t60_{int(t60 * 1000)}ms"] = mic_signals

        # Save clean signals
        clean_dir = os.path.join(instance_dir, "clean", f"t60_{int(t60 * 1000)}ms")
        os.makedirs(clean_dir, exist_ok=True)
        for mic_idx, signal in enumerate(mic_signals):
            write(os.path.join(clean_dir, f"mic_{mic_idx}.wav"), FS, signal.astype(np.float32))

        for snr in SNR_VALUES:
            noise_power = np.var(mic_signals[0]) / (10 ** (snr / 10))
            gaussian_noise = np.sqrt(noise_power) * np.random.randn(*mic_signals.shape)
            noisy_gaussian = mic_signals + gaussian_noise

            interfering_signal = random.choice(clean_signals)
            interfering_rir = interfering_rir_list[t60_index]
            interfering_convolved = [
                convolve(interfering_signal, interfering_rir[:, i]) for i in range(NUM_MICROPHONES)
            ]
            interfering_convolved = np.array([arr[: len(clean_signal)] for arr in interfering_convolved])

            if interfering_convolved.shape[1] > mic_signals.shape[1]:
                interfering_convolved = interfering_convolved[:, :mic_signals.shape[1]]
            elif interfering_convolved.shape[1] < mic_signals.shape[1]:
                interfering_convolved = np.pad(interfering_convolved,
                                               ((0, 0), (0, mic_signals.shape[1] - interfering_convolved.shape[1])),
                                               mode='constant')

            scale_factor = np.sqrt(np.var(mic_signals[0]) / np.var(interfering_convolved[0])) / (10 ** (snr / 20))
            interfering_noise = interfering_convolved * scale_factor

            noisy_interfering = mic_signals + interfering_noise

            if f"snr_{snr}dB" not in instance_dict["noisy"]["gaussian"]:
                instance_dict["noisy"]["gaussian"][f"snr_{snr}dB"] = {}
            if f"snr_{snr}dB" not in instance_dict["noisy"]["interfering"]:
                instance_dict["noisy"]["interfering"][f"snr_{snr}dB"] = {}

            instance_dict["noisy"]["gaussian"][f"snr_{snr}dB"][f"t60_{int(t60 * 1000)}ms"] = noisy_gaussian
            instance_dict["noisy"]["interfering"][f"snr_{snr}dB"][f"t60_{int(t60 * 1000)}ms"] = noisy_interfering

            # Save Gaussian noise signals
            gaussian_dir = os.path.join(instance_dir, "noisy", "gaussian", f"snr_{snr}dB", f"t60_{int(t60 * 1000)}ms")
            os.makedirs(gaussian_dir, exist_ok=True)
            for mic_idx, signal in enumerate(noisy_gaussian):
                write(os.path.join(gaussian_dir, f"mic_{mic_idx}.wav"), FS, signal.astype(np.float32))

            # Save interfering noise signals
            interfering_dir = os.path.join(instance_dir, "noisy", "interfering", f"snr_{snr}dB",
                                           f"t60_{int(t60 * 1000)}ms")
            os.makedirs(interfering_dir, exist_ok=True)
            for mic_idx, signal in enumerate(noisy_interfering):
                write(os.path.join(interfering_dir, f"mic_{mic_idx}.wav"), FS, signal.astype(np.float32))

    instances.append(instance_dict)

# Plot STFTs for a sample case (T60 = 300ms, SNR = 10dB)
clean_signal = clean_signals[0]
mic_signals = instances[0]["clean"]["t60_300ms"]
noisy_gaussian = instances[0]["noisy"]["gaussian"]["snr_10dB"]["t60_300ms"]
noisy_interfering = instances[0]["noisy"]["interfering"]["snr_10dB"]["t60_300ms"]

plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(clean_signal)), ref=np.max), sr=FS, x_axis="time",
                         y_axis="log")
plt.title("Clean Speech Signal")

plt.subplot(4, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(mic_signals[0])), ref=np.max), sr=FS,
                         x_axis="time", y_axis="log")
plt.title("Clean Signal at First Microphone")

plt.subplot(4, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(noisy_gaussian[0])), ref=np.max), sr=FS,
                         x_axis="time", y_axis="log")
plt.title("Signal at First Microphone with Gaussian Noise (SNR 10 dB)")

plt.subplot(4, 1, 4)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(noisy_interfering[0])), ref=np.max), sr=FS,
                         x_axis="time", y_axis="log")
plt.title("Signal at First Microphone with Interfering Speaker (SNR 10 dB)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}stft_signals.png")
plt.show()

print("FINISHED PROCESSING QUESTION 1")

########## QUESTION 2 ###########

import numpy as np
from scipy.linalg import eigh
from scipy.io.wavfile import write
from scipy import signal
import librosa.display
import matplotlib.pyplot as plt


# Helper function to align lengths
def align_length(reference, degraded):
    ref_len = len(reference)
    degraded_len = len(degraded)
    if ref_len > degraded_len:
        return reference[:degraded_len], degraded
    elif degraded_len > ref_len:
        return reference, degraded[:ref_len]
    else:
        return reference, degraded


def delay_and_sum_beamformer(stfts, source_angle):
    freqs, times, stft_matrix = stfts
    num_mics, num_freqs, num_time_frames = stft_matrix.shape

    # Compute wavelength (avoid division by zero)
    valid_freqs = freqs[freqs > 0]  # Exclude zero frequency
    wavelength = np.concatenate((np.array([np.inf]), C / valid_freqs))

    # Compute steering vector
    steering_vector = np.zeros((num_freqs, num_mics), dtype=np.complex64)

    for mic_idx in range(num_mics):
        steering_vector[:, mic_idx] = np.exp(
            -2j * np.pi * mic_idx * MIC_DISTANCE / wavelength[:, None] * np.cos(
                np.radians(source_angle))
        ).flatten() / NUM_MICROPHONES

    # Apply weights to STFTs and sum across microphones
    weighted_stfts = np.conj(steering_vector.T[:, :, None]) * stft_matrix
    beamformed_stft = np.sum(weighted_stfts, axis=0)  # Sum over microphones

    # Apply inverse STFT to reconstruct time-domain signal
    _, beamformed_signal = signal.istft(beamformed_stft, fs=FS)

    return beamformed_signal


# Helper function to compute STFTs for all microphones
def compute_stfts(signals, fs, nperseg=512):
    freqs, times, stft_matrices = [], [], []
    for mic_signal in signals:
        f, t, Zxx = signal.stft(mic_signal, fs=fs, nperseg=nperseg)
        freqs = f  # All microphones share the same frequency bins
        times = t  # All microphones share the same time bins
        stft_matrices.append(Zxx)
    return freqs, times, np.stack(stft_matrices, axis=0)  # Shape: (num_mics, freq_bins, time_frames)


# Fix for the MVDR function
def mvdr_beamformer(noisy_stft, noise_stft):
    freqs, times, noisy_matrix = noisy_stft
    _, _, noise_matrix = noise_stft

    # Initialize beamformed STFT matrix
    beamformed_stft = np.zeros((len(freqs), len(times)), dtype=complex)

    # Process each frequency bin independently
    for f_idx in range(len(freqs)):
        # Compute covariance matrices for the current frequency bin
        noisy_cov = np.cov(noisy_matrix[:, f_idx, :])
        noise_cov = np.cov(noise_matrix[:, f_idx, :])

        # Estimate RTF using GEVD
        eigvals, eigvecs = eigh(noisy_cov, noise_cov)
        principal_eigvec = eigvecs[:, np.argmax(eigvals)]

        # De-whiten the eigenvector using the noise covariance matrix
        dewhitened_eigvec = noise_cov @ principal_eigvec

        # Normalize the de-whitened eigenvector by the reference microphone component
        reference_microphone_component = dewhitened_eigvec[0]
        rtf = dewhitened_eigvec / reference_microphone_component

        # Compute MVDR weights
        inv_noise_cov = np.linalg.inv(noise_cov)
        numerator = inv_noise_cov @ rtf
        denominator = rtf.conj().T @ inv_noise_cov @ rtf
        mvdr_weights = numerator / denominator

        # Apply MVDR weights to the noisy STFT
        beamformed_stft[f_idx, :] = mvdr_weights.conj().T @ noisy_matrix[:, f_idx, :]

    # ISTFT to return to time-domain
    _, beamformed_signal = signal.istft(beamformed_stft, fs=FS, nperseg=512)
    return beamformed_signal


def estimate_rtf_gevd(noisy_cov, noise_cov):
    # Apply GEVD to noisy and noise covariance matrices
    eigvals, eigvecs = eigh(noisy_cov, noise_cov)

    # Get the eigenvector corresponding to the largest eigenvalue
    principal_eigvec = eigvecs[:, np.argmax(eigvals)]

    # De-whitening the eigenvector by multiplying with the noise covariance
    dewhitened_eigvec = noise_cov @ principal_eigvec

    # Normalize the de-whitened eigenvector by the reference microphone component
    reference_microphone_component = dewhitened_eigvec[0]

    rtf = dewhitened_eigvec / reference_microphone_component

    return rtf


# Metrics (PESQ, ESTOI, SI-SDR)
def compute_metrics(reference, estimated):
    reference, estimated = align_length(reference, estimated)
    pesq_score = pesq(FS, reference, estimated)
    estoi_score = stoi(reference, estimated, FS, extended=True)
    si_sdr_score = si_sdr(reference, estimated)
    return pesq_score, estoi_score, si_sdr_score


# Initialize metric accumulators
metrics_summary = {
    "das_gaussian": {"pesq": [], "estoi": [], "si_sdr": []},
    "mvdr_gaussian": {"pesq": [], "estoi": [], "si_sdr": []},
    "das_interfering": {"pesq": [], "estoi": [], "si_sdr": []},
    "mvdr_interfering": {"pesq": [], "estoi": [], "si_sdr": []},
}

# Process all instances
for instance_index, instance in enumerate(instances):
    mic_signals = instance["clean"]["t60_300ms"]
    noisy_gaussian = instance["noisy"]["gaussian"]["snr_10dB"]["t60_300ms"]
    noisy_interfering = instance["noisy"]["interfering"]["snr_10dB"]["t60_300ms"]

    noisy_gaussian_stft = compute_stfts(noisy_gaussian, fs=FS, nperseg=512)
    noise_gaussian_stft = compute_stfts(noisy_gaussian - mic_signals, fs=FS, nperseg=512)

    noisy_interfering_stft = signal.stft(noisy_interfering, fs=FS, nperseg=512)
    noise_interfering_stft = signal.stft(noisy_interfering - mic_signals, fs=FS, nperseg=512)

    # Delay-and-Sum Beamforming
    das_output_gaussian = delay_and_sum_beamformer(noisy_gaussian_stft, SOURCE_ANGLE)
    das_output_interfering = delay_and_sum_beamformer(noisy_interfering_stft, SOURCE_ANGLE)

    # MVDR Beamforming
    mvdr_output_gaussian = mvdr_beamformer(noisy_gaussian_stft, noise_gaussian_stft)
    mvdr_output_interfering = mvdr_beamformer(noisy_interfering_stft, noise_interfering_stft)

    # Save outputs
    write(f"{OUTPUT_DIR}instance_{instance_index}_das_output_gaussian.wav", FS, das_output_gaussian.astype(np.float32))
    write(f"{OUTPUT_DIR}instance_{instance_index}_mvdr_output_gaussian.wav", FS,
          mvdr_output_gaussian.astype(np.float32))
    write(f"{OUTPUT_DIR}instance_{instance_index}_das_output_interfering.wav", FS,
          das_output_interfering.astype(np.float32))
    write(f"{OUTPUT_DIR}instance_{instance_index}_mvdr_output_interfering.wav", FS,
          mvdr_output_interfering.astype(np.float32))

    # Compute metrics for current instance
    reference_signal = mic_signals[0]  # Clean signal from the first microphone
    das_gaussian_metrics = compute_metrics(reference_signal, das_output_gaussian)
    mvdr_gaussian_metrics = compute_metrics(reference_signal, mvdr_output_gaussian)
    das_interfering_metrics = compute_metrics(reference_signal, das_output_interfering)
    mvdr_interfering_metrics = compute_metrics(reference_signal, mvdr_output_interfering)

    # Accumulate metrics
    metrics_summary["das_gaussian"]["pesq"].append(das_gaussian_metrics[0])
    metrics_summary["das_gaussian"]["estoi"].append(das_gaussian_metrics[1])
    metrics_summary["das_gaussian"]["si_sdr"].append(das_gaussian_metrics[2])

    metrics_summary["mvdr_gaussian"]["pesq"].append(mvdr_gaussian_metrics[0])
    metrics_summary["mvdr_gaussian"]["estoi"].append(mvdr_gaussian_metrics[1])
    metrics_summary["mvdr_gaussian"]["si_sdr"].append(mvdr_gaussian_metrics[2])

    metrics_summary["das_interfering"]["pesq"].append(das_interfering_metrics[0])
    metrics_summary["das_interfering"]["estoi"].append(das_interfering_metrics[1])
    metrics_summary["das_interfering"]["si_sdr"].append(das_interfering_metrics[2])

    metrics_summary["mvdr_interfering"]["pesq"].append(mvdr_interfering_metrics[0])
    metrics_summary["mvdr_interfering"]["estoi"].append(mvdr_interfering_metrics[1])
    metrics_summary["mvdr_interfering"]["si_sdr"].append(mvdr_interfering_metrics[2])

# Compute average metrics across all instances
average_metrics = {}
for key, values in metrics_summary.items():
    average_metrics[key] = {
        "pesq": np.mean(values["pesq"]),
        "estoi": np.mean(values["estoi"]),
        "si_sdr": np.mean(values["si_sdr"]),
    }

# Print average metrics
print("Average Metrics Across All Instances:")
for method, metrics in average_metrics.items():
    print(f"{method}: PESQ={metrics['pesq']:.2f}, ESTOI={metrics['estoi']:.2f}, SI-SDR={metrics['si_sdr']:.2f}")

# Plot spectrograms for the first instance as an example
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(das_output_gaussian)), ref=np.max), sr=FS,
                         x_axis="time", y_axis="log")
plt.title("Delay-&-Sum Beamformer Output (Gaussian Noise)")

plt.subplot(4, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(mvdr_output_gaussian)), ref=np.max), sr=FS,
                         x_axis="time", y_axis="log")
plt.title("MVDR Beamformer Output (Gaussian Noise)")

plt.subplot(4, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(das_output_interfering)), ref=np.max), sr=FS,
                         x_axis="time", y_axis="log")
plt.title("Delay-&-Sum Beamformer Output (Interfering Noise)")

plt.subplot(4, 1, 4)
librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(mvdr_output_interfering)), ref=np.max), sr=FS,
                         x_axis="time", y_axis="log")
plt.title("MVDR Beamformer Output (Interfering Noise)")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}beamformer_spectrograms.png")
plt.show()
