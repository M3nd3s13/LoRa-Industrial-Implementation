import numpy as np
import matplotlib.pyplot as plt

# Configuration
n = 80  # Number of packets per transmitter
SF = [7, 10, 12]  # Spreading Factors 
channels = list(range(8))  # 8 channels
connecting_interval = 300  # Uplink time for each sensor

# Transmission intervals and Time-on-Air per SF
transmission_intervals = [7, 500, 600]
toa_per_sf = {7: 0.67, 10: 0.4116, 12: 1.6466}

# Demodulator (buffer) limits per SF
max_buffer_limits = {7: 16, 10: 16, 12: 8}

def check_collisions_vectorized(transmission_times, channel_vector, sf_vector):
  
    # Flatten the arrays
    start_times = transmission_times.flatten()
    channels_arr = channel_vector.flatten()
    sfs = sf_vector.flatten()

    order = np.argsort(start_times)
    start_times = start_times[order]
    channels_arr = channels_arr[order]
    sfs = sfs[order]

    # Compute end times for each transmission (the window in which collisions occur)
    toa_array = np.array([toa_per_sf[sf] for sf in sfs])
    end_times = start_times + toa_array

    # Demodulator saturation check
    j_indices = np.searchsorted(start_times, end_times, side='left')
    buffer_counts = j_indices - np.arange(len(start_times))
    max_buffers = np.array([max_buffer_limits[sf] for sf in sfs])
    # Excess beyond the buffer limit counts as collisions.
    excess_buffer = np.maximum(buffer_counts - max_buffers, 0)
    collisions_buffer = np.sum(excess_buffer)

    # --- Same SF & Same Channel Collision Check ---
    collisions_same = 0
    for sf_val in np.unique(sfs):
        for channel_val in np.unique(channels_arr):
            mask = (sfs == sf_val) & (channels_arr == channel_val)
            group_times = start_times[mask]
            if group_times.size == 0:
                continue
           
            group_toa = toa_per_sf[sf_val]
            group_end_times = group_times + group_toa
            # Use searchsorted within the group to count overlapping transmissions.
            k_indices = np.searchsorted(group_times, group_end_times, side='left')
            group_counts = k_indices - np.arange(len(group_times))
            collisions_same += np.sum(group_counts - 1)

    # Both collision counts are summed per packet (thus each pair is counted twice), so divide by 2.
    total_collisions = (collisions_buffer + collisions_same) // 2
    return int(total_collisions)

def simulate_packet_loss_rate(start_devices, end_devices, step_devices):
    num_transmitters = np.arange(start_devices, end_devices + 1, step_devices)
    packet_loss_rates = []

    for N in num_transmitters:
        if N == 0:
            packet_loss_rates.append(0)
            continue

        # Generate random channel assignments and SFs per packet.
        channel_vector = np.random.choice(channels, size=(N, n))
        sf_vector = np.random.choice(SF, size=(N, n), p=[0.6, 0.3, 0.1])
        transmission_interval_vector = np.random.choice(transmission_intervals, size=N)
        initial_start_times = np.random.uniform(0, connecting_interval, size=N)
        transmission_times = np.array([
            initial_start_times[i] + transmission_interval_vector[i] * np.arange(n)
            for i in range(N)
        ])

        collisions = check_collisions_vectorized(transmission_times, channel_vector, sf_vector)
        total_packets = N * n
        packet_loss_rate = (collisions / total_packets) * 100 if total_packets > 0 else 0
        packet_loss_rates.append(packet_loss_rate)

    return num_transmitters, packet_loss_rates

# User inputs for device count control
start_devices = 0
end_devices = 3000
step_devices = 50

# Run simulation
num_transmitters, packet_loss_rates = simulate_packet_loss_rate(start_devices, end_devices, step_devices)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(num_transmitters, packet_loss_rates, marker='o', linestyle='-')
plt.xlabel("Number of Transmitters")
plt.ylabel("Packet Loss Rate (%)")
plt.title("Packet Loss Evolution with Increasing Transmitters\n(Vectorized Collision Check)")
plt.grid()
plt.show()

