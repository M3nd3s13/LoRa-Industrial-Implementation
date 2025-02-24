import numpy as np
import matplotlib.pyplot as plt

# Configuration
n = 80  # Number of packets per transmitter
SF = [7, 8, 9, 10, 11, 12]  # Spreading Factors
channels = list(range(8))  # 8 channels (0 to 7)
connecting_interval = 300  # Uplink time for each sensor

# Transmission intervals and Time-on-Air for each SF
transmission_intervals = [300, 500, 600]
toa_per_sf = {7: 0.67, 8: 0.1132, 9: 0.2058, 10: 0.4116, 11: 0.8233, 12: 1.6466}

# Demodulator (buffer) limits per SF
max_buffer_limits = {7: 16, 8: 16, 9: 16, 10: 16, 11: 8, 12: 8}

def check_collisions_vectorized(transmission_times, channel_vector, sf_vector):
    """
    Vectorized collision check using np.searchsorted to avoid an O(N^2) loop.
    
    It calculates two types of collisions:
    1. Buffer (demodulator) limit violations: For each transmission, count the number 
       of overlapping transmissions (within its ToA window) and if that exceeds the buffer limit.
    2. Same SF & same channel overlaps: For each unique (SF, channel) group, count the overlapping
       transmissions using searchsorted.
       
    Both collision counts are summed (each collision pair counted twice) and then divided by 2.
    """
    # Flatten the arrays (each transmitter has n packets)
    start_times = transmission_times.flatten()
    channels_arr = channel_vector.flatten()
    sfs = sf_vector.flatten()

    order = np.argsort(start_times)
    start_times_sorted = start_times[order]
    channels_sorted = channels_arr[order]
    sfs_sorted = sfs[order]

    # Calculate ToA for each transmission and its ending time
    toa_array = np.array([toa_per_sf[sf] for sf in sfs_sorted])
    end_times = start_times_sorted + toa_array

   #Demodulator limit
    j_indices = np.searchsorted(start_times_sorted, end_times, side='left')
    buffer_counts = j_indices - np.arange(len(start_times_sorted))
    max_buffers = np.array([max_buffer_limits[sf] for sf in sfs_sorted])
    # Count any excess transmissions beyond the buffer limit.
    excess_buffer = np.maximum(buffer_counts - max_buffers, 0)
    collisions_buffer = np.sum(excess_buffer)

    # Same SF & Same Channel Collision Check
    collisions_same = 0
    unique_sfs = np.unique(sfs_sorted)
    for sf_val in unique_sfs:
        unique_channels = np.unique(channels_sorted[sfs_sorted == sf_val])
        for ch_val in unique_channels:
            # Find transmissions that share the same SF and channel
            mask = (sfs_sorted == sf_val) & (channels_sorted == ch_val)
            group_times = start_times_sorted[mask]
            if group_times.size == 0:
                continue
            group_toa = toa_per_sf[sf_val]
            group_end_times = group_times + group_toa
            # Use searchsorted to count how many transmissions in the group fall within each packet's window.
            k_indices = np.searchsorted(group_times, group_end_times, side='left')
            group_counts = k_indices - np.arange(len(group_times))
            # For each packet, overlapping packets are (group_counts - 1)
            collisions_same += np.sum(group_counts - 1)

    total_collisions = (collisions_buffer + collisions_same) // 2
    return int(total_collisions)

def simulate_packet_loss_rate(start_devices, end_devices, step_devices):
    num_transmitters = np.arange(start_devices, end_devices + 1, step_devices)
    packet_loss_rates = []

    for N in num_transmitters:
        
        channel_vector = np.random.choice(channels, size=(N, n))
        # Adjust probabilities for SF selection
        sf_vector = np.random.choice(SF, size=(N, n), p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05])
        transmission_interval_vector = np.random.choice(transmission_intervals, size=N)
        initial_start_times = np.random.uniform(0, connecting_interval, size=N)
        transmission_times = np.array([
            initial_start_times[i] + transmission_interval_vector[i] * np.arange(n)
            for i in range(N)
        ])
        
        collisions = check_collisions_vectorized(transmission_times, channel_vector, sf_vector)
        total_packets = N * n
        packet_loss_rate = (collisions / total_packets) * 100
        packet_loss_rates.append(packet_loss_rate)

    return num_transmitters, packet_loss_rates

# User inputs for device count control
start_devices = 1    
end_devices = 3000    
step_devices = 100   

# Run simulation
num_transmitters, packet_loss_rates = simulate_packet_loss_rate(start_devices, end_devices, step_devices)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(num_transmitters, packet_loss_rates, marker='o', linestyle='-')
plt.xlabel("Number of Transmitters")
plt.ylabel("Packet Loss Rate (%)")
plt.title("Packet Loss Evolution with Increasing Transmitters\n(Vectorized Collision Check)")
plt.grid(True)
plt.show()
