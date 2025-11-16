#!/usr/bin/env python3
import time

def read_netdev():
    stats = {}
    with open("/proc/net/dev") as f:
        for line in f.readlines()[2:]:        # skip headers
            iface, data = line.split(":", 1)
            fields = data.split()
            rx_bytes, tx_bytes = int(fields[0]), int(fields[8])
            stats[iface.strip()] = (rx_bytes, tx_bytes)
    return stats

def human_rate(bytes_per_sec, unit="MB/s"):
    if unit == "MB/s":
        return bytes_per_sec / 1024 / 1024
    elif unit == "Mb/s":
        return bytes_per_sec * 8 / 1024 / 1024
    else:
        return bytes_per_sec

def monitor(interval=1.0, unit="MB/s"):
    prev = read_netdev()
    time.sleep(interval)
    while True:
        curr = read_netdev()
        print(time.strftime("%H:%M:%S"))
        for iface in curr:
            if iface not in prev:
                continue
            rx = (curr[iface][0] - prev[iface][0]) / interval
            tx = (curr[iface][1] - prev[iface][1]) / interval
            print(f"  {iface:<8} RX: {human_rate(rx, unit):8.2f} {unit}   TX: {human_rate(tx, unit):8.2f} {unit}")
        print("-" * 60)
        prev = curr
        time.sleep(interval)

if __name__ == "__main__":
    monitor(interval=1, unit="MB/s")   # change to "Mb/s" if you prefer megabits