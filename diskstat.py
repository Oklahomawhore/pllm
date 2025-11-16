#!/usr/bin/env python3
import time

def read_diskstats():
    stats = {}
    with open("/proc/diskstats") as f:
        for line in f:
            fields = line.split()
            # columns: major minor name reads ... sectors_read writes ... sectors_written
            if len(fields) < 14:
                continue
            name = fields[2]
            # skip loop, ram, or partitions if you want only physical disks
            if name.startswith(("loop", "ram")) or name[-1].isdigit():
                continue
            sectors_read = int(fields[5])
            sectors_written = int(fields[9])
            stats[name] = (sectors_read, sectors_written)
    return stats

def human_rate(sectors_per_s, sector_size=512, unit="MB/s"):
    bytes_per_s = sectors_per_s * sector_size
    if unit == "MB/s":
        return bytes_per_s / 1024 / 1024
    elif unit == "MB/s":
        return bytes_per_s / 1000 / 1000
    else:
        return bytes_per_s

def monitor(interval=1.0, unit="MB/s"):
    prev = read_diskstats()
    time.sleep(interval)
    while True:
        curr = read_diskstats()
        print(time.strftime("%H:%M:%S"))
        for dev in curr:
            if dev not in prev:
                continue
            rps = (curr[dev][0] - prev[dev][0]) / interval
            wps = (curr[dev][1] - prev[dev][1]) / interval
            r_mb = human_rate(rps, unit=unit)
            w_mb = human_rate(wps, unit=unit)
            print(f"  {dev:<6}  READ: {r_mb:7.2f} {unit}   WRITE: {w_mb:7.2f} {unit}")
        print("-" * 60)
        prev = curr
        time.sleep(interval)

if __name__ == "__main__":
    monitor(interval=1, unit="MB/s")