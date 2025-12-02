#!/usr/bin/env python3

import argparse
import csv
import os
import socket
import fcntl
import struct
import array
import time
from collections import deque
from tabulate import tabulate
import pynvml
from datetime import datetime
import re

# Arguments
parser = argparse.ArgumentParser()
grp = parser.add_mutually_exclusive_group()
grp.add_argument("--ib", action="store_true", help="Monitor InfiniBand HCA fabric (port) counters")
grp.add_argument("--rdma", action="store_true", help="Monitor IPoIB netdev stats (default)")
parser.add_argument("-d", "--dump", action="store_true", help="Enable full history recording (dumpable later)")
args = parser.parse_args()

HISTORY_LEN = 30

# InfiniBand device discovery
def get_ib_devices():
    ib_devices = []
    try:
        devs = sorted(os.listdir('/sys/class/infiniband'))
    except FileNotFoundError:
        return ib_devices
    for dev in devs:
        if 'mlx' in dev:
            net_devs = os.listdir(f'/sys/class/infiniband/{dev}/device/net')
            if net_devs:
                ib_devices.append({"mlx": dev, "port": net_devs[0]})
    return ib_devices

def get_up_ib_devices():
    ib_devices = get_ib_devices()
    up_ib_devices = []
    for device in ib_devices:
        netdevice = device["port"]
        state_file = f'/sys/class/net/{netdevice}/operstate'
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                if f.read().strip() == 'up':
                    up_ib_devices.append(device)
    return up_ib_devices

ibd = get_up_ib_devices()
if not ibd:
    print("[WARN] No InfiniBand devices found.")

# InfiniBand Fabric Port Counters
def get_ib_fabric_counters(mlx_dev, port='1'):
    base = f"/sys/class/infiniband/{mlx_dev}/ports/{port}/counters"
    wanted = [
        "port_xmit_data",
        "port_rcv_data",
        "port_xmit_packets",
        "port_rcv_packets",
        "port_xmit_discards",
        "port_rcv_errors",
        "symbol_error",
        "link_downed",
        "link_error_recovery",
    ]
    counters = {}
    for key in wanted:
        try:
            with open(f"{base}/{key}") as f:
                counters[key] = int(f.read())
        except Exception:
            counters[key] = 0
    return counters

last_ib_fabric_counters = {}
# NEW: For max Gbps observed in IB mode
ib_net_max = {dev["mlx"]: {"rx": 0, "tx": 0} for dev in ibd}

# RDMA per-priority stats (for RDMA/IPoIB mode)
def find_rdma_nics():
    rdma_list = []
    for mlx in sorted(os.listdir("/sys/class/infiniband")):
        if not mlx.startswith("mlx"):
            continue
        ports_path = f"/sys/class/infiniband/{mlx}/ports"
        if not os.path.exists(ports_path):
            continue
        for port in os.listdir(ports_path):
            ndev_file = f"/sys/class/infiniband/{mlx}/ports/{port}/gid_attrs/ndevs/0"
            try:
                with open(ndev_file) as f:
                    nic = f.read().strip()
                    if nic.startswith("rdma"):
                        rdma_list.append((mlx, port, nic))
            except (OSError, FileNotFoundError, IOError):
                continue
    return rdma_list

last_rdma_counters = {}
def get_rdma_delta(mlx, port, nic):
    global last_rdma_counters
    try:
        output = os.popen(f"ethtool -S {nic}").read()
    except Exception:
        return [0]*8, [0]*8

    rx_vals, tx_vals = [], []
    for i in range(8):
        m = re.search(rf"rx_prio{i}_bytes:\s*(\d+)", output)
        rx_vals.append(int(m.group(1)) if m else 0)
        m = re.search(rf"tx_prio{i}_bytes:\s*(\d+)", output)
        tx_vals.append(int(m.group(1)) if m else 0)

    key = f"{mlx}_{port}_{nic}"
    now = time.time()
    prev = last_rdma_counters.get(key)

    if prev is None:
        rx_delta, tx_delta = [0]*8, [0]*8
        dt = 1.0
    else:
        prev_rx, prev_tx, prev_time = prev
        dt = now - prev_time
        if dt <= 0:
            dt = 1.0
        rx_delta = [(rx_vals[i]-prev_rx[i])/dt for i in range(8)]
        tx_delta = [(tx_vals[i]-prev_tx[i])/dt for i in range(8)]

    last_rdma_counters[key] = (rx_vals, tx_vals, now)
    rx_delta_gbps = [v*8/1e9 for v in rx_delta]
    tx_delta_gbps = [v*8/1e9 for v in tx_delta]
    return rx_delta_gbps, tx_delta_gbps

# Initialize GPUs
pynvml.nvmlInit()
deviceCount = pynvml.nvmlDeviceGetCount()

net_history = {dev["mlx"]: {"rx": deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN),
                            "tx": deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)}
               for dev in ibd}
net_max = {dev["mlx"]: {"rx": 0, "tx": 0} for dev in ibd}

gpu_utilization = {i: deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
                   for i in range(deviceCount)}
gpu_mem_util = {i: deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
                for i in range(deviceCount)}
gpu_max = {i: {"gpu": 0, "mem": 0} for i in range(deviceCount)}

archive_rows = [] if args.dump else None

def sparkline(data):
    chars = "·▂▃▄▅▆▇█"
    mn, mx = min(data), max(data)
    if mx - mn == 0:
        idx = int((mn / 100) * (len(chars) - 1)) if mx > 0 else 0
        return "".join([chars[idx]] * len(data))
    return "".join([
        chars[int((x - mn) / (mx - mn) * (len(chars) - 1))]
        for x in data
    ])

def dump_archive_csv(filename_prefix="stats_dump"):
    if not args.dump or not archive_rows:
        print("[WARN] Dump mode not enabled or no data recorded.")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{ts}.csv"
    headers = ["timestamp"]
    for i in range(deviceCount):
        headers += [f"GPU{i}_Util", f"GPU{i}_Mem", f"GPU{i}_Temp", f"GPU{i}_Power"]
    for dev in ibd:
        mlx = dev["mlx"]
        if args.ib:
            headers += [f"{mlx}_TX_Packets", f"{mlx}_RX_Packets", f"{mlx}_TX_Gbps", f"{mlx}_RX_Gbps",
                        f"{mlx}_TX_Max", f"{mlx}_RX_Max",
                        f"{mlx}_TX_Discards", f"{mlx}_RX_Errors", f"{mlx}_Symbol_Errors",
                        f"{mlx}_Link_Downs", f"{mlx}_Link_Recovery"]
        else:
            headers += [f"{mlx}_RX", f"{mlx}_TX", f"{mlx}_RX_Drop", f"{mlx}_TX_Drop"]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(archive_rows)
    print(f"[INFO] Dumped full run history to {filename}")

SIOCETHTOOL = 0x8946
ETHTOOL_GSTRINGS = 0x0000001b
ETHTOOL_GSSET_INFO = 0x00000037
ETHTOOL_GSTATS = 0x0000001d
ETH_SS_STATS = 0x1
ETH_GSTRING_LEN = 32
class Ethtool:
    def __init__(self, ifname):
        self.ifname = ifname
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
    def _send_ioctl(self, data):
        ifr = struct.pack('16sP', self.ifname.encode("utf-8"), data.buffer_info()[0])
        return fcntl.ioctl(self._sock.fileno(), SIOCETHTOOL, ifr)
    def get_gstringset(self, set_id):
        sset_info = array.array('B', struct.pack("IIQI", ETHTOOL_GSSET_INFO, 0, 1 << set_id, 0))
        self._send_ioctl(sset_info)
        sset_mask, sset_len = struct.unpack("8xQI", sset_info)
        if sset_mask == 0:
            sset_len = 0
        strings = array.array("B", struct.pack("III", ETHTOOL_GSTRINGS, ETH_SS_STATS, sset_len))
        strings.extend(b'\x00' * sset_len * ETH_GSTRING_LEN)
        self._send_ioctl(strings)
        for i in range(sset_len):
            offset = 12 + ETH_GSTRING_LEN * i
            s = strings[offset:offset+ETH_GSTRING_LEN].tobytes().partition(b'\x00')[0].decode("utf-8")
            yield s
    def get_nic_stats(self):
        strings = list(self.get_gstringset(ETH_SS_STATS))
        n_stats = len(strings)
        stats = array.array("B", struct.pack("II", ETHTOOL_GSTATS, n_stats))
        stats.extend(struct.pack('Q', 0) * n_stats)
        self._send_ioctl(stats)
        values = []
        for i in range(n_stats):
            offset = 8 + 8 * i
            value = struct.unpack('Q', stats[offset:offset+8])[0]
            values.append(value)
        rx_idx = strings.index("rx_bytes_phy") if "rx_bytes_phy" in strings else 0
        tx_idx = strings.index("tx_bytes_phy") if "tx_bytes_phy" in strings else 1
        return values[rx_idx], values[tx_idx]
last_net_counters = {}
def get_net_sample(mlx, port):
    d = Ethtool(port)
    rx_bytes, tx_bytes = d.get_nic_stats()
    now = time.time()
    key = f"{mlx}_{port}"
    prev = last_net_counters.get(key)
    last_net_counters[key] = (now, rx_bytes, tx_bytes)
    if prev is None:
        return 0, 0
    prev_time, prev_rx, prev_tx = prev
    dt = now - prev_time
    if dt <= 0:
        return 0, 0
    rx_gbps = (rx_bytes - prev_rx) * 8 / (dt * 1e9)
    tx_gbps = (tx_bytes - prev_tx) * 8 / (dt * 1e9)
    return max(rx_gbps, 0), max(tx_gbps, 0)

def get_drop_sample(iface):
    try:
        with open(f"/sys/class/net/{iface}/statistics/rx_dropped") as f:
            rx_dropped = int(f.read())
        with open(f"/sys/class/net/{iface}/statistics/tx_dropped") as f:
            tx_dropped = int(f.read())
        return rx_dropped, tx_dropped
    except Exception:
        return 0, 0

def get_gpu_sample(i):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    gpu_util = util.gpu
    mem_util = mem_info.used / mem_info.total * 100
    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    return gpu_util, mem_util, temp, power_usage, power_limit, graphics_clock, memory_clock

try:
    while True:
        row = [datetime.now().isoformat()] if args.dump else None
        print("\033c", end="")  # clear screen

        net_table = []
        if args.ib:
            # InfiniBand (native fabric) mode, with max throughput tracking
            for dev in ibd:
                mlx = dev["mlx"]
                counters = get_ib_fabric_counters(mlx, port='1')
                now = time.time()
                key = f"{mlx}_1"
                cur_tx_data = counters["port_xmit_data"]
                cur_rx_data = counters["port_rcv_data"]
                prev = last_ib_fabric_counters.get(key)
                if prev is not None:
                    prev_time, prev_tx_data, prev_rx_data = prev
                    dt = now - prev_time
                    tx_gbps = ((cur_tx_data - prev_tx_data) * 4 * 8) / (dt * 1e9)
                    rx_gbps = ((cur_rx_data - prev_rx_data) * 4 * 8) / (dt * 1e9)
                else:
                    tx_gbps, rx_gbps, dt = 0.0, 0.0, 1.0
                last_ib_fabric_counters[key] = (now, cur_tx_data, cur_rx_data)
                ib_net_max[mlx]["tx"] = max(ib_net_max[mlx]["tx"], tx_gbps)
                ib_net_max[mlx]["rx"] = max(ib_net_max[mlx]["rx"], rx_gbps)
                net_table.append([
                    f"{mlx}:1",
                    counters["port_xmit_packets"], counters["port_rcv_packets"],
                    f"{tx_gbps:.2f}", f"{rx_gbps:.2f}",
                    f"{ib_net_max[mlx]['tx']:.2f}", f"{ib_net_max[mlx]['rx']:.2f}",
                    counters["port_xmit_discards"], counters["port_rcv_errors"], counters["symbol_error"],
                    counters["link_downed"], counters["link_error_recovery"],
                ])
                if args.dump:
                    row += [
                        counters["port_xmit_packets"], counters["port_rcv_packets"],
                        tx_gbps, rx_gbps,
                        ib_net_max[mlx]["tx"], ib_net_max[mlx]["rx"],
                        counters["port_xmit_discards"], counters["port_rcv_errors"], counters["symbol_error"],
                        counters["link_downed"], counters["link_error_recovery"]
                    ]
            print("=== InfiniBand Fabric Port Counters ===")
            print(tabulate(
                net_table,
                headers=[
                    "IB Port", "TX Packets", "RX Packets", "TX Gbps", "RX Gbps",
                    "TX Max", "RX Max",
                    "TX Discards", "RX Errors", "Symbol Errors",
                    "Link Downs", "Link Recovery"
                ]
            ))
        else:
            # RDMA/IPoIB mode (no changes)
            cumulative_drops = {}
            for dev in ibd:
                mlx, port = dev["mlx"], dev["port"]
                rx, tx = get_net_sample(mlx, port)
                net_history[mlx]["rx"].append(rx)
                net_history[mlx]["tx"].append(tx)
                net_max[mlx]["rx"] = max(net_max[mlx]["rx"], rx)
                net_max[mlx]["tx"] = max(net_max[mlx]["tx"], tx)
                rx_drp, tx_drp = get_drop_sample(port)
                cumulative_drops[mlx] = {"rx": rx_drp, "tx": tx_drp}
                if args.dump:
                    row += [rx, tx, rx_drp, tx_drp]
                net_table.append([
                    f"{mlx} ({port})",
                    sparkline(net_history[mlx]["rx"]),
                    f"{net_history[mlx]['rx'][-1]:.2f}",
                    f"{net_max[mlx]['rx']:.2f}",
                    sparkline(net_history[mlx]["tx"]),
                    f"{net_history[mlx]['tx'][-1]:.2f}",
                    f"{net_max[mlx]['tx']:.2f}",
                    rx_drp, tx_drp,
                ])
            print("=== NIC Stats (Gbps, Cumulative Drops) ===")
            print(tabulate(
                net_table,
                headers=[
                    "NIC", "RX History", "RX Gbps", "RX Max",
                    "TX History", "TX Gbps", "TX Max",
                    "RX Drop (total)", "TX Drop (total)"
                ]
            ))

            prio_headers = [f"RX P{i}" for i in range(8)] + [f"TX P{i}" for i in range(8)] + ["TOTAL RX", "TOTAL TX"]
            rdma_table = []
            rdma_devices = find_rdma_nics()
            for mlx, port, nic in rdma_devices:
                rx_delta_mb, tx_delta_mb = get_rdma_delta(mlx, port, nic)
                total_rx = sum(rx_delta_mb)
                total_tx = sum(tx_delta_mb)
                rdma_table.append(
                    [f"{mlx}:{port} ({nic})"] +
                    [f"{x:.2f}" for x in rx_delta_mb] +
                    [f"{x:.2f}" for x in tx_delta_mb] +
                    [f"{total_rx:.2f}", f"{total_tx:.2f}"]
                )
            if rdma_table:
                print("\n=== RDMA NIC traffic classes per-Priority Bandwidth (Gbps) ===")
                print(tabulate(rdma_table, headers=["NIC"] + prio_headers))

        gpu_values = []
        gpu_table = []
        for i in range(deviceCount):
            gpu_util, mem_util, temp, power, power_limit, graphics_clock, memory_clock = get_gpu_sample(i)
            gpu_utilization[i].append(gpu_util)
            gpu_mem_util[i].append(mem_util)
            gpu_max[i]["gpu"] = max(gpu_max[i]["gpu"], gpu_util)
            gpu_max[i]["mem"] = max(gpu_max[i]["mem"], mem_util)
            if args.dump:
                gpu_values += [gpu_util, mem_util, temp, power]
            gpu_table.append([
                f"GPU-{i}",
                sparkline(gpu_utilization[i]),
                gpu_util,
                f"{gpu_max[i]['gpu']}",
                sparkline(gpu_mem_util[i]),
                f"{mem_util:.1f}",
                f"{gpu_max[i]['mem']:.1f}",
                f"{temp}C",
                f"{int(power)}W / {int(power_limit)}W",
                f"{graphics_clock} MHz",
                f"{memory_clock} MHz"
            ])
        print("\n=== GPU Stats ===")
        print(tabulate(gpu_table, headers=[
            "GPU","Util History","Util %","Util Max","Mem History",
            "Mem %","Mem Max","Temp","Power","Graphics Freq","Memory Freq"
        ]))

        if args.dump:
            row = [row[0]] + gpu_values + row[1:]
            archive_rows.append(row)

        time.sleep(1)

except KeyboardInterrupt:
    print("\nExiting...")
    if args.dump:
        dump_archive_csv("stats_dump")
    pynvml.nvmlShutdown()
