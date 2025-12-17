#!/usr/bin/env python3

import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime
import re
import subprocess

from rich.console import Console, Group
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from tabulate import tabulate

parser = argparse.ArgumentParser()
gpu_grp = parser.add_mutually_exclusive_group()
gpu_grp.add_argument("--amd", action="store_true", help="Show AMD GPU stats (via ROCm/pyrsmi)")
gpu_grp.add_argument("--nvidia", action="store_true", help="Show NVIDIA GPU stats via pynvml")
net_grp = parser.add_mutually_exclusive_group()
net_grp.add_argument("--ib", action="store_true", help="Monitor InfiniBand HCA fabric (port) counters")
net_grp.add_argument("--rdma", action="store_true", help="Monitor IPoIB netdev stats (default)")
parser.add_argument("-d", "--dump", action="store_true", help="Enable full history recording (dumpable later)")
args = parser.parse_args()

HISTORY_LEN = 30
archive_rows = [] if args.dump else None

# ===================== Network Section =======================
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
ib_net_max = {dev["mlx"]: {"rx": 0, "tx": 0} for dev in ibd}
net_history = {dev["mlx"]: {"rx": deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN),
                            "tx": deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)}
               for dev in ibd}
net_max = {dev["mlx"]: {"rx": 0, "tx": 0} for dev in ibd}
last_net_bytes = {}

def get_net_sample(mlx, port):
    rx_bytes = tx_bytes = None
    # Try to get *_bytes_phy with ethtool -S
    try:
        ethtool_output = subprocess.run(['ethtool', '-S', port], capture_output=True, text=True, check=True)
        # Use regular expressions to find rx_bytes_phy and tx_bytes_phy
        rx_match = re.search(r'rx_bytes_phy:\s*(\d+)', ethtool_output.stdout)
        tx_match = re.search(r'tx_bytes_phy:\s*(\d+)', ethtool_output.stdout)
        if rx_match and tx_match:
            rx_bytes = int(rx_match.group(1))
            tx_bytes = int(tx_match.group(1))
    except Exception:
        pass

    # Fallback to legacy statistics if *_bytes_phy not found/fails
    if rx_bytes is None or tx_bytes is None:
        try:
            with open(f"/sys/class/net/{port}/statistics/rx_bytes") as f:
                rx_bytes = int(f.read())
            with open(f"/sys/class/net/{port}/statistics/tx_bytes") as f:
                tx_bytes = int(f.read())
        except Exception:
            return 0, 0

    now = time.time()
    key = f"{mlx}_{port}"
    prev = last_net_bytes.get(key)
    last_net_bytes[key] = (now, rx_bytes, tx_bytes)
    if prev is None:
        return 0, 0
    prev_time, prev_rx, prev_tx = prev
    dt = now - prev_time
    if dt <= 0.0:
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

# === AMD CLOCKS ===
def get_rocm_smi_clocks_all(gpu_count):
    output = subprocess.check_output(['rocm-smi', '--showclocks'], encoding='utf-8', errors='ignore')
    clocks = []
    for i in range(gpu_count):
        d = {"fclk": None, "fclk_max": None, "mclk": None, "mclk_max": None,
             "sclk": None, "sclk_max": None, "socclk": None, "socclk_max": None}
        for clock in ["fclk", "mclk", "sclk", "socclk"]:
            current, mx = None, None
            for line in output.splitlines():
                if f"GPU[{i}]" in line and f"{clock} clock level" in line:
                    m = re.findall(r"\(([\d\.]+)Mhz\)", line)
                    curr = re.search(fr"{clock} clock level:.*S: \(([\d\.]+)Mhz\)", line)
                    current = curr.group(1) if curr else (m[-1] if m else None)
                    mx = f"{int(max([float(x) for x in m]))}" if m else None
                    break
            d[f"{clock}"] = current
            d[f"{clock}_max"] = mx
        clocks.append(d)
    return clocks

# === AMD GPU Section ===
amd_gpu_max = {}
def get_amd_gpu_stats_rocml():
    from pyrsmi import rocml
    rocml.smi_initialize()
    count = rocml.smi_get_device_count()
    clock_list = get_rocm_smi_clocks_all(count)
    stats, amd_values = [], []
    for i in range(count):
        name   = rocml.smi_get_device_name(i)
        util   = rocml.smi_get_device_utilization(i)
        mem_total = rocml.smi_get_device_memory_total(i)
        try: mem_used = rocml.smi_get_device_memory_used(i)
        except Exception: mem_used = 0
        try: power = rocml.smi_get_device_average_power(i)
        except Exception: power = 0
        try: temp = rocml.smi_get_device_temperature(i)
        except Exception: temp = "N/A"

        mem_gb = mem_total / (1024**3) if mem_total else 0
        mem_used_gb = mem_used / (1024**3) if mem_total else 0

        if i not in amd_gpu_max:
            amd_gpu_max[i] = {"util": util, "mem_gb": mem_used_gb}
        else:
            amd_gpu_max[i]["util"] = max(amd_gpu_max[i]["util"], util)
            amd_gpu_max[i]["mem_gb"] = max(amd_gpu_max[i]["mem_gb"], mem_used_gb)

        power_str = f"{int(power)} W"
        clocks = clock_list[i]
        fclk_str   = f"{clocks['fclk']}/{clocks['fclk_max']} MHz"
        mclk_str   = f"{clocks['mclk']}/{clocks['mclk_max']} MHz"
        sclk_str   = f"{clocks['sclk']}/{clocks['sclk_max']} MHz"
        socclk_str = f"{clocks['socclk']}/{clocks['socclk_max']} MHz"

        stats.append([
            f"GPU-{i}",
            name,
            util,
            amd_gpu_max[i]["util"],
            f"{mem_used_gb:.1f}/{mem_gb:.1f}",
            f"{amd_gpu_max[i]['mem_gb']:.1f}/{mem_gb:.1f}",
            power_str,
            temp,
            fclk_str,
            mclk_str,
            sclk_str,
            socclk_str
        ])
        amd_values += [util, amd_gpu_max[i]["util"], mem_used_gb, amd_gpu_max[i]["mem_gb"], power, temp]
    rocml.smi_shutdown()
    return stats, amd_values, count

# ==== NVIDIA Section ====
deviceCount = 0
gpu_utilization = {}
gpu_mem_util = {}
gpu_max = {}
nvidia_gpu_max = {}
if args.nvidia:
    import pynvml
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    gpu_utilization = {i: deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
                       for i in range(deviceCount)}
    gpu_mem_util = {i: deque([0]*HISTORY_LEN, maxlen=HISTORY_LEN)
                    for i in range(deviceCount)}
    gpu_max = {i: {"gpu": 0, "mem": 0, "mem_gb": 0} for i in range(deviceCount)}

def dump_archive_csv(filename_prefix="stats_dump", is_amd=False, amd_count=0):
    if not args.dump or not archive_rows:
        print("[WARN] Dump mode not enabled or no data recorded.")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{ts}.csv"
    headers = ["timestamp"]
    if is_amd:
        for i in range(amd_count):
            headers += [f"AMD_GPU{i}_Util", f"AMD_GPU{i}_UtilMax", f"AMD_GPU{i}_Mem", f"AMD_GPU{i}_MemMax", f"AMD_GPU{i}_Power", f"AMD_GPU{i}_Temp"]
    else:
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

# ==== rich Table Generators ====
def make_net_table():
    table = Table(title="NIC Stats (Gbps, Drops)", show_lines=False)
    table.add_column("NIC")
    for col in ["RX Gbps", "RX Max", "TX Gbps", "TX Max", "RX Drop", "TX Drop"]:
        table.add_column(col, justify="right")
    for dev in ibd:
        mlx, port = dev["mlx"], dev["port"]
        rx, tx = get_net_sample(mlx, port)
        net_history[mlx]["rx"].append(rx)
        net_history[mlx]["tx"].append(tx)
        net_max[mlx]["rx"] = max(net_max[mlx]["rx"], rx)
        net_max[mlx]["tx"] = max(net_max[mlx]["tx"], tx)
        rx_drp, tx_drp = get_drop_sample(port)
        table.add_row(
            f"{mlx} ({port})",
            f"{net_history[mlx]['rx'][-1]:.2f}",
            f"{net_max[mlx]['rx']:.2f}",
            f"{net_history[mlx]['tx'][-1]:.2f}",
            f"{net_max[mlx]['tx']:.2f}",
            str(rx_drp),
            str(tx_drp)
        )
    return table

def make_rdma_table():
    prio_headers = [f"RX P{i}" for i in range(8)] + [f"TX P{i}" for i in range(8)] + ["TOTAL RX", "TOTAL TX"]
    table = Table(title="RDMA NIC priority Bandwidth (Gbps)", show_lines=False)
    table.add_column("NIC")
    for col in prio_headers:
        table.add_column(col, justify="right")
    rdma_devices = find_rdma_nics()
    for mlx, port, nic in rdma_devices:
        rx_delta_mb, tx_delta_mb = get_rdma_delta(mlx, port, nic)
        total_rx = sum(rx_delta_mb)
        total_tx = sum(tx_delta_mb)
        row = [f"{mlx}:{port} ({nic})"] + [f"{x:.2f}" for x in rx_delta_mb] + [f"{x:.2f}" for x in tx_delta_mb] + [f"{total_rx:.2f}", f"{total_tx:.2f}"]
        table.add_row(*row)
    return table

def make_ib_table():
    table = Table(title="InfiniBand Fabric Port Counters", show_lines=False)
    for col in [
        "IB Port", "TX Packets", "RX Packets", "TX Gbps", "RX Gbps",
        "TX Max", "RX Max", "TX Discards", "RX Errors", "Symbol Errors",
        "Link Downs", "Link Recovery"
    ]:
        table.add_column(col)
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
        table.add_row(
            f"{mlx}:1",
            str(counters["port_xmit_packets"]), str(counters["port_rcv_packets"]),
            f"{tx_gbps:.2f}", f"{rx_gbps:.2f}",
            f"{ib_net_max[mlx]['tx']:.2f}", f"{ib_net_max[mlx]['rx']:.2f}",
            str(counters["port_xmit_discards"]), str(counters["port_rcv_errors"]), str(counters["symbol_error"]),
            str(counters["link_downed"]), str(counters["link_error_recovery"])
        )
    return table

def make_amd_gpu_table(stats):
    table = Table(title="AMD GPU Stats", show_lines=False)
    hdr = [
        "GPU", "Name", "Util (%)", "Util Max",
        "Mem Usage (GB)", "Mem Max (GB)", "Power (W)", "Temp",
        "FCLK (MHz)", "MCLK (MHz)", "SCLK (MHz)", "SOCCLK (MHz)"
    ]
    for col in hdr:
        table.add_column(col, justify="right")
    for row in stats:
        table.add_row(*[str(x) for x in row])
    return table

def make_nvidia_gpu_table(stats):
    table = Table(title="NVIDIA GPU Stats", show_lines=False)
    hdr = [
        "GPU", "Util %", "Util Max", "Mem Usage (GB)", "Mem Max (GB)",
        "Mem %", "Mem Max %", "Temp", "Power", "Graphics Freq", "Memory Freq"
    ]
    for col in hdr:
        table.add_column(col, justify="right")
    for row in stats:
        table.add_row(*[str(x) for x in row])
    return table

# ==== MAIN LOOP using `rich.Live` ====
try:
    with Live(refresh_per_second=1, screen=True) as live:
        while True:
            panels = []

            # Network Tables
            if args.ib:
                panels.append(Panel(make_ib_table(), title="InfiniBand"))
            else:
                panels.append(Panel(make_net_table(), title="NIC"))
                panels.append(Panel(make_rdma_table(), title="RDMA Priority"))

            # GPU Tables
            if args.amd:
                stats, amd_gpu_values, amd_count = get_amd_gpu_stats_rocml()
                panels.append(Panel(make_amd_gpu_table(stats)))
                row = [datetime.now().isoformat()] + amd_gpu_values if args.dump else None
            elif args.nvidia:
                gpu_stats = []
                gpu_values = []
                for i in range(deviceCount):
                    import pynvml
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_util = util.gpu
                    mem_util = mem_info.used / mem_info.total * 100 if mem_info.total else 0
                    used_gb = mem_info.used / (1024 ** 3)
                    total_gb = mem_info.total / (1024 ** 3)
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    max_graphics_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    max_memory_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    gpu_utilization[i].append(gpu_util)
                    gpu_mem_util[i].append(mem_util)
                    if i not in nvidia_gpu_max:
                        nvidia_gpu_max[i] = {"util": gpu_util, "mem": mem_util, "mem_gb": used_gb}
                    else:
                        nvidia_gpu_max[i]["util"] = max(nvidia_gpu_max[i]["util"], gpu_util)
                        nvidia_gpu_max[i]["mem"] = max(nvidia_gpu_max[i]["mem"], mem_util)
                        nvidia_gpu_max[i]["mem_gb"] = max(nvidia_gpu_max[i]["mem_gb"], used_gb)
                    if args.dump:
                        gpu_values += [gpu_util, mem_util, temp, power_usage]
                    gpu_stats.append([
                        f"GPU-{i}",
                        gpu_util,
                        int(nvidia_gpu_max[i]["util"]),
                        f"{used_gb:.1f}/{total_gb:.1f}",
                        f"{nvidia_gpu_max[i]['mem_gb']:.1f}/{total_gb:.1f}",
                        f"{mem_util:.1f}",
                        f"{nvidia_gpu_max[i]['mem']:.1f}",
                        f"{temp}C",
                        f"{int(power_usage)}/{int(power_limit)} W",
                        f"{graphics_clock}/{max_graphics_clock} MHz",
                        f"{memory_clock}/{max_memory_clock} MHz"
                    ])
                panels.append(Panel(make_nvidia_gpu_table(gpu_stats)))
                row = [datetime.now().isoformat()] + gpu_values if args.dump else None
            else:
                row = None

            live.update(Group(*panels))

            if args.dump and row:
                archive_rows.append(row)

            time.sleep(1)

except KeyboardInterrupt:
    print("\nExiting...")
    if args.dump:
        dump_archive_csv("stats_dump", is_amd=args.amd, amd_count=(amd_count if args.amd else 0))
    if args.nvidia:
        import pynvml
        pynvml.nvmlShutdown()
