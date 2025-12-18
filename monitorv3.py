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
amd_count = 0  # ensure defined for CSV dump at shutdown

# ===================== Network Section =======================
def get_ib_devices():
    ib_devices = []
    try:
        devs = sorted(os.listdir('/sys/class/infiniband'))
    except FileNotFoundError:
        return ib_devices
    for dev in devs:
        if 'mlx' in dev:
            try:
                net_devs = os.listdir(f'/sys/class/infiniband/{dev}/device/net')
            except Exception:
                net_devs = []
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
    base = "/sys/class/infiniband"
    if not os.path.exists(base):
        return rdma_list
    for mlx in sorted(os.listdir(base)):
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

# Stable list for headers and row ordering
rdma_devices_static = find_rdma_nics()

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
def update_max(gpu_idx, util, mem_used_gb, power):
    global amd_gpu_max
    if gpu_idx not in amd_gpu_max:
        amd_gpu_max[gpu_idx] = {'util': util, 'mem_used_gb': mem_used_gb, 'power': power}
    else:
        amd_gpu_max[gpu_idx]['util'] = max(amd_gpu_max[gpu_idx]['util'], util)
        amd_gpu_max[gpu_idx]['mem_used_gb'] = max(amd_gpu_max[gpu_idx]['mem_used_gb'], mem_used_gb)
        amd_gpu_max[gpu_idx]['power'] = max(amd_gpu_max[gpu_idx]['power'], power)

def parse_amd_smi_monitor(output):
    """
    Parse `amd-smi monitor` one-shot output.
    Returns list of dicts per GPU with keys:
    idx, power_w, temp_c, gfx_util, vram_used_gb, vram_total_gb
    """
    gpus = []
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    if not lines:
        return gpus
    # Skip header; parse subsequent lines
    for ln in lines[1:]:
        m = re.search(
            r'^(\d+)\s+'          # GPU index
            r'(\d+)\s*W\s+'       # POWER W
            r'(\d+)\s*°C\s+'      # GPU_T
            r'(\d+)\s*°C\s+'      # MEM_T (ignored)
            r'(\d+)\s*MHz\s+'     # GFX_CLK (ignored)
            r'(\d+)\s*%\s+'       # GFX% (util)
            r'(\d+|N/A)\s*%\s+'   # MEM% (ignored)
            r'(?:\S+)\s+'         # ENC% (may be N/A)
            r'(\d+|N/A)\s*%\s+'   # DEC% (ignored)
            r'([\d.]+)\/([\d.]+)\s*GB$',  # VRAM_USAGE used/total
            ln
        )
        if not m:
            # fallback split-based parse
            parts = ln.split()
            try:
                idx = int(parts[0])
                power_w = int(parts[1])
                temp_c = int(parts[3])
                gfx_util = int(parts[7])
                vram_pair = parts[-2]  # e.g., "0.3/192.0"
                used_s, total_s = vram_pair.split("/")
                vram_used_gb = float(used_s)
                vram_total_gb = float(total_s)
                gpus.append({
                    "idx": idx, "power_w": power_w, "temp_c": temp_c,
                    "gfx_util": gfx_util, "vram_used_gb": vram_used_gb, "vram_total_gb": vram_total_gb
                })
            except Exception:
                continue
        else:
            idx = int(m.group(1))
            power_w = int(m.group(2))
            temp_c = int(m.group(3))
            gfx_util = int(m.group(6))
            vram_used_gb = float(m.group(9))
            vram_total_gb = float(m.group(10))
            gpus.append({
                "idx": idx, "power_w": power_w, "temp_c": temp_c,
                "gfx_util": gfx_util, "vram_used_gb": vram_used_gb, "vram_total_gb": vram_total_gb
            })
    return gpus

def get_amd_gpu_stats_rocml():
    stats = []
    amd_gpu_values = []
    amd_count_local = 0

    # --- First try pyrsmi ---
    try:
        from pyrsmi import rocml
        rocml.smi_initialize()
        count = rocml.smi_get_device_count()
        if count > 0:
            for i in range(count):
                try:
                    name = rocml.smi_get_device_name(i)
                    util = rocml.smi_get_device_utilization(i)
                    mem_total = rocml.smi_get_device_memory_total(i)
                    mem_used = rocml.smi_get_device_memory_used(i)
                    power = rocml.smi_get_device_average_power(i)
                    temp = rocml.smi_get_device_temperature(i)
                    mem_gb = mem_total / (1024**3) if mem_total else 0
                    mem_used_gb = mem_used / (1024**3) if mem_total else 0
                    update_max(i, util, mem_used_gb, power)
                    row = {
                        "GPU": i,
                        "Name": name,
                        "Util (%)": util,
                        "Util Max": amd_gpu_max[i]["util"],
                        "Mem Usage (GB)": f"{mem_used_gb:.1f}/{mem_gb:.1f}",
                        "Mem Max (GB)": f"{amd_gpu_max[i]['mem_used_gb']:.1f}/{mem_gb:.1f}",
                        "Power (W)": power,
                        "Power Max (W)": amd_gpu_max[i]["power"],
                        "Temp": temp,
                    }
                    row = {k: v for k, v in row.items() if v not in [None, "N/A"]}
                    stats.append(row)
                    amd_gpu_values.extend([
                        util,
                        amd_gpu_max[i]["util"],
                        mem_used_gb,
                        amd_gpu_max[i]['mem_used_gb'],
                        power,
                        temp,
                    ])
                except Exception:
                    continue
            amd_count_local = len(stats)
            rocml.smi_shutdown()
            return stats, amd_gpu_values, amd_count_local
    except Exception:
        pass  # Continue to CLI fallback

    # --- Fallback on amd-smi CLI (table mode) ---
    try:
        out = subprocess.check_output(['amd-smi'], encoding='utf-8', errors='ignore', timeout=3)
        cli_pattern = re.compile(
            r"\|\s+([0-9a-fA-F:.]+)\s+(.*?)\s+\|\s+(\d+) %\s+(\d+) °C.*?(\d+)/(\d+)\s+W.*?\n\|\s+\d+\s+\d+\s+\d+\s+\S+\s+\|\s+(\d+) %\s+([N/A\d\.\-%]+)\s+(\d+)/(\d+)\s+MB",
            re.DOTALL)
        for m in cli_pattern.finditer(out):
            try:
                gpu_idx = len(stats)
                raw_name = m.group(2)
                name = " ".join(raw_name.split())
                if "GPU-Name" in name:
                    continue
                temp = int(m.group(4))
                power_used = int(m.group(5))
                power_max = int(m.group(6))
                gfx_util = int(m.group(7))
                mem_used = float(m.group(9))
                mem_total = float(m.group(10))
                mem_used_gb = mem_used / 1024
                mem_total_gb = mem_total / 1024
                update_max(gpu_idx, gfx_util, mem_used_gb, power_used)
                row = {
                    "GPU": gpu_idx,
                    "Name": name,
                    "Util (%)": gfx_util,
                    "Util Max": amd_gpu_max[gpu_idx]["util"],
                    "Mem Usage (GB)": f"{mem_used_gb:.1f}/{mem_total_gb:.1f}",
                    "Mem Max (GB)": f"{amd_gpu_max[gpu_idx]['mem_used_gb']:.1f}/{mem_total_gb:.1f}",
                    "Power (W)": power_used,
                    "Power Max (W)": amd_gpu_max[gpu_idx]["power"],
                    "Temp": temp,
                }
                row = {k: v for k, v in row.items() if v not in [None, "N/A"]}
                stats.append(row)
                amd_gpu_values.extend([
                    gfx_util,
                    amd_gpu_max[gpu_idx]["util"],
                    mem_used_gb,
                    amd_gpu_max[gpu_idx]['mem_used_gb'],
                    power_used,
                    temp,
                ])
            except Exception:
                continue
        amd_count_local = len(stats)
        if amd_count_local > 0:
            return stats, amd_gpu_values, amd_count_local
    except Exception:
        pass

    # --- Second fallback: amd-smi monitor (works on older <7.x in many envs) ---
    try:
        # Run with a short timeout in case 'monitor' streams continuously
        res = subprocess.run(['amd-smi', 'monitor'], capture_output=True, text=True, timeout=2)
        out = res.stdout
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if hasattr(e, 'stdout') else ""
    except Exception:
        out = ""

    try:
        gpu_rows = parse_amd_smi_monitor(out)
        for g in gpu_rows:
            i = g["idx"]
            name = f"GPU-{i}"
            gfx_util = g["gfx_util"]
            power_used = g["power_w"]
            temp = g["temp_c"]
            mem_used_gb = g["vram_used_gb"]
            mem_total_gb = g["vram_total_gb"]
            update_max(i, gfx_util, mem_used_gb, power_used)
            row = {
                "GPU": i,
                "Name": name,
                "Util (%)": gfx_util,
                "Util Max": amd_gpu_max[i]["util"],
                "Mem Usage (GB)": f"{mem_used_gb:.1f}/{mem_total_gb:.1f}",
                "Mem Max (GB)": f"{amd_gpu_max[i]['mem_used_gb']:.1f}/{mem_total_gb:.1f}",
                "Power (W)": power_used,
                "Power Max (W)": amd_gpu_max[i]["power"],
                "Temp": temp,
            }
            stats.append(row)
            amd_gpu_values.extend([
                gfx_util,
                amd_gpu_max[i]["util"],
                mem_used_gb,
                amd_gpu_max[i]['mem_used_gb'],
                power_used,
                temp,
            ])
        amd_count_local = len(gpu_rows)
        if amd_count_local > 0:
            return stats, amd_gpu_values, amd_count_local
    except Exception:
        pass

    # --- If all fails ---
    return [], [], 0

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

# ==== CSV DUMP SUPPORT ====
def dump_archive_csv(filename_prefix="stats_dump", is_amd=False, amd_count=0):
    if not args.dump or not archive_rows:
        print("[WARN] Dump mode not enabled or no data recorded.")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{ts}.csv"
    headers = ["timestamp"]
    # GPU headers
    if is_amd:
        for i in range(amd_count):
            headers += [f"AMD_GPU{i}_Util", f"AMD_GPU{i}_UtilMax",
                        f"AMD_GPU{i}_Mem", f"AMD_GPU{i}_MemMax",
                        f"AMD_GPU{i}_Power", f"AMD_GPU{i}_Temp"]
    else:
        for i in range(deviceCount):
            headers += [f"GPU{i}_Util", f"GPU{i}_Mem", f"GPU{i}_Temp", f"GPU{i}_Power"]
    # Network headers
    if args.ib:
        for dev in ibd:
            mlx = dev["mlx"]
            headers += [
                f"{mlx}_TX_Packets", f"{mlx}_RX_Packets",
                f"{mlx}_TX_Gbps", f"{mlx}_RX_Gbps",
                f"{mlx}_TX_Max", f"{mlx}_RX_Max",
                f"{mlx}_TX_Discards", f"{mlx}_RX_Errors",
                f"{mlx}_Symbol_Errors", f"{mlx}_Link_Downs",
                f"{mlx}_Link_Recovery"
            ]
    else:
        # Basic NIC stats per mlx
        for dev in ibd:
            mlx = dev["mlx"]
            headers += [f"{mlx}_RX_Gbps", f"{mlx}_TX_Gbps", f"{mlx}_RX_Drop", f"{mlx}_TX_Drop",
                        f"{mlx}_RX_Max", f"{mlx}_TX_Max"]
        # RDMA per-priority stats
        for (mlx, port, nic) in rdma_devices_static:
            headers += [f"{mlx}_{nic}_RX_P{i}" for i in range(8)]
            headers += [f"{mlx}_{nic}_TX_P{i}" for i in range(8)]
            headers += [f"{mlx}_{nic}_TOTAL_RX", f"{mlx}_{nic}_TOTAL_TX"]
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(archive_rows)
    print(f"[INFO] Dumped full run history to {filename}")

# ==== rich Table Generators ====
def make_net_table():
    table = Table(title="NIC Stats (Gbps, Drops)", show_lines=False)
    hdrs = [
        ("NIC", 18, 22),
        ("RX Gbps", 8, 8),
        ("RX Max", 8, 8),
        ("TX Gbps", 8, 8),
        ("TX Max", 8, 8),
        ("RX Drop", 9, 9),
        ("TX Drop", 9, 9),
    ]
    for col, minw, maxw in hdrs:
        table.add_column(col, justify="right", min_width=minw, max_width=maxw, no_wrap=True)
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
            f"{net_history[mlx]['rx'][-1]:7.2f}",
            f"{net_max[mlx]['rx']:7.2f}",
            f"{net_history[mlx]['tx'][-1]:7.2f}",
            f"{net_max[mlx]['tx']:7.2f}",
            f"{rx_drp:7d}",
            f"{tx_drp:7d}"
        )
    return table

def make_rdma_table():
    prio_headers = [f"RX P{i}" for i in range(8)] + [f"TX P{i}" for i in range(8)] + ["TOTAL RX", "TOTAL TX"]
    hdrs = [("NIC", 7, 7)] + [(hdr, 8, 8) for hdr in prio_headers]
    table = Table(title="RDMA NIC priority Bandwidth (Gbps)", show_lines=False)
    for col, minw, maxw in hdrs:
        table.add_column(col, justify="right", min_width=minw, max_width=maxw, no_wrap=True)
    rdma_devices = rdma_devices_static
    for mlx, port, nic in rdma_devices:
        rx_delta_mb, tx_delta_mb = get_rdma_delta(mlx, port, nic)
        total_rx = sum(rx_delta_mb)
        total_tx = sum(tx_delta_mb)
        row = [f"{mlx}:{port}"] + [f"{x:7.2f}" for x in rx_delta_mb] + [f"{x:7.2f}" for x in tx_delta_mb] + [f"{total_rx:7.2f}", f"{total_tx:7.2f}"]
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
    hdrs = [
        ("GPU", 3, 3),
        ("Name", 20, 24),
        ("Util (%)", 8, 8),
        ("Util Max", 9, 9),
        ("Mem Usage (GB)", 16, 16),
        ("Mem Max (GB)", 16, 16),
        ("Power (W)", 10, 10),
        ("Temp", 6, 6),
        ("FCLK (MHz)", 8, 8),
        ("MCLK (MHz)", 8, 8),
        ("SCLK (MHz)", 8, 8),
        ("SOCCLK (MHz)", 9, 9),
    ]
    for col, minw, maxw in hdrs:
        table.add_column(col, justify="right", min_width=minw, max_width=maxw, no_wrap=True)
    for row in stats:
        table.add_row(
            str(row.get("GPU", "")),
            str(row.get("Name", "")),
            f"{row.get('Util (%)', 0):>7}",
            f"{row.get('Util Max', 0):>8}",
            f"{row.get('Mem Usage (GB)', '0.0/0.0'):>15}",
            f"{row.get('Mem Max (GB)', '0.0/0.0'):>15}",
            f"{row.get('Power (W)', 0):>9}",
            f"{row.get('Temp', 0):>5}",
            f"{row.get('FCLK (MHz)', ''):>7}",
            f"{row.get('MCLK (MHz)', ''):>7}",
            f"{row.get('SCLK (MHz)', ''):>7}",
            f"{row.get('SOCCLK (MHz)', ''):>8}",
        )
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

# ==== Helpers to gather values for CSV rows ====
def sample_ib_values():
    row_vals = []
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
            tx_gbps = rx_gbps = 0.0
        last_ib_fabric_counters[key] = (now, cur_tx_data, cur_rx_data)
        ib_net_max[mlx]["tx"] = max(ib_net_max[mlx]["tx"], tx_gbps)
        ib_net_max[mlx]["rx"] = max(ib_net_max[mlx]["rx"], rx_gbps)
        row_vals += [
            counters.get("port_xmit_packets", 0),
            counters.get("port_rcv_packets", 0),
            f"{tx_gbps:.4f}", f"{rx_gbps:.4f}",
            f"{ib_net_max[mlx]['tx']:.4f}", f"{ib_net_max[mlx]['rx']:.4f}",
            counters.get("port_xmit_discards", 0),
            counters.get("port_rcv_errors", 0),
            counters.get("symbol_error", 0),
            counters.get("link_downed", 0),
            counters.get("link_error_recovery", 0)
        ]
    return row_vals

def sample_nic_values():
    row_vals = []
    for dev in ibd:
        mlx, port = dev["mlx"], dev["port"]
        rx_gbps, tx_gbps = get_net_sample(mlx, port)
        net_history[mlx]["rx"].append(rx_gbps)
        net_history[mlx]["tx"].append(tx_gbps)
        net_max[mlx]["rx"] = max(net_max[mlx]["rx"], rx_gbps)
        net_max[mlx]["tx"] = max(net_max[mlx]["tx"], tx_gbps)
        rx_drp, tx_drp = get_drop_sample(port)
        row_vals += [f"{rx_gbps:.4f}", f"{tx_gbps:.4f}", rx_drp, tx_drp,
                     f"{net_max[mlx]['rx']:.4f}", f"{net_max[mlx]['tx']:.4f}"]
    return row_vals

def sample_rdma_values():
    row_vals = []
    for (mlx, port, nic) in rdma_devices_static:
        rx_prios, tx_prios = get_rdma_delta(mlx, port, nic)
        total_rx = sum(rx_prios)
        total_tx = sum(tx_prios)
        row_vals += [f"{x:.4f}" for x in rx_prios]
        row_vals += [f"{x:.4f}" for x in tx_prios]
        row_vals += [f"{total_rx:.4f}", f"{total_tx:.4f}"]
    return row_vals

# ==== MAIN LOOP using `rich.Live` ====
try:
    with Live(refresh_per_second=1, screen=True) as live:
        while True:
            loop_start = time.time()
            panels = []
            # Network Tables
            if args.ib:
                panels.append(Panel(make_ib_table(), title="InfiniBand"))
            else:
                panels.append(Panel(make_net_table(), title="NIC"))
                panels.append(Panel(make_rdma_table(), title="RDMA Priority"))
            # GPU Tables
            row = [datetime.now().isoformat()] if args.dump else None

            if args.amd:
                stats, amd_gpu_values, amd_count_local = get_amd_gpu_stats_rocml()
                amd_count = amd_count_local  # update for CSV headers at shutdown
                panels.append(Panel(make_amd_gpu_table(stats)))
                if args.dump:
                    row += amd_gpu_values
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
                if args.dump:
                    row += gpu_values
            # Network collection for dump
            if args.dump:
                if args.ib:
                    row += sample_ib_values()
                else:
                    row += sample_nic_values()
                    row += sample_rdma_values()

            live.update(Group(*panels))

            if args.dump and row:
                archive_rows.append(row)
            intended_interval = 1  # seconds between updates
            elapsed = time.time() - loop_start
            if elapsed < intended_interval:
                time.sleep(intended_interval - elapsed)
except KeyboardInterrupt:
    print("\nExiting...")
    if args.dump:
        dump_archive_csv("stats_dump", is_amd=args.amd, amd_count=(amd_count if args.amd else 0))
    if args.nvidia:
        import pynvml
        pynvml.nvmlShutdown()
