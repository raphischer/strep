from multiprocessing import Process, Event
import os
import json
import time
import subprocess
import platform
import re
import sys

import numpy as np


def init_monitoring(monitor_interval, output_dir):
    try: # use codecarbon if available
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(measure_power_secs=monitor_interval, log_level='warning', country_iso_code="DEU", save_to_file=True, output_dir=output_dir)
        tracker.start()
    except ImportError:
        try: # use Jetson / jtop
            tracker = JetsonMonitor(monitor_interval, os.path.join(output_dir, 'emissions.csv'))
        except ImportError:
            raise NotImplementedError('Codecarbon and jtop not available, no other profiling implemented!')
    # time.sleep(monitor_interval) # allow for initialization
    return tracker


class JetsonMonitor:

    def __init__(self, interval=1, outfile='emissions.csv') -> None:
        pass
        from jtop import jtop
        self.jetson = jtop()
        start_failed = 0
        while start_failed > -1:
            start_failed += 1
            try:
                a = self.jetson.start()
                start_failed = -1
            except Exception as e: # sometimes jtop crashes and needs to be restarted?
                res = os.system('sudo /usr/bin/systemctl restart jtop.service')
                print(f'  restarting jtop service - {res}')
                time.sleep(5)
                if start_failed > 10:
                    sys.exit(1)
        self.stopper = Event()
        self.jetson.ok()
        time.sleep(0.5)
        self.p = Process(target=monitor_jetson, args=(self.jetson, interval, outfile, self.stopper))
        self.p.start()

    def stop(self):
        self.stopper.set() # stops loop in profiling processing
        self.p.join()
        self.jetson.close()


def monitor_jetson(jetson, interval, logfile, stopper):
    stats = ["CPU1", "CPU2", "CPU3", "CPU4", "CPU5", "CPU6", "CPU7", "CPU8", "CPU9", "CPU10", "CPU11", "RAM", "GPU"]
    pwr_stats = ["power", "avg"]
    t0 = time.time()
    jetson_entries = { stat: [jetson.stats[stat]] for stat in stats }
    for pwr_stat in pwr_stats:
        jetson_entries[f'power_total_{pwr_stat}'] = [jetson.power['tot'][pwr_stat]]
    while not stopper.is_set() and jetson.ok():
        start = time.time()
        for stat in stats:
            jetson_entries[stat].append(jetson.stats[stat])
        for pwr_stat in pwr_stats:
            jetson_entries[f'power_total_{pwr_stat}'].append(jetson.power['tot'][pwr_stat])
        profile_duration = time.time() - start
        sleep_time = interval - profile_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
    # once stopped, write the output file (in similar fashion as the emissions.csv by codecarbon)
    header = '' if os.path.isfile(logfile) else ','.join(['duration', 'energy_consumed', 'total_cpu', 'no_meas', 'nvp model'] + [key for key in jetson_entries.keys() if 'CPU' not in key]) + '\n'
    jstr_parts = [
        float(time.time() - t0),         # duration in seconds
        0,                               # energy_consumed
        0,                               # total_cpu
        len(jetson_entries['RAM']),      # no_meas
        jetson.stats['nvp model']        # model
    ]
    for key, val in jetson_entries.items():
        if "CPU" not in key:
            jstr_parts.append(np.mean(val))
        else:
            jstr_parts[2] += np.mean(val)
    jstr_parts[1] = np.mean(jetson_entries['power_total_power']) * jstr_parts[0] / 3.6e9 # milliwatt to mWs to kWh
    with open(logfile, 'a') as outf:
        outf.write(header)
        outf.write(','.join([str(part) for part in jstr_parts]) + '\n')


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip().decode('ascii')
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1).strip()
    return ""


def log_system_info(filename):
    sysinfo = {}
    uname = platform.uname()
    sysinfo.update({
        "System": uname.system,
        "Node Name": uname.node,
        "Release": uname.release,
        "Version": uname.version,
        "Machine": uname.machine,
        "Processor": get_processor_name(),
    })
    try:
        import psutil
        cpufreq = psutil.cpu_freq()
        svmem = psutil.virtual_memory()
        sysinfo.update({
            "Physical cores": psutil.cpu_count(logical=False),
            "Total cores": psutil.cpu_count(logical=True),
            # CPU frequencies
            "Max Frequency": cpufreq.max,
            "Min Frequency": cpufreq.min,
            "Current Frequency": cpufreq.current,
            # System memory
            "Total": svmem.total,
            "Available": svmem.available,
            "Used": svmem.used
        })
    except ImportError:
        pass
    try:
        import GPUtil
        sysinfo["GPU"] = {}
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            if "CUDA_VISIBLE_DEVICES" not in os.environ or gpu.id in VISIBLE_GPUS:
                sysinfo["GPU"][gpu.id] = {
                    "Name": gpu.name,
                    "Memory": gpu.memoryTotal,
                    "UUID": gpu.uuid
                }
    except ImportError:
        pass
    # write file
    with open(filename, "w") as f:
        json.dump(sysinfo, f, indent=4)
