from collections import defaultdict
from multiprocessing import Process, Event
from datetime import datetime
import os
import argparse
import json
import time
import subprocess
import platform
import re

import numpy as np


VISIBLE_GPUS = [int(did) for did in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if did.isnumeric()]
if -1 in VISIBLE_GPUS:
    VISIBLE_GPUS = VISIBLE_GPUS[:VISIBLE_GPUS.index(-1)]


# TODO improve RAPL compability, maybe using https://github.com/djselbeck/rapl-read-ryzen


def aggregate_monitoring_log(fpath):
    if not os.path.isfile(fpath):
        return None
    results = defaultdict(dict)
    with open(fpath, 'r') as fc:
        log = json.load(fc)
        device_fields = [key for key in list(log.values())[0].keys() if key not in ['timestamp']]
        for gpu_id, meas in log.items():
            if 'duration' not in meas:
                meas['duration'] = [(ts - meas['timestamp'][i-1]) / 1000 if i > 0 else 0 for i, ts in enumerate(meas['timestamp'])]
            results[gpu_id]['start_unix'] = min(meas['timestamp']) / 1000
            results[gpu_id]['end_unix'] = max(meas['timestamp']) / 1000
            results[gpu_id]['start_utc'] = datetime.utcfromtimestamp(results[gpu_id]['start_unix']).strftime('%Y-%m-%d %H:%M:%S')
            results[gpu_id]['end_utc'] = datetime.utcfromtimestamp(results[gpu_id]['end_unix']).strftime('%Y-%m-%d %H:%M:%S')
            results[gpu_id]['total_duration'] = results[gpu_id]['end_unix'] - results[gpu_id]['start_unix']
            results[gpu_id]['nr_measurements'] = len(meas['timestamp'])
            for field in device_fields:
                results[gpu_id][field] = {m.__name__: m(meas[field]) for m in [min, max, np.mean, np.std]}
            for power_key in ['power_usage', 'package-0']:
                if power_key in results[gpu_id]:
                    results[gpu_id]['total_power_draw'] = sum([d * p for d, p in zip(log[gpu_id]['duration'], log[gpu_id][power_key])])
                    break
            else:
                results[gpu_id]['total_power_draw'] = -1
    # aggregate over all devices
    results['total'] = {
        'start_unix': min([val['start_unix'] for val in results.values()]),
        'end_unix': max([val['end_unix'] for val in results.values()]),
        'nr_measurements': sum([val['nr_measurements'] for val in results.values()]),
        'total_power_draw': sum([val['total_power_draw'] for val in results.values()]),
    }
    results['total']['start_utc'] = datetime.utcfromtimestamp(results['total']['start_unix']).strftime('%Y-%m-%d %H:%M:%S')
    results['total']['end_utc'] = datetime.utcfromtimestamp(results['total']['end_unix']).strftime('%Y-%m-%d %H:%M:%S')
    results['total']['total_duration'] = results['total']['end_unix'] - results['total']['start_unix']
    return results


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


def monitor_pynvml(interval, logfile, stopper, gpu_id):
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage, nvmlDeviceGetMemoryInfo
    from pynvml import nvmlDeviceGetUtilizationRates, nvmlDeviceGetTemperature, NVML_TEMPERATURE_GPU, nvmlDeviceGetCount
    nvmlInit()
    out = defaultdict(lambda: defaultdict(list))
    if gpu_id is None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            gpu_id = VISIBLE_GPUS
        else:
            deviceCount = nvmlDeviceGetCount()
            gpu_id = list(range(deviceCount))
    else:
        if not isinstance(gpu_id, list):
            gpu_id = [gpu_id]
    if len(gpu_id) < 1:
        print('No monitoring with pynvml possible!\nCould not access any GPU.')
        return
    print('PROFILING FOR GPUs', gpu_id)
    i = 0
    while not stopper.is_set():
        # TODO check amount of stored data and flush out if necessary
        start = time.time()
        i += 1
        for gid in gpu_id:
            handle = nvmlDeviceGetHandleByIndex(gid)
            memory_t = nvmlDeviceGetMemoryInfo(handle)
            utilization_t = nvmlDeviceGetUtilizationRates(handle)
            out[gid]['temperature'].append(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)),
            out[gid]['util.gpu'].append(utilization_t.gpu),
            out[gid]['util.memory'].append(utilization_t.memory),
            out[gid]['power_usage'].append(nvmlDeviceGetPowerUsage(handle) / 1000.0),
            out[gid]['memory.used'].append(memory_t.used),
            out[gid]['memory.free'].append(memory_t.free),
            out[gid]['timestamp'].append((datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0)
        profile_duration = time.time() - start
        sleep_time = interval - profile_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
    print(f'Wrote {i} GPU profilings to {logfile}')
    with open(logfile, 'w') as log:
        json.dump(out, log)


def monitor_psutil(interval, logfile, stopper, process_id):
    try:
        import psutil
    except ImportError as e:
        print('No monitoring with psutil possible!\n', e)
        return
    proc = psutil.Process(process_id)
    out = defaultdict(lambda: defaultdict(list))
    if not isinstance(process_id, list):
        process_id = [process_id]
    print('PROFILING FOR PROCESSES', process_id)
    i = 0
    while not stopper.is_set():
        # TODO check amount of stored data and flush out if necessary
        start = time.time()
        i += 1
        for gid in process_id:
            mem_info = proc.memory_full_info()
            out[gid]['cpu_num'].append(proc.cpu_num()),
            out[gid]['cpu_percent'].append(proc.cpu_percent()),
            for attr in ['rss', 'vms', 'shared', 'text', 'lib', 'data', 'uss', 'pss']:
                out[gid][f'mem_{attr}'].append(getattr(mem_info, attr))
            out[gid]['timestamp'].append((datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0)
        profile_duration = time.time() - start
        sleep_time = interval - profile_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
    print(f'Wrote {i} process profilings to {logfile}')
    with open(logfile, 'w') as log:
        json.dump(out, log)


def monitor_pyrapl(interval, logfile, stopper, process_id):
    try:
        import rapl
        print('PROFILING WITH RAPL')
    except ImportError as e:
        print('No monitoring with RAPL possible!\n', e)
        return
    monitor = rapl.RAPLMonitor
    sample = monitor.sample()
    out = defaultdict(lambda: defaultdict(list))
    i = 0
    while not stopper.is_set():
        i += 1
        start = time.time()
        new_sample = monitor.sample()
        diff = new_sample - sample
        out['0']['duration'].append(diff.duration)
        for d in diff.domains:
            domain = diff.domains[d]
            out['0'][domain.name].append(diff.average_power(package=domain.name))
            for sd in domain.subdomains:
                subdomain = domain.subdomains[sd].name
                out['0'][f'{domain.name}.{subdomain}'].append((diff.average_power(package=domain.name, domain=subdomain)))
        out['0']['timestamp'].append((datetime.now() - datetime.utcfromtimestamp(0)).total_seconds() * 1000.0)
        sample = new_sample
        profile_duration = time.time() - start
        sleep_time = interval - profile_duration
        if sleep_time > 0:
            time.sleep(sleep_time)
    print(f'Wrote {i} RAPL profilings to {logfile}')
    with open(logfile, 'w') as log:
        json.dump(out, log)


def monitor_flops_papi(prof_func, event='PAPI_DP_OPS'):
    try:
        from pypapi import events, papi_high
        print('PROFILING WITH PAPI')
        papi_high.start_counters([getattr(events, event)])
        prof_func()
        return papi_high.stop_counters()
    except Exception as e:
        print('No monitoring with PAPI possible!\nMake sure that papi-tools and python-papi are available.\n', e)
        return -1


class Monitoring:

    def __init__(self, gpu_interval, cpu_interval, output_dir, prefix=None) -> None:
        if prefix is None:
            prefix = ''
        else:
            prefix = prefix + '_'
        self.monitoring = []
        if gpu_interval > 0:
            self.monitoring.append(Monitor(monitor_pynvml, interval=gpu_interval, outfile=os.path.join(output_dir, f'{prefix}pynvml.monitoring')))
        if cpu_interval > 0:
            self.monitoring.append(Monitor(monitor_psutil, interval=cpu_interval, outfile=os.path.join(output_dir, f'{prefix}psutil.monitoring'), device_id=os.getpid()))
            self.monitoring.append(Monitor(monitor_pyrapl, interval=cpu_interval, outfile=os.path.join(output_dir, f'{prefix}pyrapl.monitoring')))
        time.sleep(1 * min([cpu_interval, gpu_interval]))

    def stop(self):
        for monitor in self.monitoring:
            monitor.stop()


class Monitor:

    def __init__(self, prof_func, interval=1, outfile=None, device_id=None) -> None:
        if outfile is None:
            raise NotImplementedError('Invalid filename to write monitoring results to:', outfile)
        self.stopper = Event()
        self.p = Process(target=prof_func, args=(interval, outfile, self.stopper, device_id))
        self.p.start()

    def stop(self):
        self.stopper.set() # stops loop in profiling processing
        self.p.join()
