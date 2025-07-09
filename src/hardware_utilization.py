import platform
import psutil
import subprocess
import pynvml


def get_cpu_info():
    system = platform.system()

    if system == "Windows":
        return get_cpu_info_windows()
    elif system == "Linux":
        return get_cpu_info_linux()
    else:
        return "Unsupported OS"


def get_cpu_info_windows():
    try:
        result = subprocess.run(["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed"],
                                capture_output=True, text=True)
        cpu_info = result.stdout.strip()
        return cpu_info
    except Exception as e:
        return f"An error occurred: {e}"


def get_cpu_info_linux():
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True)
        cpu_info = result.stdout.strip()
        return cpu_info
    except Exception as e:
        return f"An error occurred: {e}"


def get_system_utilization():
    cpu_util = get_cpu_utilization()
    ram_util = get_ram_utilization()

    gpu_util, vram_util = get_gpu_utilization()

    print(f"CPU Utilization: {cpu_util}%")
    print(f"RAM Utilization: {ram_util}%")
    print(f"GPU Utilization: {gpu_util}%")
    print(f"VRAM Utilization: {vram_util} MB")


def get_cpu_utilization():
    return psutil.cpu_percent(interval=1)


def get_ram_utilization():
    memory = psutil.virtual_memory()
    return memory.percent


def get_gpu_utilization():
    pynvml.nvmlInit()

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)

        vram_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        return gpu_utilization.gpu, vram_info.used / 1024 ** 2
    except Exception as e:
        return f"Error: {e}"


def main():
    print("CPU Info:")
    print(get_cpu_info())

    print("\nSystem Utilization:")
    get_system_utilization()


main()
