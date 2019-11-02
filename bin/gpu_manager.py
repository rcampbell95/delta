import multiprocessing
import queue
from tensorflow.python.client import device_lib


class GPU_Manager():
    def __init__(self):
        self.gpu_queue = queue.Queue()
        self.default_device = "/device:CPU:0"
        local_device_protos = device_lib.list_local_devices()

        gpu_count = 0
        for x in local_device_protos: 
            if "gpu" in x.device_type.lower():
                gpu_count += 1
                self.gpu_queue.put(x.name) 
        if gpu_count == 0:
            self.gpu_available = False
            
    def request_device(self):
        if self.gpu_available:
            gpu_name = self.gpu_queue.get()
            return gpu_name
        else:
            return self.default_device

    def free_device(self, device_name):
        self.gpu_queue.put(device_name)


manager = GPU_Manager()

device_name = manager.request_device()

print(device_name)

manager.free_device(device_name)

print(manager.request_device())

print(manager.request_device())

manager.request_device()