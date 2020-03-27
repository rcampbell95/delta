import multiprocessing
import tensorflow as tf


class GPU_Manager():
    def __init__(self):
        self.cpus = [x.name for x in tf.config.experimental.list_physical_devices('CPU')]
        self.gpus = [x.name for x in tf.config.experimental.list_physical_devices('GPU')]
        self._device_queue = multiprocessing.Queue(0)

        print(self.gpus)
        gpu_count = len(self.gpus)

        self.cpus.append(self.cpus[0])
        self.cpus.append(self.cpus[0])

        if gpu_count == 0:
            for cpu in self.cpus:
                cpu = cpu.replace("physical_device", "device")
                self._device_queue.put(cpu)
        elif gpu_count > 0:
            for gpu in self.gpus:
                gpu = gpu.replace("physical_device", "device")
                self._device_queue.put(gpu)

    def request(self):
        device = self._device_queue.get(block=True)
        return [device]

    def release(self, device_name):
        self._device_queue.put(device_name)
