import multiprocessing
import tensorflow as tf


class GPU_Manager():
    def __init__(self):
        self.device_queue = multiprocessing.Queue()

        self.cpus = [x.name for x in tf.config.experimental.list_logical_devices('CPU')]
        self.gpus = [x.name for x in tf.config.experimental.list_logical_devices('GPU')]

        gpu_count = len(self.gpus)

        if gpu_count == 0:
            for cpu in self.cpus:
                self.device_queue.put(cpu)
        elif gpu_count > 0:
            for gpu in self.gpus:
                self.device_queue.put(gpu)

    def request_device(self):
        return self.device_queue.get()

    def free_device(self, device_name):
        self.device_queue.put(device_name)