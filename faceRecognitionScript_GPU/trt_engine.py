import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class TRTModule:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self.input_binding_idxs = []
        self.output_binding_idxs = []

        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                self.input_binding_idxs.append(i)
            else:
                self.output_binding_idxs.append(i)

        # We assume single input / single output (true for SCRFD + ArcFace)
        assert len(self.input_binding_idxs) == 1
        assert len(self.output_binding_idxs) >= 1

        self.bindings = [None] * self.engine.num_bindings

    def infer(self, input_array):
        input_array = np.ascontiguousarray(input_array)

        input_idx = self.input_binding_idxs[0]

        # Set input shape dynamically
        self.context.set_binding_shape(input_idx, input_array.shape)

        # Allocate input buffer
        input_size = input_array.nbytes
        input_device = cuda.mem_alloc(input_size)
        self.bindings[input_idx] = int(input_device)

        # Copy input to device
        cuda.memcpy_htod_async(input_device, input_array, self.stream)

        outputs = []

        # Allocate output buffers AFTER shape is known
        for out_idx in self.output_binding_idxs:
            out_shape = tuple(self.context.get_binding_shape(out_idx))
            out_dtype = trt.nptype(self.engine.get_binding_dtype(out_idx))
            out_size = int(np.prod(out_shape) * np.dtype(out_dtype).itemsize)

            out_device = cuda.mem_alloc(out_size)
            out_host = np.empty(out_shape, dtype=out_dtype)

            self.bindings[out_idx] = int(out_device)
            outputs.append((out_host, out_device))

        # Execute
        self.context.execute_async_v2(self.bindings, self.stream.handle)

        # Copy outputs back
        for out_host, out_device in outputs:
            cuda.memcpy_dtoh_async(out_host, out_device, self.stream)

        self.stream.synchronize()

        # Return single output or list
        if len(outputs) == 1:
            return outputs[0][0]
        else:
            return [o[0] for o in outputs]

