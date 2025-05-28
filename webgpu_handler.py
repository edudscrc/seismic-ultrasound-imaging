import wgpu
from wgpu.backends import wgpu_native
import re
from pathlib import Path


class WebGpuHandler:
    def __init__(self):
        self.shader_module = None
        self.pipeline_layout = None

        self.buffers = []

        self.bind_group_layout_entries = {}
        self.bind_group_layouts = []

        self.bind_group_entries = {}
        self.bind_groups = []

        self.buffers_info = {}

        self.device = wgpu.utils.get_default_device()

    def create_shader_module(self, shader_path, roi_size, workgroup_size: tuple):
        self.workgroup_size = list(workgroup_size)
        roi_size = list(roi_size)
        self.num_workgroups_to_dispatch = []

        while len(self.workgroup_size) < 3:
            self.workgroup_size.append(1)
        
        while len(roi_size) < 3:
            roi_size.append(1)

        for i in range(3):
            self.num_workgroups_to_dispatch.append((roi_size[i] + self.workgroup_size[i] - 1) // self.workgroup_size[i])

        self.shader_string = Path(shader_path).read_text(encoding='utf-8')
        for idx, k in enumerate(["wsx", "wsy", "wsz"]):
            self.shader_string = self.shader_string.replace(k, f'{self.workgroup_size[idx]}')

        self.shader_module = self.device.create_shader_module(code=self.shader_string)

    def set_buffers(self, data, *copy_src_buffers):
        re_pattern = r"@group\((\d+)\)\s+@binding\((\d+)\)\s+var<([^>]+)>\s+(\w+)\s*:"
        matches = re.findall(re_pattern, self.shader_string)

        for m in matches:
            binding_types = m[2].split(",")

            if "storage" in binding_types:
                bu = wgpu.BufferUsage.STORAGE
                if "read_write" in binding_types:
                    bt = wgpu.BufferBindingType.storage
                else:
                    bt = wgpu.BufferBindingType.read_only_storage
            elif "uniform" in binding_types:
                bu = wgpu.BufferUsage.UNIFORM
                bt = wgpu.BufferBindingType.uniform
            
            if m[3] in copy_src_buffers:
                bu = bu | wgpu.BufferUsage.COPY_SRC

            bu = bu | wgpu.BufferUsage.COPY_DST

            self.buffers_info[f"g{m[0]}b{m[1]}"] = {
                "group": int(m[0]),
                "binding": int(m[1]),
                "binding_type": bt,
                "buffer_usage": bu,
                "name": m[3],
                "data": data[m[3]][0],
                "zero_initialized": data[m[3]][1],
            }

    def create_buffers(self):
        command_encoder = self.device.create_command_encoder()
        cleared_buffer = False

        for v in self.buffers_info.values():
            if v["zero_initialized"]:
                self.buffers.append(
                    self.device.create_buffer(
                        size=v["data"],
                        usage=v["buffer_usage"]
                    )
                )
                command_encoder.clear_buffer(self.buffers[-1], 0, self.buffers[-1].size)
                cleared_buffer = True
                print(f"\nCreated buffer:\nSize: {v["data"]}\nGroup: {v["group"]}\nBinding: {v["binding"]}\nName: {v["name"]}")
            else:
                self.buffers.append(
                    self.device.create_buffer_with_data(
                        data=v["data"],
                        usage=v["buffer_usage"]
                    )
                )
                print(f"\nCreated buffer:\nSize: {v["data"].nbytes}\nGroup: {v["group"]}\nBinding: {v["binding"]}\nName: {v["name"]}")
        
        if cleared_buffer:
            self.device.queue.submit([command_encoder.finish()])
    
    def create_bind_group_layouts(self):
        for v in self.buffers_info.values():
            if self.bind_group_layout_entries.get(v["group"]) is None:
                self.bind_group_layout_entries.update({v["group"]: []})

            self.bind_group_layout_entries[v["group"]].append(
                {
                    "binding": v["binding"],
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {
                        "type": v["binding_type"],
                    },
                }
            )
        
        for v in self.bind_group_layout_entries.values():
            self.bind_group_layouts.append(
                self.device.create_bind_group_layout(entries=v)
            )
    
    def create_pipeline_layout(self):
        self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=self.bind_group_layouts)


    def create_bind_groups(self):
        for idx, v in enumerate(self.buffers_info.values()):
            if self.bind_group_entries.get(v["group"]) is None:
                self.bind_group_entries.update({v["group"]: []})
            
            self.bind_group_entries[v["group"]].append(
                {
                    "binding": v["binding"],
                    "resource": {
                        "buffer": self.buffers[idx],
                        "offset": 0,
                        "size": self.buffers[idx].size,
                    },
                }
            )

        for k, v in self.bind_group_entries.items():
            self.bind_groups.append(
                self.device.create_bind_group(layout=self.bind_group_layouts[k], entries=v)
            )

    def create_compute_pipeline(self, entry_point):
        return self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={
                "module": self.shader_module,
                "entry_point": entry_point,
            }
        )
    
    def dispatch_workgroups_to_pipeline(self, compute_pass: wgpu_native._api.GPUComputePassEncoder, compute_pipeline: wgpu_native._api.GPUComputePipeline, workgroups_to_dispatch=None):
        compute_pass.set_pipeline(compute_pipeline)
        if workgroups_to_dispatch is None:
            compute_pass.dispatch_workgroups(self.num_workgroups_to_dispatch[0], self.num_workgroups_to_dispatch[1], self.num_workgroups_to_dispatch[2])
        else:
            while len(workgroups_to_dispatch) < 3:
                workgroups_to_dispatch.append(1)
            compute_pass.dispatch_workgroups(workgroups_to_dispatch[0], workgroups_to_dispatch[1], workgroups_to_dispatch[2])
