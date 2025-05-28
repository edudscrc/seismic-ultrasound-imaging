import wgpu
from pathlib import Path
import re


class WebGpuHandler:
    def __init__(self):
        self.shader_module = None
        self.pipeline_layout = None
        self.buffers = []

        self.bind_group_layout_entries = {}
        self.bind_group_layouts = []
        self.bind_group_entries = []

        self.buffers_info = {}

        self.device = wgpu.utils.get_default_device()

    def create_shader_module(self, shader_path, workgroup_size_x=8, workgroup_size_y=8, workgroup_size_z=4):
        self.workgroup_size = (workgroup_size_x, workgroup_size_y, workgroup_size_z)

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
        for v in self.buffers_info.values():
            if v["zero_initialized"]:
                self.buffers.append(
                    self.device.create_buffer(
                        size=v["data"],
                        usage=v["buffer_usage"]
                    )
                )
                print(f"\nCreated buffer:\nSize: {v["data"]}\nGroup: {v["group"]}\nBinding: {v["binding"]}\nName: {v["name"]}")
            else:
                self.buffers.append(
                    self.device.create_buffer_with_data(
                        data=v["data"],
                        usage=v["buffer_usage"]
                    )
                )
                print(f"\nCreated buffer:\nSize: {v["data"].nbytes}\nGroup: {v["group"]}\nBinding: {v["binding"]}\nName: {v["name"]}")
    
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

    def create_compute_pipeline(self, entry_point):
        return self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={
                "module": self.shader_module,
                "entry_point": entry_point,
            }
        )

    def create_bind_groups(self):
        pass
