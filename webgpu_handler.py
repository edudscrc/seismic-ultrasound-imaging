import wgpu
from pathlib import Path
import re


class WebGpuHandler:
    def __init__(self, shader_path, workgroup_size_x=8, workgroup_size_y=8, workgroup_size_z=4):
        self.shader_module = None
        self.pipeline_layout = None
        self.bind_groups = None
        self.buffers = []

        self.bind_group_layout_entries = []
        self.bind_group_entries = []

        self.buffers_info = {}

        self.workgroup_size = (workgroup_size_x, workgroup_size_y, workgroup_size_z)

        self.shader_string = Path(shader_path).read_text(encoding='utf-8')
        for idx, k in enumerate(["wsx", "wsy", "wsz"]):
            self.shader_string = self.shader_string.replace(k, f'{self.workgroup_size[idx]}')

        self.device = wgpu.utils.get_default_device()

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

    def create_shader_module(self):
        self.shader_module = self.device.create_shader_module(code=self.shader_string)

    def create_compute_pipeline(self, entry_point):
        """
        Creates a compute pipeline.

        Arguments:
            entry_point (str): @compute function's name declared in wgsl file.

        Returns:
            GPUComputePipeline.
        """
        return self.device.create_compute_pipeline(
            layout=self.pipeline_layout,
            compute={"module": self.shader_module, "entry_point": entry_point}
        )

    def create_buffer(
            self,
            data,
            binding_number,
            buffer_binding_type,
            bind_groups_layouts_entries: list,
            bind_groups_entries: list
    ):
        """
        Creates a buffer using create_buffer_with_data() and also appends a dictionary with the passed arguments into
        bind_groups_layouts_entries and into bind_groups_entries. (Those two lists are passed by reference).

        Arguments:
            data (dict): Dictionary containing any object supporting the Python buffer protocol.
                         It's the data that will be passed to bindings on gpu.
            binding_number (int): The number 'x' specified in @binding(x).
            buffer_binding_type (enum): WebGPU binding type (read, read_only, etc...).
            bind_groups_layouts_entries (list): Passed by reference. An element of this list is a dict with parameters
                                                that will help to build the bind_group_layout.
            bind_groups_entries (list): Passed by reference. An element of this list is a dict with parameters that will
                                        help to build the bind_group.

        Returns:
            GPUBuffer object: Buffer created with create_buffer_with_data().
        """
        if binding_number == "0" and buffer_binding_type == wgpu.BufferBindingType.storage:
            new_buffer = self.device.create_buffer_with_data(data=data,
                                                         usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        else:
            new_buffer = self.device.create_buffer_with_data(data=data,
                                                         usage=wgpu.BufferUsage.STORAGE)

        bind_groups_layouts_entries.append({
            'binding': binding_number,
            'visibility': wgpu.ShaderStage.COMPUTE,
            'buffer': {
                "type": buffer_binding_type,
            }
        })

        bind_groups_entries.append({
            "binding": binding_number,
            "resource": {
                "buffer": new_buffer,
                "offset": 0,
                "size": new_buffer.size,
            }
        })

        return new_buffer

    def create_pipeline_layout(self, bind_groups_layouts_entries: list, bind_groups_entries: list):
        """
        Creates bind_group_layouts and bind_groups from the arguments to create a pipeline_layout. Sets the class'
        pipeline layout and bind_groups.

        Arguments:
            bind_groups_layouts_entries (list): An element of this list is a dict with parameters that will help
                                                to build the bind_group_layout.
            bind_groups_entries (list): An element of this list is a dict with parameters that will help
                                        to build the bind_group.
        """
        bind_groups_layouts = []
        for bind_group_layout_entries in bind_groups_layouts_entries:
            bind_groups_layouts.append(self.device.create_bind_group_layout(entries=bind_group_layout_entries))

        self.pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=bind_groups_layouts)

        bind_groups = []
        for index, bind_group_entries in enumerate(bind_groups_entries):
            bind_groups.append(self.device.create_bind_group(layout=bind_groups_layouts[index],
                                                             entries=bind_group_entries))

        self.bind_groups = bind_groups


def read_shader_bindings(shader_lines: list):
    """
    Creates a dictionary containing important information about bindings on wgsl file.

    Arguments:
        shader_lines (list): List of strings where each element is a line of a wgsl file.

    Returns:
        dict: a dictionary containing the group's number, binding's number, binding's type (read or read_only) and the
              binding's name.
    """
    shader_bindings = dict()

    last_group = None
    last_binding = None

    for line in shader_lines:
        if line.find(f'@group(') != -1:
            current_group = ''.join(line.split('@group(')[1].split(')')[0])
            current_binding = ''.join(line.split('@binding(')[1].split(')')[0])

            if f'{current_group}' not in shader_bindings:
                shader_bindings[f'{current_group}'] = dict()

            shader_bindings[f'{current_group}'][f'{current_binding}'] = []

            last_group = current_group
            last_binding = current_binding

        elif line.find('var<storage,') != -1:
            buffer_binding_type = ''.join(line.split('var<storage,')[1].split('>')[0])

            binding_name = ''.join(line.split('>')[1].split(':')[0])
            binding_name = binding_name.strip()

            shader_bindings[last_group][last_binding].append(binding_name)
            shader_bindings[last_group][last_binding].append(buffer_binding_type)

    return shader_bindings
