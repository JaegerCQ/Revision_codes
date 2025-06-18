import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3DC
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.mapgen import make_cart_circle
import matplotlib.pyplot as plt
from kwave.utils.data import scale_SI
from scipy.ndimage import uniform_filter
import h5py
import torch


def read_ply_to_3d_array(file_path, scale=1):
    # 打开并读取.ply文件
    with open(file_path, 'r') as f:
        # 读取头部直到end_header
        while True:
            line = f.readline().strip()
            if line == "end_header":
                break
        
        # 逐点读取并处理数据
        points = []  
        for line in f:
            parts = line.split()
            x, y, z = [float(parts[i])*scale for i in range(3)]  # 缩放坐标
            index = int(parts[3])  # 索引
            points.append((np.round(x).astype(int), np.round(y).astype(int), np.round(z).astype(int), index))
            
    # 统计缩放后的点云坐标范围
    max_dims = np.max(np.array(points)[:,:3], axis=0)
    array_3d = np.zeros((max_dims[0]+1, max_dims[1]+1, max_dims[2]+1), dtype=np.int32)
    
    # 映射点云索引到三维数组
    for x, y, z, index in points:
        array_3d[x, y, z] = index


    d2 = array_3d.shape[1]
    d3 = array_3d.shape[2]
    print(d2)
    shrunken_arr = array_3d[:,2*(d2//3):d2,0:3*(d3//4)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step1: 读取.mat文件
    print("Starting to read .mat file...")

    # Step2: 转换为PyTorch tensor并移动到GPU
    tensor_data = torch.tensor(shrunken_arr, dtype=torch.float32, device=device)
    print("Data has been successfully transferred to the GPU.")

    # 过滤和准备点云数据
    # 初始化点列表
    points = []

    value_threshold = 0.5
    tensor_data = tensor_data.permute(2, 1, 0)  

    # 使用布尔索引来查找满足条件的元素
    mask = tensor_data > value_threshold
    indices = mask.nonzero().to('cpu')


    for idx in indices:
        # 转回CPU以处理非Tensor操作
        z, y, x = idx.numpy()
        value = tensor_data[z, y, x].item()
        index = int(value)
        points.append([x, y, z, index])



    # 过滤和筛选数据点步骤完成：
    print("Data filtering and preparation completed.")

    # Step4: 写入.ply文件，使用tqdm检视进度
    print("Starting to write .ply file...")

    with open("cut_part.ply", "w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar index\n")
        ply_file.write("end_header\n")
        
        for point in tqdm(points, desc="Writing PLY"):
            ply_file.write(f"{point[0]} {point[1]} {point[2]} {point[3]}\n")
            
    print("PLY file writing has been completed successfully.")

    return shrunken_arr


def insert_source_P0_in_grid(array_input):
    array_3d = array_input
    small_dim = array_3d.shape
    print(small_dim)
    large_dim = (400, 1000, 800)

    # 创建大数组，并用0初始化
    large_array = np.zeros(large_dim, dtype=np.int32)

    # 计算大数组的中心坐标
    center_large_x, center_large_y, center_large_z = [d // 2 for d in large_dim ]

    # 计算小数组的一半尺寸
    half_small_x, half_small_y, half_small_z = [d // 2 for d in small_dim]

    # 计算填充的起始坐标
    start_x = center_large_x - half_small_x
    start_y = center_large_y - half_small_y
    start_z = center_large_z - half_small_z

    # 确定填充的结束坐标
    end_x = start_x + small_dim[0]
    end_y = start_y + small_dim[1]
    end_z = start_z + small_dim[2]

    d2 = array_3d.shape[0]
    large_array[0:d2, start_y:end_y, start_z:end_z] = array_3d
    large_positive_elements = large_array[large_array > 0]
    large_num_positive_elements = large_positive_elements.size
    print(large_num_positive_elements)

    return large_array

def generate(source_array):
    # 定义网格大小
    Nx: int = 400     # 数量沿x轴的网格点
    Ny: int = 1000    # 数量沿y轴的网格点
    Nz: int = 800   # 数量沿z轴的网格点

    x: float = 80e-3
    y: float = 200e-3
    z: float = 160e-3


    # grid point spacing in the x direction [m]
    dx: float = x / Nx #0.2mm
    dy: float = y / Ny
    dz: float = z / Nz

    # 定义物理属性
    sound_speed = 1500   # 声速 [m/s]
    density = 1000       # 密度 [kg/m^3]

    # 初始化网格（声学介质）
    medium = kWaveMedium(sound_speed=sound_speed, density=density)

    source = kSource()
    source.p0 = source_array  # 将声压赋值到声源
    sensor = kSensor()
    sensor.mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    sensor.mask[275, 150:850:2, 50:750:2] = True
    sensor.record = ["p"]


    # 设置仿真时间
    kgrid = kWaveGrid([Nx, Ny, Nz],[dx, dy, dz])
    kgrid.setTime(4096, 25e-9) 

    simulation_options = SimulationOptions(
        save_to_disk=True,
        data_cast="single",
    )

    execution_options = SimulationExecutionOptions(is_gpu_simulation=False, delete_data=False, verbose_level=2)

    sensor_data_3D = kspaceFirstOrder3DC(
        medium=medium,
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )
    out = sensor_data_3D["p"]
    print(out)
    print(out.shape)

    # plot the simulations
    t_end: float = 4096e-9  # [s]
    t_sc, t_scale, t_prefix, _ = scale_SI(t_end)
    _, ax1 = plt.subplots()
    selected_data = sensor_data_3D["p"][:, 2]
    # 规范化选中的数据
    normalized_data = selected_data / np.max(np.abs(selected_data))
    ax1.plot(np.squeeze(kgrid.t_array * t_scale), normalized_data, "k-", label="3D column 12")
    ax1.set(xlabel=f"Time [{t_prefix}s]", ylabel="Recorded Pressure [au]")
    ax1.grid(True)
    ax1.legend(loc="upper right")
    plt.show()
    
    
    np.save('sensor_data.dat', sensor_data_3D["p"])

if __name__ == "__main__":
    sound_source = read_ply_to_3d_array("data/cloud_full_sampled.ply",1)
    print(sound_source.shape)
    stimulate_space = insert_source_P0_in_grid(sound_source)
    print(stimulate_space.shape)
    generate(stimulate_space)

