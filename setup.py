import os
import subprocess
import json
from pathlib import Path
import platform
from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_VERSION_JSON = Path("/usr/local/cuda/version.json")

def get_spconv_version():
    if CUDA_VERSION_JSON.is_file():
        # Read cuda version as integer (11.7.1 --> 117, 12.1.1 --> 121, etc.)
        with open(CUDA_VERSION_JSON, "r") as file:
            data = json.load(file)
        cuda_version = int("".join(data["cuda"]["version"].split('.')[:2]))
        
        # Assign correct spconv 
        if cuda_version < 113:
            raise ValueError("Please install CUDA >= 11.3. It is recommended to use >= 11.4 if possible")
        elif cuda_version == 115:
            spconv_version = f"spconv-cu114"
        elif cuda_version >= 120:
            spconv_version = f"spconv-cu120"
        else:
            spconv_version = f"spconv-cu{cuda_version}"
    else:
        if platform.system() == 'Linux':
            spconv_version = "spconv"
        else:
            raise ValueError(f"CPU version of spconv is only available in LINUX. Please install CUDA toolkit >= 11.3")
    return spconv_version


def get_git_commit_number():
    if not os.path.exists('.git'):
        return '0000000'

    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.0.0'
    write_version_to_file(version, 'pcdet/version.py')

    setup(
        name='pcdet',
        version=version,
        description='OpenPCDet is a general codebase for 3D object detection from point cloud',
        install_requires=[
            'numpy',
            'llvmlite',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            'torch',
            'torchvision',
            'SharedArray==3.1.0',
            'opencv-python',
            'pyquaternion',
            get_spconv_version()
        ],

        author='Shaoshuai Shi',
        author_email='shaoshuaics@gmail.com',
        license='Apache License 2.0',
        packages=find_packages(exclude=['tools', 'data', 'output']),
        cmdclass={
            'build_ext': BuildExtension,
        },
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='pcdet.ops.iou3d_nms',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roiaware_pool3d_cuda',
                module='pcdet.ops.roiaware_pool3d',
                sources=[
                    'src/roiaware_pool3d.cpp',
                    'src/roiaware_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='roipoint_pool3d_cuda',
                module='pcdet.ops.roipoint_pool3d',
                sources=[
                    'src/roipoint_pool3d.cpp',
                    'src/roipoint_pool3d_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='pointnet2_stack_cuda',
                module='pcdet.ops.pointnet2.pointnet2_stack',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu', 
                    'src/interpolate.cpp', 
                    'src/interpolate_gpu.cu',
                    'src/voxel_query.cpp', 
                    'src/voxel_query_gpu.cu',
                    'src/vector_pool.cpp',
                    'src/vector_pool_gpu.cu'
                ],
            ),
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='pcdet.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',

                ],
            ),
            make_cuda_ext(
                name="bev_pool_ext",
                module="pcdet.ops.bev_pool",
                sources=[
                    "src/bev_pool.cpp",
                    "src/bev_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name='ingroup_inds_cuda',
                module='pcdet.ops.ingroup_inds',
                sources=[
                    'src/ingroup_inds.cpp',
                    'src/ingroup_inds_kernel.cu',
                ]
            ),
        ],
    )
