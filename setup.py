from pathlib import Path

from setuptools import find_packages, setup

def parse_requirements(path: Path):
    with path.open() as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and "@" not in line
        ]


def collect_package_data():
    base_dir = Path(__file__).parent
    knn_base = base_dir / "models" / "graspnet" / "knn"
    pointnet_base = base_dir / "models" / "graspnet" / "pointnet2"

    knn_files = [
        str(path.relative_to(knn_base))
        for pattern in [
            "src/*.cpp",
            "src/*.h",
            "src/cpu/*.cpp",
            "src/cpu/*.h",
            "src/cuda/*.cu",
            "src/cuda/*.h",
        ]
        for path in knn_base.glob(pattern)
    ]

    pointnet_files = [
        str(path.relative_to(pointnet_base))
        for pattern in [
            "_ext_src/include/*",
            "_ext_src/src/*.cpp",
            "_ext_src/src/*.cu",
        ]
        for path in pointnet_base.glob(pattern)
    ]

    return {
        "models.graspnet.knn": knn_files,
        "models.graspnet.pointnet2": pointnet_files,
    }


setup(
    name="a2",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    description="Action Prior Alignment",
    author="Kechun Xu",
    author_email="kcxu@zju.edu.cn",
    install_requires=parse_requirements(Path("requirements.txt")),
    package_data=collect_package_data(),
)
