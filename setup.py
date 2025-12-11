from setuptools import setup, find_packages

def load_requirements():
    reqs = []
    for line in open("requirements.txt"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("--"):
            continue
        reqs.append(line)
    return reqs


setup(
    name='a2',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    description="Action Prior Alignment",
    author='Kechun Xu',
    author_email='kcxu@zju.edu.cn',
    install_requires=load_requirements(),
)
