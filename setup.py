from setuptools import find_packages, setup

setup(
    name="crisisflow",
    version="1.0.0",
    description="CrisisFlow disaster-response sequential environment (OpenEnv-style API).",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "numpy",
        "pyyaml",
        "python-multipart",
    ],
)
