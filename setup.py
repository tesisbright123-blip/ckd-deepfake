from setuptools import setup, find_packages

setup(
    name="ckd-deepfake",
    version="0.1.0",
    description="Continual Knowledge Distillation for Cross-Generational Deepfake Detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
