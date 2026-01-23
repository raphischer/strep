from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="strep",
    version="0.0.3",
    author="Raphael Fischer",
    author_email="raphael.fischer@tu-dortmund.de",
    description="Software for sustainable and trustworty reporting (STREP) in ML and AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphischer/strep",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas==2.2.3",
        "tqdm",
    ],
    extras_require={
        "frontend": [
            "pint",
            "dash",
            "plotly==5.24.1",
            "dash-bootstrap-components",
            "reportlab",
            "PyMuPDF",
            "qrcode",
            "kaleido",
    ]},
)