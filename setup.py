from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements, curr_extra = {'general': []}, 'general'
with open('requirements.txt') as rf:
    req = rf.readlines()

for req_line in req:
    if len(req_line.strip()) > 0: # empty line
        if req_line.startswith('#'):
            curr_extra = req_line.replace('#', '').strip()
            requirements[curr_extra] = []
        else:
            requirements[curr_extra].append(req_line.strip())

setup(
    name="strep",
    version="0.0.1",
    author="Raphael Fischer",
    author_email="raphael.fischer@tu-dortmund.de",
    description="Software for sustainable and trustworty reporting (STREP) in ML and AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raphischer/strep",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
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
    install_requires=requirements['general'],
    extras_require={k: req for k, req in requirements.items() if k != 'general'}
)