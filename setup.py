from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = list(map(lambda x: x.strip(), f.readlines()))

info = {
    "name": "masktect",
    "version": "v0.0.1",
    "maintainer": "Gaurang Tandon, Kanish Tandon, Animesh Sinha",
    "maintainer_email": "1gaurangtandon@gmail.com, therealkanish@gmail.com, animeshsinha.1309@gmail.com",
    "url": "https://github.com/GaurangTandon/mask-detective",
    "license": "Apache License 2.0",
    "packages": find_packages(where="."),
    "entry_points": {"console_scripts": ["masktect-test=masktect.tests:cli"]},
    "description": "A package to find the people in a video feed who are not mask.",
    "long_description": open("README.md").read(),
    "long_description_content_type": "text/markdown",
    "provides": ["masktect"],
    "install_requires": requirements,
    "package_data": {"qleet": ["tests/pytest.ini"]},
    "include_package_data": True,
}

classifiers = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3 :: Only",
]

setup(classifiers=classifiers, **info)
