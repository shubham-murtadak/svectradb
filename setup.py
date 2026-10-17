from setuptools import setup, find_packages

setup(
    name="svectradb",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "svectra": [
            "server/static/*",
            "server/static/**/*",
        ],
    },
)
