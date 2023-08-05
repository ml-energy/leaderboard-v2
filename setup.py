from setuptools import setup, find_packages

setup(
    name="spitfight",
    version="0.0.1",
    url="https://github.com/ml-energy/leaderboard",
    packages=find_packages("."),
    install_requires=[
        "plotly==5.15.0",
        "gradio==3.35.2",
        "pydantic==1.10.9",
    ]
)
