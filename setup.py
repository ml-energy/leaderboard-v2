from setuptools import setup, find_packages

setup(
    name="spitfight",
    version="0.0.1",
    url="https://github.com/ml-energy/leaderboard",
    packages=find_packages("."),
    extras_require={
        # TODO: text_generation actually means our internal fork.
        #       One way to go about this is to open source only the python client
        #       in a separate repo and then use that as a dependency here (git+https).
        "colosseum-controller": ["fastapi", "fschat==0.2.20", "./text_generation"],
        "app": ["plotly==5.15.0", "gradio==3.39.0", "pydantic==1.10.9"],
        "benchmark": ["zeus-ml", "fschat==0.2.20", "tyro", "rich"],
    }
)
