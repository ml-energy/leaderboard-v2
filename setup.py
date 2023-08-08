from setuptools import setup, find_packages

extras_require = {
    "colosseum-controller": [
        "fastapi",
        "fschat==0.2.23",
        "text_generation @ git+https://github.com/ml-energy/text_generation_energy@master",
    ],
    "app": ["plotly==5.15.0", "gradio==3.39.0", "pydantic==1.10.9"],
    "benchmark": ["zeus-ml", "fschat==0.2.23", "tyro", "rich"],
}

extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="spitfight",
    version="0.0.1",
    url="https://github.com/ml-energy/leaderboard",
    packages=find_packages("."),
    extras_require=extras_require,
)
