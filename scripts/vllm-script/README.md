WIP: temporarily storing some common commands here

# run tgi server
```
token=<TOKEN>
docker run --gpus all --shm-size 1g -p 8000:80 \
        -e HUGGING_FACE_HUB_TOKEN=$token \
        -v $PWD/Llama-2-7b-chat-hf-tokenizer_config.json:/app/tokenizer_config.json \
        --name tgi_v1.4.0 \
        tgi_v1.4.0 \
        --tokenizer-config-path /app/tokenizer_config.json \
        --model-id meta-llama/Llama-2-7b-chat-hf
```
Make sure to use `--gpus device=0` if using `benchmark_server_zeus.py`.

# client side: run benchmark script
```
python benchmark_server.py \
        --backend tgi \
        --dataset ../../sharegpt/ShareGPT_V3_filtered.json \
        --model meta-llama/Llama-2-7b-hf \
        --host 127.0.0.1 \
        --num-prompts 2  # default=1000
```

# client side: docker way (need to use docker network here and on backend)
Dockerfile:
```
# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /benchmark

# Copy the current directory contents into the container at /benchmark
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run script.py when the container launches
ENTRYPOINT ["python", "benchmark_server.py"]
```
CLI:
```
docker build -t benchmark:latest .
docker run \
        -v /home/ohjun/leaderboard/sharegpt:/data \
        benchmark:latest \
        --backend tgi \
        --dataset /data/ShareGPT_V3_filtered.json \
        --model meta-llama/Llama-2-7b-hf \
```

# requirements.txt
```
argparse
asyncio
aiohttp
numpy
tqdm
```