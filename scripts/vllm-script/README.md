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
python benchmark_server_zeus.py \
        --backend tgi \
        --dataset ../../sharegpt/ShareGPT_V3_filtered_2000.json \
        --port 8000 \
        --model meta-llama/Llama-2-7b-chat-hf \
        --out-filename /home/ohjun/leaderboard/scripts/vllm-script/results/llama2-7b_gpu0_rr1.txt \
        --request-rate 0.25
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
        --gpus device=0 \
        --name llama7 \
        --network benchmark-net \
        -v /home/ohjun/leaderboard/sharegpt:/data \
        -v /home/ohjun/leaderboard/scripts/vllm-script/results:/results \
        -d benchmark:latest \
        --backend tgi \
        --dataset /data/ShareGPT_V3_filtered_1000.json \
        --host tgi \
        --port 80 \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --out-filename /results/llama-7b_gpu0_rr025.txt \
        --request-rate 0.25
```

# requirements.txt
```
argparse
asyncio
aiohttp
numpy
tqdm
```