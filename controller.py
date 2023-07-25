
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
import argparse
import requests
import random
import threading
import time
import yaml

from collections import defaultdict
import logging
logger = logging.getLogger('Controller')
from text_generation import Client

class WorkerInfo:
    def __init__(self):
        self.last_heart_beat = -1

    def check_heartbeat(self, time):
        self.last_heart_beat = time

class Controller:
    def __init__(self, args):
        self.model_dest = defaultdict(set)
        self.args = args
        self.worker_info = {}
        self.network_name = args.network_name
        self.deploy_workers(args.deploy_yml)

        self.heart_beat_thread = threading.Thread(
            target=self.heart_beat_controller,
        )
        self.heart_beat_thread.start()

    def deploy_workers(self, deploy_yml):
        # Load YAML data from the file
        with open(deploy_yml, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        instance_id = 0
        for model in data['models']:
            model_name = model['name']
            for instance in model['instances']:
                ip_address = instance['ip_address']
                port = instance['docker_port']
                self.worker_info[model_name, ip_address, port] = WorkerInfo()
                instance_id += 1

    def receive_request(self,  model_name, prompt):
        print(f"Received request for {model_name} with prompt {prompt}")
        # random pick
        worker_addr, worker_port = random.choice(list(self.model_dest[model_name]))

        url = f'http://{worker_addr}:{worker_port}/generate'
        data = {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': self.args.max_len,
            }
        }
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, json=data, headers=headers)
        print(response.json()['generated_text'])
        return response.json()['generated_text']

    def receive_request_stream(self, model_name, prompt):
        worker_addr, worker_port = random.choice(list(self.model_dest[model_name]))
        url = f'http://{worker_addr}:{worker_port}'
        client = Client(url)
        text = ""
        for response in client.generate_stream(prompt, max_new_tokens=args.max_len):
            if not response.token.special:
                text += response.token.text
                yield response.token.text
        print(text)

    def get_models(self):
        return list(self.model_dest.keys())

    def check_health(self):
        for worker_name, w_info in self.worker_info.items():
            model_name, ip_address, port = worker_name
            url = f'http://{ip_address}:{port}/health'
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    self.deactivate_worker(worker_name)
                else:
                    w_info.check_heartbeat(time.time())
                    if worker_name[0] not in self.model_dest[model_name]:
                        self.model_dest[model_name].add((ip_address, port))
                        print(f"Registered worker {model_name} at {ip_address}:{port}")
            except:
                self.deactivate_worker(worker_name)

    def deactivate_worker(self, worker_name: str):
        self.worker_info[worker_name].last_heart_beat = -1
        if worker_name[0] in self.model_dest:
            self.model_dest[worker_name[0]].remove(worker_name[1:])

    def heart_beat_controller(self):
        while True:
            self.check_health()
            time.sleep(args.heart_beat_interval)


app = FastAPI()

@app.post("/request")
async def request(request: Request):
    data = await request.json()
    model_name = data["model_name"]
    prompt = data["prompt"]
    return controller.receive_request_stream(model_name, prompt)

@app.post("/get_models")
async def get_models():
    return controller.get_models()

@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(data["model_name"], data["ip_address"], data["port"], data["queue_length"])
    return {"exist": exist}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--deploy_yml", type=str, default="deployment.yaml")
    parser.add_argument("--network_name", type=str, default="mynetwork")
    parser.add_argument("--heart_beat_interval", type=int, default=60)
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["random", "shortest_queue"],
        default="random",
    )
    args = parser.parse_args()

    controller = Controller(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


