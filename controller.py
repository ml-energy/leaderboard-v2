
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
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
import json

from logger import setup_logger
logger = setup_logger("Controller_instant.log")
user_logger = setup_logger("User.log")

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
        # TODO: add user state expiration

        self.user_state = defaultdict(dict)
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

    def receive_request_stream_energy(self,  model_name, prompt):
        if model_name not in self.model_dest:
            return None
        worker_addr, worker_port = random.choice(list(self.model_dest[model_name]))

        url = f'http://{worker_addr}:{worker_port}/generate_stream'
        data = {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': self.args.max_len,
            }
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=data, headers=headers)
        json_string = response.text.replace('data:', '')
        json_strings = json_string.strip().split('\n')
        for json_string in json_strings:
            try:
                data = json.loads(json_string)
                # Access the specific fields from the parsed JSON object
                token_text = data['token']['text']
                token_energy = data['token']['energy']
                yield token_text, token_energy
            except json.JSONDecodeError:
                print("Error parsing JSON:", json_string)


    def receive_request_stream(self, model_name, prompt, user_id):
        if model_name not in self.model_dest:
            return None
        worker_addr, worker_port = random.choice(list(self.model_dest[model_name]))
        if not worker_addr or not worker_port:
            yield None
        url = f'http://{worker_addr}:{worker_port}'
        client = Client(url)
        text = ""
        self.user_state[user_id]['prompt'].append(prompt)
        model_id = self.user_state[user_id]['model'].index(model_name)
        for response in client.generate_stream(prompt, max_new_tokens=args.max_len):
            if not response.token.special:
                text += response.token.text
                print(response.token)
                self.user_state[user_id]['energy'][model_id] += response.token.energy
                yield json.dumps(response.token.text).encode() + b"\0"

        logger.info(f"User {user_id} request {prompt} from {model_name} "
                    f"with energy {self.user_state[user_id]['energy'][model_id]}. "
                    f"with response {text} ")
        if user_id in self.user_state:
            self.user_state[user_id]['response'][model_id].append(text)
        # yield text

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
                # TODO: restart worker
                self.deactivate_worker(worker_name)

    def deactivate_worker(self, worker_name: str):
        if worker_name[0] in self.model_dest:
            self.model_dest[worker_name[0]].remove(worker_name[1:])

    def heart_beat_controller(self):
        while True:
            self.check_health()
            time.sleep(args.heart_beat_interval)

    # TODO: redis user server
    def random_assign_models(self, user_id):
        if user_id not in self.user_state:
            self.user_state[user_id]['energy'] = [0, 0]
            self.user_state[user_id]['response'] = [[], []]
            self.user_state[user_id]['prompt'] = []
            self.user_state[user_id]['model'] = random.sample(self.model_dest.keys(), min(2, len(self.model_dest.keys())))

    def remove_user(self, user_id):
        if user_id in self.user_state:
            del self.user_state[user_id]

app = FastAPI()

@app.post("/request")
async def request(request: Request):
    # import asyncio
    data = await request.json()
    prompt = data["prompt"]
    index = data["index"]
    user_id = request.headers.get("X-User-ID")
    controller.random_assign_models(user_id)
    model_name = controller.user_state[user_id]['model'][index]

    return StreamingResponse( controller.receive_request_stream(model_name, prompt, user_id))


@app.post("/get_models")
async def get_models():
    return controller.get_models()

@app.post("/get_nlp_voting")
async def get_nlp_voting(request: Request):
    user_id = request.headers.get("X-User-ID")
    data = await request.json()
    nlp_voting = data["nlp_voting"]
    logger.info(f"User {user_id} return nlp_voting {nlp_voting} between models {controller.user_state[user_id]['model']}")
    controller.user_state[user_id]['nlp_voting'] = nlp_voting
    return controller.user_state[user_id]['model'] + controller.user_state[user_id]['energy']

@app.post("/get_energy_voting")
async def get_energy_voting(request: Request):
    user_id = request.headers.get("X-User-ID")
    data = await request.json()
    energy_voting = data["energy_voting"]
    controller.user_state[user_id]['energy_voting'] = energy_voting
    user_logger.info(controller.user_state[user_id])

@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(data["model_name"], data["ip_address"], data["port"], data["queue_length"])
    return {"exist": exist}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--deploy_yml", type=str, default="deployment.yaml")
    parser.add_argument("--network_name", type=str, default="mynetwork")
    parser.add_argument("--heart_beat_interval", type=int, default=45)
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["random", "shortest_queue"],
        default="random",
    )
    args = parser.parse_args()

    controller = Controller(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


