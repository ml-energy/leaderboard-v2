import enum
import json
import time
import yaml
import heapq
import random
import argparse
import requests
import threading
from collections import defaultdict

import uvicorn
from text_generation import Client
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse

from spitfight.common import (
    COLOSSEUM_PROMPT_ROUTE,
    COLOSSEUM_RESP_VOTE_ROUTE,
    COLOSSEUM_ENERGY_VOTE_ROUTE
)
from spitfight.log import setup_logger

controller_logger = setup_logger("Controller_instant.log")
user_logger = setup_logger("User.log")


class State(enum.Enum):
    """State of the Colosseum."""

    WAITING_PROMPT = "waiting for the user prompt"
    WAITING_RESPONSE_VOTE = "waiting for the user to vote for model responses"
    WAITING_ENERGY_VOTE = "waiting for the user to vote for model energy"

class BoundedExpiringDict:
    def __init__(self, expiration_time: int):
        self.data_dict = defaultdict(dict)
        self.timestamp_heap = []
        self.timeout = expiration_time

        # Without this, the controller is vulnerable to "user flood attacks,"
        # where someone can create a bunch of users by polling /request before
        # self.timeout expires and blow up memory.
        self.max_size = 10000

    def __getitem__(self, key):
        # begin counting timeout from the first access
        if key not in self.data_dict:
            heapq.heappush(self.timestamp_heap, (time.time(), key))
        return self.data_dict[key]

    def __setitem__(self, key, value):
        if len(self.data_dict) >= self.max_size:
            self.cleanup()

        if key not in self.data_dict:
            heapq.heappush(self.timestamp_heap, (time.time(), key))
        self.data_dict[key] = value

    def __delitem__(self, user_id):
        del self.data_dict[user_id]

    def __contains__(self, user_id):
        return user_id in self.data_dict

    def __len__(self):
        return len(self.data_dict)

    def cleanup(self):
        threshold = time.time() - self.timeout
        # After the while loop, the dictionary will be smaller than max_size
        # and all users will have been accessed within the timeout.
        while (self.timestamp_heap and self.timestamp_heap[0][0] < threshold) or len(self.data_dict) > self.max_size:
            _, user_id = heapq.heappop(self.timestamp_heap)
            del self.data_dict[user_id]
            controller_logger.debug("User %s evicted", user_id)


class WorkerInfo:
    def __init__(self):
        self.last_heart_beat = -1


class Controller:
    def __init__(self, args):
        self.model_dest = defaultdict(set)
        self.args = args
        self.network_name = args.network_name
        self.max_user_state = args.max_user_state

        self.user_state = BoundedExpiringDict(args.user_state_expiration_time)
        self.heart_beat_thread = threading.Thread(
            target=self.heart_beat_controller,
        )
        self.heart_beat_thread.start()

    def deploy_workers(self, deploy_yml):
        # Load YAML data from the file
        with open(deploy_yml, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        worker_info = {}
        instance_id = 0
        for model in data['models']:
            model_name = model['name']
            for instance in model['instances']:
                ip_address = instance['ip_address']
                port = instance['docker_port']
                worker_info[model_name, ip_address, port] = WorkerInfo()
                instance_id += 1
        return worker_info

    def receive_request_stream(self, model_name, prompt, user_id):
        if model_name not in self.model_dest:
            return None
        worker_addr, worker_port = random.choice(list(self.model_dest[model_name]))
        if not worker_addr or not worker_port:
            yield None
        url = f'http://{worker_addr}:{worker_port}'
        client = Client(url)
        text = ""
        self.user_state[user_id]['prompt'] = prompt
        model_id = self.user_state[user_id]['model'].index(model_name)
        for response in client.generate_stream(prompt, max_new_tokens=args.max_len):
            if not response.token.special:
                text += response.token.text
                self.user_state[user_id]['energy'][model_id] += response.token.energy
                yield json.dumps(response.token.text).encode() + b"\0"

        controller_logger.info(f"User {user_id} request {prompt} from {model_name} "
                    f"with energy {self.user_state[user_id]['energy'][model_id]}. "
                    f"with response {text} ")
        if user_id in self.user_state:
            self.user_state[user_id]['response'][model_id] = text
        # yield text

    # def get_models(self):
    #     return list(self.model_dest.keys())

    def check_health(self):
        worker_info = self.deploy_workers(self.args.deploy_yml)
        for worker_name, w_info in worker_info.items():
            model_name, ip_address, port = worker_name
            url = f'http://{ip_address}:{port}/health'
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    self.deactivate_worker(worker_name)
                else:
                    if worker_name[0] not in self.model_dest[model_name]:
                        self.model_dest[model_name].add((ip_address, port))
                        print(f"Registered worker {model_name} at {ip_address}:{port}")
            except:
                # TODO: restart worker automatically
                self.deactivate_worker(worker_name)

    def deactivate_worker(self, worker_name: str):
        if worker_name[0] in self.model_dest:
            self.model_dest[worker_name[0]].remove(worker_name[1:])

    def heart_beat_controller(self):
        while True:
            self.check_health()
            time.sleep(args.heart_beat_interval)
            if len(self.user_state) > self.max_user_state:
                self.user_state.cleanup(self.user_state_expiration_time)
            time.sleep(args.heart_beat_interval)

    # redis user server?
    def random_assign_models(self, user_id):
        if user_id not in self.user_state or self.user_state[user_id]['model'] is None:
            self.user_state[user_id]['model'] = random.sample(list(self.model_dest.keys()), min(2, len(self.model_dest.keys())))
        self.user_state[user_id]['response'] = ["", ""]
        self.user_state[user_id]['prompt'] = ""
        self.user_state[user_id]['energy'] = [0, 0]
        self.user_state[user_id]['nlp_voting'] = -1
        self.user_state[user_id]['energy_voting'] = -1

    def remove_user(self, user_id):
        if user_id in self.user_state:
            del self.user_state[user_id]

app = FastAPI()

@app.post(COLOSSEUM_PROMPT_ROUTE)
async def request(request: Request):
    data = await request.json()
    system_prompt = "A chat between a human user and an assistant, who gives helpful and polite answers to the user's questions. "
    prompt = system_prompt + data["prompt"]
    index = data["index"]
    user_id = request.headers.get("X-User-ID")
    controller.random_assign_models(user_id)
    model_name = controller.user_state[user_id]['model'][index]

    return StreamingResponse(controller.receive_request_stream(model_name, prompt, user_id))

# @app.post("/get_models")
# async def get_models():
#     return controller.get_models()

@app.post(COLOSSEUM_RESP_VOTE_ROUTE)
async def get_nlp_voting(request: Request):
    user_id = request.headers.get("X-User-ID")
    data = await request.json()
    nlp_voting = data["nlp_voting"]
    controller_logger.info(f"User {user_id} return nlp_voting {nlp_voting} between models {controller.user_state[user_id]['model']}")
    controller.user_state[user_id]['nlp_voting'] = nlp_voting
    if user_id in controller.user_state:
        model_names = controller.user_state[user_id]['model']
        controller.user_state[user_id]['model'] = None
        return model_names + controller.user_state[user_id]['energy']
    else:
        return 'TIMEOUT' + 0

@app.post(COLOSSEUM_ENERGY_VOTE_ROUTE)
async def get_energy_voting(request: Request):
    user_id = request.headers.get("X-User-ID")
    data = await request.json()
    energy_voting = data["energy_voting"]
    if user_id in controller.user_state:
        controller.user_state[user_id]['energy_voting'] = energy_voting
        user_logger.info(controller.user_state[user_id])
        controller.remove_user(user_id)
    else:
        controller_logger.info(f"User {user_id} expired energy voting {energy_voting}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--deploy_yml", type=str, default="deployment.yaml")
    parser.add_argument("--network_name", type=str, default="mynetwork")
    parser.add_argument("--heart_beat_interval", type=int, default=300)
    parser.add_argument("--max_user_state", type=int, default=10000)
    parser.add_argument("--user_state_expiration_time", type=int, default=600)
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["random", "shortest_queue"],
        default="random",
    )
    args = parser.parse_args()

    controller = Controller(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
