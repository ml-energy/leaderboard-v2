# Instructions to set up the serving system  

## Install


## Setup container network
```
docker network create mynetwork
```


## Start workers
Should read from deployment.yaml. Here we just start one worker manually.
```commandline
model=bigscience/bloom-560m
num_shard=1
volume=$PWD/data 
docker_port=8000
port=8001
 
docker run  --name=worker0 --network=mynetwork --gpus 1 --shm-size 1g -p $port:$docker_port -v $volume:/data tgi:origin --model-id $model --num-shard $num_shard --port $docker_port 
```

```commandline
model=tiiuae/falcon-7b
num_shard=1
volume=$PWD/data 
docker_port=8000
port=8002
 
docker run  --name=worker1 --network=mynetwork --gpus 1 --shm-size 1g -p $port:$docker_port -v $volume:/data tgi:origin --model-id $model --num-shard $num_shard --port $docker_port 
```

More models is coming soon.


## Start controller and web server
```commandline
port=8000
docker_port=8000
docker run --name=controller -v $HOME/leaderboard:/code  --network=mynetwork -p 8000:8000 -it  tgi:origin
```

```commandline
python controller.py --host 0.0.0.0
python app.py --share
```

The controller will check the live workers every min.

## TODO
- [ ] Logging
- [ ] Conversation context





