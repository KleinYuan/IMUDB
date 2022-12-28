clean:
	docker rm -f $$(docker ps -qa)

clean-images:
	docker rmi $$(docker images -f "dangling=true" -q)

build:
	docker build -t imudb-docker .

run:
	sudo docker run -it \
        --runtime=nvidia \
        --name="imudb-experiment-gpu" \
        --net=host \
        --privileged=true \
        --ipc=host \
        --memory="20g" \
        --memory-swap="20g" \
        -v ${PWD}:/root/imudb \
        -v ${PWD}/EuRoC:/root/EuRoC \
        -v ${PWD}/TUMVI:/root/TUMVI \
      	imudb-docker bash

train-euroc-limu-bert:
	export PYTHONPATH='.' && python experiments/train_euroc_limu_bert.py

train-euroc-imudb:
	export PYTHONPATH='.' && python experiments/train_euroc_imudb.py

train-tumvi-imudb:
	export PYTHONPATH='.' && python experiments/train_tumvi_imudb.py