Project for HRI and RBC (EAI 2, Sapienza University of Rome). Using Pepper robot from SoftBank Robotics.

# Installation

## Pre-requisites

- Docker
- Docker Compose

## Download necessary software

- HRI Software: in your `$HOME` directory, run:
    ```bash
    git clone https://bitbucket.org/iocchi/hri_software.git
    ```

- Pepper Tools:
    ```bash
    mkdir -p $HOME/src/Pepper
    cd $HOME/src/Pepper
    git clone https://bitbucket.org/mtlazaro/pepper_tools.git
    ```

- MODIM:

    ```bash
    mkdir -p $HOME/src/Pepper
    cd $HOME/src/Pepper
    git clone https://bitbucket.org/mtlazaro/modim.git
    ```

## Actual project files

Create a `playground` folder in the $HOME directory and clone this repository there:
```bash
mkdir -p $HOME/playground
git clone https://github.com/cogsp/campus-pepper.git
```

## Build the Docker image

Then build the docker image by running:
```bash
cd $HOME/hri_software/docker/
./build.bash
```

# Run

Now you can run the docker container with
```
cd hri_software/docker
./run.bash
```
This basically wraps `docker compose dc_x11.yml`, that start the service `pepperhri`, mounting playground, src/Pepper/pepper_tools, and src/Pepper/modim in the container as volume.

Now you can access the container with
```bash
docker exec -it pepperhri bash
```
Access the container and install requirements with:
```bash
cd playground/campus-pepper
pip install -r requirements.txt
```


## NAOqi
On a terminal, access the container and run:
```bash
cd /opt/Aldebaran/naoqi-sdk-2.5.7.1-linux64
./naoqi
```


## Choregraphe
On another terminal, access the container and run:
```bash
cd /opt/Aldebaran/choregraphe-suite-2.5.10.7-linux64
./choregraphe
```
You may need to enter the licence key.

## Animation
Load the animation for the Pepper Robot in Choregraphe by going to **File > Import Content > Folder** and choose the proper directory.

## Tablet
On the local machine (no access to the container) run:
```bash
cd $HOME/hri_software/docker
./run_ngnix.bash $HOME/playground/campus-pepper/tablet
```
Now, on the local machine, access the browser tablet interface by opening:
```bash
$HOME/playground/campus-pepper/tablet/index.html
```
For instance, using **Live Server extension in VSCode**.

## MODIM
Access the container and run:
```bash
cd src/Pepper/modim/src/GUI
export PEPPER_PORT=<your_pepper_robot_port>
python ws_server.py -robot pepper
```

## Launch the main script
Access the container and run:
```bash
cd playground/campus-pepper
python main.py --pport <your_pepper_robot_port>
```

## Notes
The docker container handles all the versioning, but if you want to run the code by yourself, note that the codebase uses python2.x
