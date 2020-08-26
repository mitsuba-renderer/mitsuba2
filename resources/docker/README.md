# Docker files for Mitsuba 2

Here are Docker files for running Mitsuba 2 either CPU only, or also with GPU support.

**Attention!** For GPU and CUDA/OptiX to work, you need to have new Nvidia drivers with builtin OptiX libraries, and also updated Docker and Nvidia Docker that provide proper mounting of OptiX libraries. You do **not** need to install OptiX SDK, it is **not** necessary for Mitsuba 2! Only the updated Nvidia drivers and Docker are needed.

## Building the images

Building the CPU only version:

```
docker build -t mitsuba2:cpu -f linux-cpu.Dockerfile .
```

Building the CPU/GPU version:

```
docker build -t mitsuba2:gpu -f linux-gpu.Dockerfile .
```

The images will be tagged as `mitsuba2:cpu` or `mitsuba2:gpu`.

## Starting the containers

Running the CPU image:

```
cd /path/to/mitsuba2
docker run -it -p 45678:8888 -v $(pwd):/mitsuba2 -u $(id -u):$(id -g) mitsuba2:cpu bash
```

This will mount the Mitsuba 2 root directory to `/mitsuba2` inside the container and map port `8888` (used for Jupyter) from the container to port `45678` in the host machine.

Running the GPU image:

```
cd /path/to/mitsuba2
docker run --runtime=nvidia --gpus all -it -p 45678:8888 -v $(pwd):/mitsuba2 -u $(id -u):$(id -g) mitsuba2:gpu bash
```

This will do the same as above, and also mount all available GPUs to the container.

## Compiling Mitsuba 2

Works the same as described in Mitsuba 2 documentation:

```
cd /mitsuba2
mkdir build
cd build
cmake -GNinja ..
ninja
```

## Running Mitsuba 2

```
cd /mitsuba2
source setpath.sh
mitsuba2 --help
```

## Running Jupyter

```
cd /mitsuba2
source setpath.sh
jupyter notebook --port 8888 --ip 0.0.0.0
```

And then you can connect to the Jupyter notebook via `localhost:45678` in your web browser.