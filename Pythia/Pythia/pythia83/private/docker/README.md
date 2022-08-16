# Overview

A Docker repository is available for Pythia 8 at
[cloud.docker.com/u/pythia8](https://cloud.docker.com/u/pythia8). Currently
there are three repositories.

* dev: intended for all the development and testing containers that we
  need. Currently this repository holds a number of containers used
  for testing various methods for generating Python bindings, all
  based on the `fedora:30` image.

* releases: this repository will hold containers with publicly
  available Pythia 8 releases. Exactly how this should be structured
  is up for discussion.

* tutorials: the containers used for specific tutorials, e.g. MCNet
  summer schools, can be stored here.  Anyone who would like access to
  the Docker repository can send Phil a request to be added as a
  developer.

# Images

There currently are recipes for six images.

* `binder.dev` - intended as a light-weight container for building the
  Python bindings using `binder`.

* `test.dev` - a comprehensive testing container with all possible
  external dependencies. Note, this container is large.

* `rivet-stack.dev` - sets up the Rivet stack with its dependencies,
  based on ubuntu:latest, ready to install Pythia on top.

* `cern-centos7` - sets up an image equivalent to CERN CentOS 7 on
  LXPlus, ready to install Pythia on top.

* `cern-centos8` - sets up an image equivalent to CERN CentOS 8 on
  LXPlus, ready to install Pythia on top.

* `ubuntu` - sets up an image based on ubuntu:latest, with dependencies
  for Pythia installed. 

To build these containers using the recipes in the directories,
perform the following.
```bash
docker build ./ -f <recipe>.dev
docker tag <hash> pythia8/<repo>:<recipe>
docker push pythia8/<repo>:<recipe>
```
Here, `<recipe>` is the container to build, e.g. `test`, and `<repo>`
is the from above, e.g. `dev`. Finally, `hash` is the hash of the
container as obtained from `docker image ls`.

# Testing Image

A testing image with all external dependencies, based on Fedora 30, is
available on DockerHub. To use the image to build Pythia, do the
following.
```bash
docker run -i -t -v "$PWD:$PWD" -w $PWD -u `id -u` --cap-add=SYS_PTRACE --rm pythia8/dev:test bash
```
See the [private scripts](../README.md) documentation on how to use
this to run tests.

# Cheatsheet for Docker

| Docker command | action |
| ------ | ------ |
| `docker image ls` | List all available Docker images. |
| `docker image rm <image ID>` | Remove a Docker image. |
| `docker ps` | List active Docker containers. |
| `docker ps -a` | List all Docker containers. |
| `docker stop <container ID>` | Stop a Docker container. |
| `docker rm <container ID>` | Remove a Docker container. |
| `docker stop $(docker ps -a -q)` | Stop all Docker containers. |
| `docker rm $(docker ps -a -q)` | Remove all Docker containers. |
| `docker run -it -v <local directory path>:<docker docker directory path> <group>/<repo>:<image> /bin/bash` | Run a Docker `<image>`, which is pulled from Docker hub for a specified `<group>` and `<repo>` if not available locally. The `-i` option keeps the container running, even if not attached, and the `-t` option allows graphics via a TTY interface. Finall, the `-v` option allows one to mount a local folder within the container. |
| `docker exec -it <container ID> /bin/bash` | Connect to an already running container. |

