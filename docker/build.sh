VER=1.0
docker build -f docker/Dockerfile --progress=plain . --network=host -t graspdatagen:$VER -t graspdatagen:latest
