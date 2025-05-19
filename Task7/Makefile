all: gpu profile

gpu:
        pgc++ -o task7 -lboost_program_options -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include -I/usr/local/> -acc=gpu -Minfo=all  tas>
        ./task7 --size=128 --accuracy=0.000001 --max_iterations=1000000

profile:
        nsys profile --trace=nvtx,cuda,openacc --stats=true ./task7 --size=256 >

