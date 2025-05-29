gpu:
	nvcc -lboost_program_options -o task8 task8.cu
	./task8 --size=1024 --accuracy=0.000001 --max_iterations=1000000
profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./task8 --size=128 --accuracy=0.0001 --max_iterations=50