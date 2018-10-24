#include "userFitness.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <stdio.h>

__device__ float fitnessOfChromosome_TSPGA(int size, short* chromosome, int startPoint, float* coord) {
	float nilai = 0;
	int i_d = startPoint;
	for (int j = 1; j < size; j++) {
		int xDist = coord[chromosome[i_d + j] * 2] - coord[chromosome[i_d + (j - 1)] * 2];
		int yDist = coord[(chromosome[i_d + j] * 2) + 1] - coord[(chromosome[i_d + (j - 1)] * 2) + 1];
		nilai += (sqrtf((xDist*xDist) + (yDist*yDist)));
	}
	int xDist = coord[chromosome[i_d] * 2] - coord[chromosome[i_d + (size - 1)] * 2];
	int yDist = coord[(chromosome[i_d] * 2) + 1] - coord[(chromosome[i_d + (size - 1)] * 2) + 1];
	nilai += (sqrtf((xDist*xDist) + (yDist*yDist)));
	return 99999.0f - nilai;
};

__global__ void fitnessCheckGPU_TSPGA(int size, short* chromosome, float* fitness, float* coord) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	fitness[i] = fitnessOfChromosome_TSPGA(size, chromosome, i * size, coord);
}

void callFitnessCheckGPU_TSPGA(int size, short* chromosome, float* fitness, long chromosomeAmount, float* coord) {
	fitnessCheckGPU_TSPGA << < 1, chromosomeAmount >> >(size, chromosome, fitness, coord);
}
