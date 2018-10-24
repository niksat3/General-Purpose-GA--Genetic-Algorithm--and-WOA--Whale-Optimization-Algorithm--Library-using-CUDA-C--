#include "userFitness.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

__device__ float fitnessOfSearchAgent_SudokuWOA(float* searchAgent, int startPoint) {
	float nilai = 0.0f;
	int i_d = startPoint;
	float* searchAgentTemp = searchAgent;
	for (int j = 0; j < 9; j++)
	{
		bool baris = true, kolom = true, kotak = true;
		bool angkaBaris[9], angkaKolom[9], angkaKotak[9];
		for (int k = 0; k < 9; k++)
		{
			angkaBaris[k] = angkaKolom[k] = angkaKotak[k] = false;
		}
		for (int k = 0; k < 9; k++)
		{
			int baris_i = i_d + (j * 9) + k;
			int kolom_i = i_d + (k * 9) + j;
			int kotak_i = i_d + ((3 * j) + (18 * int(j / 3))) + (k + (int(k / 3) * 6));
			if (angkaBaris[(abs((int)searchAgentTemp[baris_i]) % 9)]) baris = false;
			if (angkaKolom[(abs((int)searchAgentTemp[kolom_i]) % 9)]) kolom = false;
			if (angkaKotak[(abs((int)searchAgentTemp[kotak_i]) % 9)]) kotak = false;
			angkaBaris[(abs((int)searchAgentTemp[baris_i]) % 9)] = true;
			angkaKolom[(abs((int)searchAgentTemp[kolom_i]) % 9)] = true;
			angkaKotak[(abs((int)searchAgentTemp[kotak_i]) % 9)] = true;
		}
		if (baris) nilai += 1.0f;
		if (kolom) nilai += 1.0f;
		if (kotak) nilai += 1.0f;
	}
	return nilai;
};

__global__ void fitnessCheckGPU_SudokuWOA(int size, float* searchAgent, float* fitness)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	fitness[i] = fitnessOfSearchAgent_SudokuWOA(searchAgent, i * size);
}

void callFitnessCheckGPU_SudokuWOA(int size, float* searchAgent, float* fitness, long chromosomeAmount) {
	fitnessCheckGPU_SudokuWOA << < 1, chromosomeAmount >> >(size, searchAgent, fitness);
}