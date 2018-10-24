#include "userFitness.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

__device__ float fitnessOfChromosome_SudokuGA(int* chromosome, int startPoint) {
	float nilai = 0.0f;
	int i_d = startPoint;
	int* chromosomeTemp = chromosome;
	for (int jj = 0; jj < 9; jj++)
	{
		bool baris = true, kolom = true, kotak = true;
		bool angkaBaris[9], angkaKolom[9], angkaKotak[9];
		for (int kkk = 0; kkk < 9; kkk++)
		{
			angkaBaris[kkk] = angkaKolom[kkk] = angkaKotak[kkk] = false;
		}
		for (int kk = 0; kk < 9; kk++)
		{
			int baris_i = i_d + (jj * 9) + kk;
			int kolom_i = i_d + (kk * 9) + jj;
			int kotak_i = i_d + ((3 * jj) + (18 * int(jj / 3))) + (kk + (int(kk / 3) * 6));
			int kromosomBaris = chromosomeTemp[baris_i] % 9;
			int kromosomKolom = chromosomeTemp[kolom_i] % 9;
			int kromosomKotak = chromosomeTemp[kotak_i] % 9;
			if (angkaBaris[kromosomBaris]) baris = false;
			if (angkaKolom[kromosomKolom]) kolom = false;
			if (angkaKotak[kromosomKotak]) kotak = false;
			angkaBaris[kromosomBaris] = true;
			angkaKolom[kromosomKolom] = true;
			angkaKotak[kromosomKotak] = true;
		}
		if (baris) nilai += 1.0f;
		if (kolom) nilai += 1.0f;
		if (kotak) nilai += 1.0f;
	}
	return nilai;
};

__global__ void fitnessCheckGPU_SudokuGA(int size, int* chromosome, float* fitness) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	fitness[i] = fitnessOfChromosome_SudokuGA(chromosome, i * size);
}

void callFitnessCheckGPU_SudokuGA(int size, int* chromosome, float* fitness, long chromosomeAmount) {
	fitnessCheckGPU_SudokuGA << < 1, chromosomeAmount >> >(81, chromosome, fitness);
}