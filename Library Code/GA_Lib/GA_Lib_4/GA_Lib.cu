/*-------------------------------------------------------------------------------------------------*/
/*                                                 RULES:										   */
/*							1. chromosomePerGeneration can't be odd number						   */
/*-------------------------------------------------------------------------------------------------*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GA_Lib.h"

#include <curand_kernel.h>
#include <chrono>
#include <math.h>
#include <map>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <Windows.h>

using namespace std;
using namespace std::chrono;

#pragma region GPU method

__global__ void assignSequencedValue(int* data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	data[i] = i;
};

template <typename T>
__global__ void assignChromosome(int* indexData, T* chromosome, int size) {
	extern __shared__ float sFloat[];
	T* tempChromosome = (T*)sFloat;
	int i = threadIdx.x;
	int index = indexData[i];

	for (int j = 0; j < size; j++) {
		tempChromosome[i * size + j] = chromosome[index * size + j];
	}
	__syncthreads();
	for (int j = 0; j < size; j++) {
		chromosome[i * size + j] = tempChromosome[i * size + j];
	}

};

template <typename T>
__global__ void doRandomResetting(float mutationRate, T* chromosome, bool* genChangedOption, T* randChromosome) {
	int size = blockDim.x;
	int i = blockIdx.x + gridDim.x;
	int j = threadIdx.x;

	if (j == 0) {
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);
		float rand = curand_uniform(&state);
		if (rand < mutationRate) {
			long randIndex = (long)(curand_uniform(&state) * (size - 1));
			if (genChangedOption[randIndex]) {
				chromosome[i * size + randIndex] = randChromosome[randIndex];
			}
		}
	}
};

template <typename T>
__global__ void doSwapMutation(float mutationRate, T* chromosome, bool* genChangedOption) {
	int size = blockDim.x;
	int i = blockIdx.x + gridDim.x;
	int j = threadIdx.x;

	if (j == 0) {
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);

		float rand = curand_uniform(&state);
		if (rand < mutationRate) {
			long index1, index2;
			index1 = (long)(curand_uniform(&state) * (size - 1));
			index2 = (long)(curand_uniform(&state) * (size - 1));
			if (genChangedOption[index1] && genChangedOption[index2]) {
				T temp = chromosome[i * size + index1];
				chromosome[i * size + index1] = chromosome[i * size + index2];
				chromosome[i * size + index2] = temp;
			}
		}
	}
};

template <typename T>
__global__ void doScrambleMutation(float mutationRate, T* chromosome, bool* genChangedOption) {
	int size = blockDim.x;
	extern __shared__ float sFloat[];
	__shared__ float rand;
	int i = blockIdx.x + gridDim.x;
	int j = threadIdx.x;

	curandState state;
	curand_init((unsigned long)clock() + i + j, 0, 0, &state);
	if (j == 0) rand = curand_uniform(&state);
	__syncthreads();

	if (rand < mutationRate) {
		T* newChromosome = (T*)sFloat;
		__shared__ long startPoint;
		__shared__ long stopPoint;

		if (j == 0) {
			startPoint = (long)(curand_uniform(&state) * (size - 1));
			stopPoint = (long)(curand_uniform(&state) * (size - 1));
			if (startPoint > stopPoint) {
				long tempPoint = startPoint;
				startPoint = stopPoint;
				stopPoint = tempPoint;
			}
		}
		newChromosome[j] = chromosome[i * size + j];
		__syncthreads();

		long index1 = (long)(curand_uniform(&state) * (stopPoint - startPoint)) + startPoint;
		long index2 = (long)(curand_uniform(&state) * (stopPoint - startPoint)) + startPoint;
		for (int k = 0; k < size; k++) {
			if (genChangedOption[index1] && genChangedOption[index2] && k == j) {
				T temp = newChromosome[index1];
				newChromosome[index1] = newChromosome[index2];
				newChromosome[index2] = temp;
			}
			__syncthreads();
		}
		__syncthreads();
		chromosome[i * size + j] = newChromosome[j];
	}
};

template <typename T>
__global__ void doInversionMutation(float mutationRate, T* chromosome, bool* genChangedOption) {
	int size = blockDim.x;
	extern __shared__ float sFloat[];
	__shared__ float rand;
	int i = blockIdx.x + gridDim.x;
	int j = threadIdx.x;

	curandState state;
	curand_init((unsigned long)clock() + i + j, 0, 0, &state);
	if (j == 0) rand = curand_uniform(&state);
	__syncthreads();

	if (rand < mutationRate)
	{
		T* newChromosome = (T*)sFloat;
		__shared__ long startPoint;
		__shared__ long stopPoint;
		if (j == 0) {
			startPoint = (long)(curand_uniform(&state) * (size - 1));
			stopPoint = (long)(curand_uniform(&state) * (size - 1));
			if (startPoint > stopPoint) {
				long tempPoint = startPoint;
				startPoint = stopPoint;
				stopPoint = tempPoint;
			}
		}
		newChromosome[j] = chromosome[i * size + j];
		__syncthreads();
		T temp;
		if (j >= startPoint && j <= stopPoint) temp = newChromosome[stopPoint - j + startPoint];
		__syncthreads();
		if (j >= startPoint && j <= stopPoint) newChromosome[j] = temp;
		__syncthreads();
		chromosome[i * size + j] = newChromosome[j];
	}
};

//Look if better do search with binary search/linear search
__global__ void doSearch(float* fitnessTemp, float searchedFitness, long chromosomePerGeneration, long* answer) {
	//Break if 1 kernel found the answer
	__shared__ bool breakKernelLoop;
	breakKernelLoop = false;
	__syncthreads();
	//Stored in register kernel memory for faster access
	long chromosomePerGenerationR = chromosomePerGeneration;
	float searchedFitnessR = searchedFitness;
	float* fitnessTempR = fitnessTemp;
	//Random starting pivot
	int pivot = (chromosomePerGenerationR / blockDim.x) * threadIdx.x;
	//Perform Linear Search
	while (pivot < chromosomePerGenerationR - 1 && !breakKernelLoop) {
		float diff = searchedFitnessR - fitnessTempR[pivot];
		if (pivot == 0 && searchedFitnessR - fitnessTempR[1] <= 0.0f) {
			breakKernelLoop = true;
			answer[0] = pivot;
			break;
		}
		else if (diff >= 0.0f && searchedFitnessR - fitnessTempR[pivot + 1] <= 0.0f) {
			breakKernelLoop = true;
			answer[0] = pivot;
			break;
		}
		pivot++;
	}
};

__global__ void randomizeChromosome(long* tempChromosome, float* tempFitness, float* fitness, long chromosomePerGeneration) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	curandState state;
	curand_init((unsigned long)clock() + i, 0, 0, &state);
	long pivot = (long)(curand_uniform(&state) * (chromosomePerGeneration - 1));

	tempChromosome[i] = pivot;
	tempFitness[i] = fitness[pivot];
};

template <typename T>
__global__ void doOnePointCrossover(long* chromosomeParent1, long* chromosomeParent2, int chromosomePerGeneration, T* chromosome, short mutationType, float mutationRate, bool* genChangedOption) {
	int size = blockDim.x;
	__shared__ int borderOffspring;
	long chP1, chP2;
	int i = blockIdx.x;
	int j = threadIdx.x;

	if (j == 0) {
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);
		borderOffspring = (int)(curand_uniform(&state) * (size - 1));
	}

	if (i % 2 == 0) {
		chP1 = chromosomeParent1[i / 2];
		chP2 = chromosomeParent2[i / 2];
	}
	else {
		chP1 = chromosomeParent2[i / 2];
		chP2 = chromosomeParent1[i / 2];
	}
	__syncthreads();

	int indexNewChromosome = i + chromosomePerGeneration;
	T temp;
	if (j >= borderOffspring && genChangedOption[j]) temp = chromosome[chP1 * size + j];
	else temp = chromosome[chP2 * size + j];
	__syncthreads();
	chromosome[indexNewChromosome * size + j] = temp;
	__syncthreads();

};

template <typename T>
__global__ void doMultiPointCrossover(long* chromosomeParent1, long* chromosomeParent2, int chromosomePerGeneration, T* chromosome, short mutationType, float mutationRate, bool* genChangedOption) {
	int size = blockDim.x;
	__shared__ int borderOffspring1;
	__shared__ int borderOffspring2;
	long chP1, chP2;
	int i = blockIdx.x;
	int j = threadIdx.x;

	if (j == 0) {
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);
		borderOffspring1 = (int)(curand_uniform(&state) * (size - 1));
		borderOffspring2 = (int)(curand_uniform(&state) * (size - 1));
		if (borderOffspring1 > borderOffspring2) {
			int temp = borderOffspring1;
			borderOffspring1 = borderOffspring2;
			borderOffspring2 = temp;
		}
	}

	if (i % 2 == 0) {
		chP1 = chromosomeParent1[i / 2];
		chP2 = chromosomeParent2[i / 2];
	}
	else {
		chP1 = chromosomeParent2[i / 2];
		chP2 = chromosomeParent1[i / 2];
	}
	__syncthreads();

	int indexNewChromosome = i + chromosomePerGeneration;

	T temp;
	if (j >= borderOffspring1 && j <= borderOffspring2 && genChangedOption[j]) temp = chromosome[chP1 * size + j];
	else temp = chromosome[chP2 * size + j];
	__syncthreads();
	chromosome[indexNewChromosome * size + j] = temp;
	__syncthreads();

};

template <typename T>
__global__ void doUniformCrossover(long* chromosomeParent1, long* chromosomeParent2, int chromosomePerGeneration, T* chromosome, short mutationType, float mutationRate, bool* genChangedOption) {
	int size = blockDim.x;
	long chP1, chP2;
	int i = blockIdx.x;
	int j = threadIdx.x;

	if (i % 2 == 0) {
		chP1 = chromosomeParent1[i / 2];
		chP2 = chromosomeParent2[i / 2];
	}
	else {
		chP1 = chromosomeParent2[i / 2];
		chP2 = chromosomeParent1[i / 2];
	}

	int indexNewChromosome = i + chromosomePerGeneration;

	curandState state;
	curand_init((unsigned long)clock() + i + j, 0, 0, &state);
	float rand = curand_uniform(&state);

	T temp;
	if (rand > 0.5f && genChangedOption[j]) temp = chromosome[chP1 * size + j];
	else temp = chromosome[chP2 * size + j];
	__syncthreads();
	chromosome[indexNewChromosome * size + j] = temp;
	__syncthreads();

};

template <typename T>
__global__ void doPMXCrossover(long* chromosomeParent1, long* chromosomeParent2, int chromosomePerGeneration, T* chromosome, short mutationType, float mutationRate, bool* genChangedOption) {
	int size = blockDim.x;
	extern __shared__ float sFloat[];
	__shared__ int startPoint;
	__shared__ int stopPoint;
	long chP1, chP2;
	int i = blockIdx.x;
	int j = threadIdx.x;

	if (j == 0) {
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);
		startPoint = (int)(curand_uniform(&state)*(size - 1));
		stopPoint = (int)(curand_uniform(&state)*(size - 1));
		if (startPoint > stopPoint) {
			int tempPoint = startPoint;
			startPoint = stopPoint;
			stopPoint = tempPoint;
		}
	}
	__syncthreads();
	int startPointR = startPoint, stopPointR = stopPoint;
	if (i % 2 == 0) {
		chP1 = chromosomeParent1[i / 2];
		chP2 = chromosomeParent2[i / 2];
	}
	else {
		chP1 = chromosomeParent2[i / 2];
		chP2 = chromosomeParent1[i / 2];
	}

	//Map and Values (Because stdlib isn't allowed inside CUDA) and its size
	T* mapChromosome = (T*)sFloat;
	T* valuesChromosome = (T*)&mapChromosome[size];
	__shared__ int sizeMap;

	//New Child
	T* newChromosome = (T*)&valuesChromosome[size];

	T* chromosomeParent1R = (T*)&newChromosome[size];
	T* chromosomeParent2R = (T*)&chromosomeParent1R[size];
	//Boolean for PMX
	bool* canUseMap = (bool*)&chromosomeParent2R[size];

	//Copying Parent from Global Memory to Local Memory (with each thread working)
	sizeMap = 0;
	chromosomeParent1R[j] = chromosome[chP1 * size + j];
	__syncthreads();
	chromosomeParent2R[j] = chromosome[chP2 * size + j];
	__syncthreads();
	
	//Do Mapping Based on Gen Between Start and Stop Point
	for (int k = startPointR; k <= stopPointR; k++) {
		if (j == k) {
			//Filling Gen of newChromosome Between Start and Stop Points with Parent 1 (every thread between the start and stop points working)
			newChromosome[k] = chromosomeParent1R[k];

			T pointer = chromosomeParent1R[k];
			int index1 = -1, index2 = -1;
			int sizeTemp = sizeMap;
			for (int l = 0; l < sizeTemp && (index1 == -1 || index2 == -1); l++) {
				if (valuesChromosome[l] == chromosomeParent1R[k] && canUseMap[l]) index1 = l;
				if (mapChromosome[l] == chromosomeParent2R[k] && canUseMap[l]) index2 = l;
			}
			if (index1 != -1) {
				valuesChromosome[index1] = chromosomeParent2R[k];
				pointer = mapChromosome[index1];
			}
			if (index2 != -1) {
				mapChromosome[index2] = pointer;
				if (index1 != -1) canUseMap[index1] = false;
			}
			if (index1 == -1 && index2 == -1) {
				mapChromosome[sizeTemp] = chromosomeParent1R[j];
				valuesChromosome[sizeTemp] = chromosomeParent2R[j];
				canUseMap[sizeTemp] = true;
				sizeMap++;
			}
		}
		__syncthreads();
	}

	//Filling Other Gen of newChromosome Outside The Start and Stop Points with Values Based On PMX (every thread outside the start and stop points working)
	if (j < startPointR || j > stopPointR) {
		int indexFound = -1;
		int sizeTemp = sizeMap;
		for (int k = 0; k < sizeTemp && indexFound == -1; k++) {
			if (mapChromosome[k] == chromosomeParent2R[j] && canUseMap[k]) indexFound = k;
		}
		if (indexFound != -1) newChromosome[j] = valuesChromosome[indexFound];
		else newChromosome[j] = chromosomeParent2R[j];
	}
	__syncthreads();
	
	//Filling newChromosome to Its Place Inside Chromosome
	int indexNewChromosome = i + chromosomePerGeneration;
	chromosome[indexNewChromosome * size + j] = newChromosome[j];
	__syncthreads();

};

template <typename T>
__global__ void doOrder1Crossover(long* chromosomeParent1, long* chromosomeParent2, int chromosomePerGeneration, T* chromosome, short mutationType, float mutationRate, bool* genChangedOption) {
	int size = blockDim.x;
	extern __shared__ float sFloat[];
	__shared__ int startPoint;
	__shared__ int stopPoint;
	long chP1, chP2;
	int i = blockIdx.x;
	int j = threadIdx.x;

	if (j == 0) {
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);
		startPoint = (int)(curand_uniform(&state)*(size - 1));
		stopPoint = (int)(curand_uniform(&state)*(size - 1));
		if (startPoint > stopPoint) {
			int tempPoint = startPoint;
			startPoint = stopPoint;
			stopPoint = tempPoint;
		}
	}
	__syncthreads();
	int startPointR = startPoint, stopPointR = stopPoint;

	if (i % 2 == 0) {
		chP1 = chromosomeParent1[i / 2];
		chP2 = chromosomeParent2[i / 2];
	}
	else {
		chP1 = chromosomeParent2[i / 2];
		chP2 = chromosomeParent1[i / 2];
	}

	T* newChromosome = (T*)sFloat;
	T* chromosomeParent1R = (T*)&newChromosome[size];
	T* chromosomeParent2R = (T*)&chromosomeParent1R[size];

	//Using Parallel Inclusive Scan to Determine New Place of Gen
	int* indexChanged = (int*)&chromosomeParent2R[size];
	int* newIndex = (int*)&indexChanged[size];

	chromosomeParent1R[j] = chromosome[chP1 * size + j];
	chromosomeParent2R[j] = chromosome[chP2 * size + j];
	__syncthreads();

	if (j >= startPointR && j <= stopPointR) newChromosome[j] = chromosomeParent1R[j];

	//Filling IndexChanged With 1 If Gen(j) in Parent 2 Exists in newChromosome Between StartPoint and StopPoint and 0 If It's Not
	indexChanged[j] = 1;
	T parentTemp = chromosomeParent2R[j];
	for (int k = startPointR; k <= stopPointR; k++) {
		if (chromosomeParent1R[k] == parentTemp) indexChanged[j] = 0;
	}
	newIndex[j] = indexChanged[j];
	if (j == stopPointR + 1) newIndex[j] -= 1;
	__syncthreads();

	//Do Inclusive Scan (Hillis Steele Scan) Starting with -1
	int gap = 1;
	int index = (j + (size - 1 - stopPointR)) % size;
	__syncthreads();
	int loop = (int) __log2f(size - 1);
	for (int k = 0; k <= loop; k++) {
		int temp = 0;
		if (index >= gap) {
			int indexNeighbor = j - gap;
			if (indexNeighbor < 0) indexNeighbor += size;
			temp = newIndex[indexNeighbor];
		}
		__syncthreads();
		gap *= 2;
		newIndex[j] += temp;
		__syncthreads();
	}
	__syncthreads();

	if (newIndex[j] != -1) {
		int indexNeighbor = j - 1;
		if (indexNeighbor < 0) indexNeighbor += size;
		if (newIndex[indexNeighbor] != newIndex[j]) newChromosome[(stopPointR + 1 + newIndex[j]) % size] = chromosomeParent2R[j];
	}
	__syncthreads();

	int indexNewChromosome = i + chromosomePerGeneration;
	chromosome[indexNewChromosome * size + j] = newChromosome[j];
	__syncthreads();

};

template <typename T>
__global__ void generateChild(long* chromosomeParent1, long* chromosomeParent2, int chromosomePerGeneration, long startingPoint, T* chromosome, short mutationType, float mutationRate, bool* genChangedOption) {
	int size = blockDim.x;
	int i = blockIdx.x;
	int j = threadIdx.x;
	int start = startingPoint;

	int chP;
	if (i % 2 == 0) chP = chromosomeParent1[i / 2 + start];
	else chP = chromosomeParent2[i / 2 + start];

	int indexNewChromosome = i + (start * 2) + chromosomePerGeneration;

	T temp = chromosome[chP * size + j]; 
	__syncthreads();
	chromosome[indexNewChromosome * size + j] = temp;
	__syncthreads();

};
#pragma endregion

#pragma region Constructor
template <typename T>
GA_Lib<T>::GA_Lib(int size) {
	this->size = size;
	chromosomePerGeneration = 200;
	generation = 10000;
	crossoverRate = 0.35f;
	mutationRate = 0.15f;
	doInitializationClass();
};

template <typename T>
GA_Lib<T>::GA_Lib(long generation, int size, long chromosomePerGeneration, float mutationRate, float crossoverRate
	, CrossoverType crossoverType, MutationType mutationType
	, SelectionType selectionType) {
	this->size = size;
	this->generation = generation;
	this->chromosomePerGeneration = chromosomePerGeneration;
	this->crossoverRate = crossoverRate;
	this->mutationRate = mutationRate;
	this->stopCriteria = StopCriteria::GenerationComplete;
	this->crossoverType = crossoverType;
	this->mutationType = mutationType;
	this->selectionType = selectionType;
	doInitializationClass();
};

template <typename T>
GA_Lib<T>::GA_Lib(float fitnessBoundary, int size, long chromosomePerGeneration, float mutationRate, float crossoverRate
	, CrossoverType crossoverType, MutationType mutationType
	, SelectionType selectionType) {
	this->size = size;
	this->fitnessBoundary = fitnessBoundary;
	this->chromosomePerGeneration = chromosomePerGeneration;
	this->mutationRate = mutationRate;
	this->crossoverRate = crossoverRate;
	this->stopCriteria = StopCriteria::FitnessBoundaryAchieved;
	this->crossoverType = crossoverType;
	this->mutationType = mutationType;
	this->selectionType = selectionType;
	doInitializationClass();
};

template <typename T>
GA_Lib<T>::GA_Lib(long generation, float fitnessBoundary, int size, long chromosomePerGeneration, float mutationRate
	, float crossoverRate, CrossoverType crossoverType, MutationType mutationType
	, SelectionType selectionType) {
	this->size = size;
	this->generation = generation;
	this->fitnessBoundary = fitnessBoundary;
	this->chromosomePerGeneration = chromosomePerGeneration;
	this->crossoverRate = crossoverRate;
	this->mutationRate = mutationRate;
	this->stopCriteria = StopCriteria::FitnessAndGenerationCheck;
	this->crossoverType = crossoverType;
	this->mutationType = mutationType;
	this->selectionType = selectionType;
	doInitializationClass();
};

template <typename T>
GA_Lib<T>::~GA_Lib() {
	cudaDeviceReset();
}
#pragma endregion

#pragma region List of Accessor and Mutator
template <typename T>
long GA_Lib<T>::getGeneration() { return generation; };

template <typename T>
void GA_Lib<T>::setGeneration(long generation) { this->generation = generation; };

template <typename T>
float GA_Lib<T>::getFitnessBoundary() { return fitnessBoundary; };

template <typename T>
void GA_Lib<T>::setFitnessBoundary(float fitnessBoundary) { this->fitnessBoundary = fitnessBoundary; };

template <typename T>
long GA_Lib<T>::getChromosomePerGeneration() { return chromosomePerGeneration; };

template <typename T>
void GA_Lib<T>::setChromosomePerGeneration(long chromosomePerGeneration) { this->chromosomePerGeneration = chromosomePerGeneration; };

template <typename T>
float GA_Lib<T>::getCrossoverRate() { return crossoverRate; };

template <typename T>
void GA_Lib<T>::setCrossoverRate(float crossoverRate) { this->crossoverRate = crossoverRate; };

template <typename T>
float GA_Lib<T>::getMutationRate() { return mutationRate; };

template <typename T>
void GA_Lib<T>::setMutationRate(float mutationRate) { this->mutationRate = mutationRate; };

template <typename T>
GA_Lib<T>::StopCriteria GA_Lib<T>::getStopCriteria() { return stopCriteria; };

template <typename T>
void GA_Lib<T>::setStopCriteria(StopCriteria stopCriteria) { this->stopCriteria = stopCriteria; };

template <typename T>
GA_Lib<T>::CrossoverType GA_Lib<T>::getCrossoverType() { return crossoverType; };

template <typename T>
void GA_Lib<T>::setCrossoverType(CrossoverType crossoverType) { this->crossoverType = crossoverType; };

template <typename T>
GA_Lib<T>::MutationType GA_Lib<T>::getMutationType() { return mutationType; };

template <typename T>
void GA_Lib<T>::setMutationType(MutationType mutationType) { this->mutationType = mutationType; };

template <typename T>
GA_Lib<T>::SelectionType GA_Lib<T>::getSelectionType() { return selectionType; };

template <typename T>
void GA_Lib<T>::setSelectionType(SelectionType selectionType) { this->selectionType = selectionType; };

template <typename T>
T* GA_Lib<T>::getBestChromosome() { return bestChromosome; };

template <typename T>
T* GA_Lib<T>::getChromosome() { return chromosome; };

template <typename T>
void GA_Lib<T>::setChromosome(T* chromosomeArray) { chromosome = chromosomeArray; };

template <typename T>
float* GA_Lib<T>::getFitness() { return fitness; };

template <typename T>
float GA_Lib<T>::getLastBestFitness() { return lastBestFitness; };

template <typename T>
void GA_Lib<T>::setFitness(float* fitnessArray) { fitness = fitnessArray; };

template <typename T>
int GA_Lib<T>::getSize() { return size; };

template <typename T>
void GA_Lib<T>::setSize(int size) { this->size = size; };

template <typename T>
float GA_Lib<T>::getTotalTime() { return totalTime; };

template <typename T>
long GA_Lib<T>::getLastGeneration() { return currentGeneration; };

#pragma endregion

#pragma region Protected Method

template <typename T>
void GA_Lib<T>::run() {
	t1 = high_resolution_clock::now();
	doInitialize();
	while (!checkStoppingCriteria()) {
		//if (currentGeneration % 100 == 0) {
			//t2 = high_resolution_clock::now();

			//auto duration = duration_cast<milliseconds>(t2 - t1).count();
			//totalTime = (float)duration / 1000.00;
			//printf("%f\n", totalTime);
		//}
		currentGeneration++;
		doLoopInitialization();
		doGPUOperation();
		doSaveHistory();
	}
	doFreeMemory();
	doPrintResults();
};

template <typename T>
float* GA_Lib<T>::generateBestFitnessHistory() { return bestFitnessPerGeneration.data(); };

template <typename T>
float* GA_Lib<T>::generateAverageFitnessHistory() { return averageFitnessPerGeneration.data(); };
#pragma endregion

template <typename T>
__host__ __device__ float GA_Lib<T>::randomUniform() {
#ifdef __CUDA_ARCH__
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state;
	curand_init((unsigned long)clock() + i, 0, 0, &state);
	return curand_uniform(&state);
#else
	return rand() / (RAND_MAX);
#endif
};

template <typename T>
__host__ __device__ int GA_Lib<T>::randomInt(int max) {
#ifdef __CUDA_ARCH__
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state;
	curand_init((unsigned long)clock() + i, 0, 0, &state);
	return curand_uniform(&state) * max;
#else
	return rand() % (max + 1);
#endif
};

template <typename T>
__host__ __device__ int GA_Lib<T>::randomInt(int min, int max) {
#ifdef __CUDA_ARCH__
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state;
	curand_init((unsigned long)clock() + i, 0, 0, &state);
	return (curand_uniform(&state) * (max - min)) + min;
#else
	return (rand() % (max + 1 - min)) + min;
#endif
};

template <typename T>
__host__ __device__ float GA_Lib<T>::randomFloat(float max) {
#ifdef __CUDA_ARCH__
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state;
	curand_init((unsigned long)clock() + i, 0, 0, &state);
	return curand_uniform(&state) * max;
#else
	return (((float)rand()) / (float)RAND_MAX) * max;
#endif
};

template <typename T>
__host__ __device__ float GA_Lib<T>::randomFloat(float min, float max) {
#ifdef __CUDA_ARCH__
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	curandState state;
	curand_init((unsigned long)clock() + i, 0, 0, &state);
	return (curand_uniform(&state) * (max - min)) + min;
#else
	return ((((float)rand()) / (float)RAND_MAX) * (max - min)) + min;
#endif
};

#pragma region Private Method

template <typename T>
void GA_Lib<T>::doInitializationClass() {
	printf("Do initialization Class..\n");
	srand(time(NULL));
	cudaSetDevice(0);
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("Finished Do initialization Class...\n");
};

template <typename T>
void GA_Lib<T>::doInitialize() {
	printf("Do initialization..\n");

	currentGeneration = -1;
	lastBestFitness = 0.0f;
	cudaMallocManaged(&fitness, sizeof(float) * chromosomePerGeneration * 2);
	cudaMallocManaged(&chromosome, sizeof(T) * size * chromosomePerGeneration * 2);
	cudaMallocManaged(&genChangedOption, sizeof(bool)*size);
	cudaMallocHost(&bestChromosome, sizeof(T)*size);

	this->doInitialization();
	this->setGenChangedOption(genChangedOption);
	this->doFitnessCheck(chromosomePerGeneration);
	cudaDeviceSynchronize();

	doSortingAndAveraging(chromosomePerGeneration);
	cudaDeviceSynchronize();

	counterStop = 0;

	printf("Finished Do Initialization...\n");
};

template <typename T>
void GA_Lib<T>::doLoopInitialization() {
	//system("CLS");
	//printf("Iteration %i...\n", currentGeneration);
	//printf("Looping init\n");
	//printf("f: %f \n", fitness[0]);
	//printf("c: %i \n", chromosome[0]);
};

template <typename T>
void GA_Lib<T>::doGPUOperation() {
	//printf("Starting GPU Operation..\n");

	//printf("S GPU OP\n");
	//printf("f: %f \n", fitness[0]);
	//printf("c: %i \n", chromosome[0]);
	doGeneticOperation();
	cudaDeviceSynchronize();
	//printf("E GPU OP\n");
	//printf("f: %f \n", fitness[0]);
	//printf("c: %i \n", chromosome[0]);
	this->doFitnessCheck(chromosomePerGeneration * 2);
	cudaDeviceSynchronize();
	//printf("E Fitness Check\n");
	//printf("f: %f \n", fitness[0]);
	//printf("c: %i \n", chromosome[0]);
	doSortingAndAveraging(chromosomePerGeneration * 2);
	cudaDeviceSynchronize();
	//system("pause");

	//printf("Finished Starting GPU Operation..\n");
};

template <typename T>
void GA_Lib<T>::doGeneticOperation() {
	//printf("Crossover And mutation..\n");
	float randVal;
	long currentChromosomeLoop = 0;
	int chromosomeCount1 = 0, chromosomeCount2 = 0;
	long *parent1, *parent2;
	cudaMallocManaged(&parent1, sizeof(int)*chromosomePerGeneration / 2);
	cudaMallocManaged(&parent2, sizeof(int)*chromosomePerGeneration / 2);
	if (mutationType == 0) cudaMallocManaged(&randChromosome, sizeof(T)*size);
	while (currentChromosomeLoop < chromosomePerGeneration / 2) {
		parent1[currentChromosomeLoop] = doSelection(-1);
		parent2[currentChromosomeLoop] = doSelection(parent1[currentChromosomeLoop]);

		randVal = (float)rand() / (RAND_MAX);
		if (randVal <= crossoverRate) chromosomeCount1++;
		else chromosomeCount2++;

		currentChromosomeLoop++;
	}

	//Do Crossover
	if (this->crossoverType == CrossoverType::OnePointCrossover) {
		doOnePointCrossover<T><<<chromosomeCount1 * 2, size>>> (parent1, parent2, chromosomePerGeneration, chromosome, mutationType, mutationRate, genChangedOption);
	}
	else if (this->crossoverType == CrossoverType::MultiPointCrossover) {
		doMultiPointCrossover<T><<<chromosomeCount1 * 2, size>>> (parent1, parent2, chromosomePerGeneration, chromosome, mutationType, mutationRate, genChangedOption);
	}
	else if (this->crossoverType == CrossoverType::UniformCrossover) {
		doUniformCrossover<T><<<chromosomeCount1 * 2, size>>> (parent1, parent2, chromosomePerGeneration, chromosome, mutationType, mutationRate, genChangedOption);
	}
	else if (this->crossoverType == CrossoverType::PMXCrossover) {
		size_t memory = 5 * size * sizeof(T) + size * sizeof(bool);
		doPMXCrossover<T><<<chromosomeCount1 * 2, size, memory>>> (parent1, parent2, chromosomePerGeneration, chromosome, mutationType, mutationRate, genChangedOption);
	}
	else if (this->crossoverType == CrossoverType::Order1Crossover) {
		size_t memory = 3 * size * sizeof(T) + 2 * size * sizeof(int);
		doOrder1Crossover<T><<<chromosomeCount1 * 2, size, memory>>> (parent1, parent2, chromosomePerGeneration, chromosome, mutationType, mutationRate, genChangedOption);
	}
	generateChild<T><<<chromosomeCount2 * 2, size>>> (parent1, parent2, chromosomePerGeneration, chromosomeCount1, chromosome, mutationType, mutationRate, genChangedOption);
	cudaDeviceSynchronize();
	
	//Do mutation
	if (this->mutationType == MutationType::RandomResetting) {
		doRandomResetting<T><<<chromosomePerGeneration,size>>>(mutationRate, chromosome, genChangedOption, randChromosome);
	}
	else if (this->mutationType == MutationType::SwapMutation) {
		doSwapMutation<T><<<chromosomePerGeneration, size>>>(mutationRate, chromosome, genChangedOption);
	}
	else if (this->mutationType == MutationType::ScrambleMutation) {
		size_t memory = size * sizeof(T);
		doScrambleMutation<T><<<chromosomePerGeneration, size, memory>>>(mutationRate, chromosome, genChangedOption);
	}
	else if (this->mutationType == MutationType::InversionMutation) {
		size_t memory = size * sizeof(T);
		doInversionMutation<T><<<chromosomePerGeneration, size, memory>>>(mutationRate, chromosome, genChangedOption);
	}
	cudaDeviceSynchronize();
	/*
	for (int j = 0; j < size; j++)
	{
		printf("%i ", chromosome[parent1[0] * size + j]);
	}
	printf("\n");
	for (int j = 0; j < size; j++)
	{
		printf("%i ", chromosome[parent2[0] * size + j]);
	}
	printf("\n");
	for (int j = 0; j < size; j++)
	{
		printf("%i ", chromosome[chromosomePerGeneration * size + j]);
	}
	printf("\n\n");
	system("pause");*/
	cudaFree(parent1);
	cudaFree(parent2);
	if (mutationType == 0) cudaFree(randChromosome);
	//printf("Finished Crossover And mutation..\n");
};

template <typename T>
long GA_Lib<T>::doSelection(long exceptChromosome) {
	long returnVal = exceptChromosome;
	while (returnVal == exceptChromosome) {
		if (selectionType == SelectionType::RankSelection) {
			//TODOList: Check alternative type (Arithmatic Progression) (Because this will produce very large number)
			int a = chromosomePerGeneration * 10 - 1;
			int b = -10;
			int n = chromosomePerGeneration;

			long totalRank = n / 2 * (2 * a + ((n - 1)*b));

			//The result for random
			long long randRankVal = (long long)((((float)rand()) / (float)RAND_MAX) * totalRank) + 1;

			//Quadratic function x
			returnVal = ((b - (2 * a)) + pow(pow((2 * a - b), 2) + (8 * b*randRankVal), 0.5)) / (2 * b);
		}
		else if (selectionType == SelectionType::RouletteWheelSelection) {
			//Do reduction on device using thrust
			float* fitnessTemp;
			cudaMallocManaged(&fitnessTemp, sizeof(float) * chromosomePerGeneration);
			float totalFitness = thrust::reduce(fitness, fitness + chromosomePerGeneration, 0);
			thrust::inclusive_scan(fitness, fitness + chromosomePerGeneration, fitnessTemp);
			cudaDeviceSynchronize();

			int numBlocks = 1;
			int threadPerBlocks = (deviceProp.maxThreadsPerBlock / 10 >= chromosomePerGeneration) ? chromosomePerGeneration : deviceProp.maxThreadsPerBlock / 10;

			//The result for random
			float randFitnessVal = (((float)rand()) / (float)RAND_MAX) * totalFitness;

			long* returnChromosome;
			cudaMallocManaged(&returnChromosome, sizeof(long));
			doSearch << <numBlocks, threadPerBlocks >> >(fitnessTemp, randFitnessVal, chromosomePerGeneration, returnChromosome);
			cudaDeviceSynchronize();

			returnVal = returnChromosome[0];

			cudaFree(fitnessTemp);
			cudaFree(returnChromosome);
		}
		else if (selectionType == SelectionType::TournamentSelection) {
			float* tempFitness;
			long* tempChromosome;

			int passedChromosome = (int)pow(chromosomePerGeneration, 0.5f);
			int numBlocks = (int)ceil(passedChromosome*1.0f / deviceProp.maxThreadsPerBlock*1.0f);
			int threadPerBlocks = passedChromosome / numBlocks;

			//Tournament selection amount = root(chromosomePerGeneration)
			cudaMallocManaged(&tempFitness, sizeof(float) * passedChromosome);
			cudaMallocManaged(&tempChromosome, sizeof(long) * passedChromosome);

			//Random chromosome index that will be included in tournament selection
			randomizeChromosome << <numBlocks, threadPerBlocks >> > (tempChromosome, tempFitness, fitness, chromosomePerGeneration);
			cudaDeviceSynchronize();

			thrust::sort_by_key(tempFitness, tempFitness + passedChromosome, tempChromosome, thrust::greater<float>());
			cudaDeviceSynchronize();

			//Copy from device, index of chromosome that has been sorted based on its fitness
			returnVal = tempChromosome[0];

			cudaFree(tempFitness);
			cudaFree(tempChromosome);
		}
	}
	return returnVal;
};

template <typename T>
void GA_Lib<T>::doSortingAndAveraging(long fitnessSize) {
	//printf("Sorting And Averaging..\n");

	int *resultedIndexChromosome;
	cudaMallocManaged(&resultedIndexChromosome, sizeof(int)*fitnessSize);

	int numBlocks = (int)ceil(fitnessSize * 1.0f / deviceProp.maxThreadsPerBlock * 1.0f);
	int threadPerBlocks = fitnessSize * 1.0f / numBlocks * 1.0f;
	assignSequencedValue << <numBlocks, threadPerBlocks >> > (resultedIndexChromosome);
	cudaDeviceSynchronize();

	if (fitnessSize == chromosomePerGeneration * 2) {
		averageFitnessThisGeneration = thrust::reduce(fitness, fitness + chromosomePerGeneration, 0, thrust::plus<float>());
		cudaDeviceSynchronize();
		averageFitnessThisGeneration /= chromosomePerGeneration;
	}
	//Do sort_by_key using thrust, then return it to origin fitness and chromosome
	cudaDeviceSynchronize();
	thrust::sort_by_key(fitness, fitness + fitnessSize, resultedIndexChromosome, thrust::greater<float>());
	cudaDeviceSynchronize();

	numBlocks = 1;
	threadPerBlocks = fitnessSize;
	//printf("S Sorting\n");
	//printf("f: %f \n", fitness[0]);
	//printf("c: %i \n", chromosome[0]);
	assignChromosome<T> << <numBlocks, threadPerBlocks, sizeof(T) * size * fitnessSize>> > (resultedIndexChromosome, chromosome, size);
	cudaDeviceSynchronize();
	//printf("E Sorting\n");
	//printf("f: %f \n", fitness[0]);
	//printf("c: %i \n", chromosome[0]);

	cudaFree(resultedIndexChromosome);
	//printf("Finished Sorting and Averaging..\n");
};

template <typename T>
void GA_Lib<T>::doSaveHistory() {
	//printf("Saving History...\n");
	//printf("S Save History\n");
	//printf("f: %f \n", fitness[0]);
	//printf("c: %i \n", chromosome[0]);
	this->bestFitnessPerGeneration.push_back(fitness[0]);
	this->averageFitnessPerGeneration.push_back(this->averageFitnessThisGeneration);
	for (int i = 0; i < size; i++) {
		bestChromosome[i] = chromosome[i];
	}
	//printf("E Save History\n");
	//printf("f: %f \n", fitness[0]);
	//printf("c: %i \n", chromosome[0]);
	//printf("Finished...\n");
};

template <typename T>
void GA_Lib<T>::doFreeMemory() {
	cudaFree(fitness);
	cudaFree(chromosome);
	cudaFree(genChangedOption);
};

template <typename T>
void GA_Lib<T>::doPrintResults() {
	t2 = high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(t2 - t1).count();
	totalTime = (float)duration / 1000.00;
	printf("Operation Finished..\n");
	printf("Total Execution Time: %f s\n", totalTime);
	printf("Operation finished in generation %d...\n", currentGeneration);
	printf("Best Fitness in last generation: %f\n", lastBestFitness);
}

template <typename T>
bool GA_Lib<T>::checkStoppingCriteria() {
	//If in 10000 generation the best fitness is the same, stop the process
	//if (lastBestFitness == fitness[0]) {
		//if (counterStop >= 100000) {
			//printf("Best Fitness this generation: %f\n", lastBestFitness);
			//return true;
		//}
		//counterStop++;
	//}
	//else counterStop = 0;
	lastBestFitness = fitness[0];

	if (currentGeneration == -1) return false;
	//else printf("Best Fitness this generation: %f\n", lastBestFitness);
 
	if (stopCriteria == StopCriteria::FitnessBoundaryAchieved && bestFitnessPerGeneration[currentGeneration] >= fitnessBoundary) { return true; }
	else if (stopCriteria == StopCriteria::GenerationComplete && currentGeneration >= generation) { return true; }
	else if (stopCriteria == StopCriteria::FitnessAndGenerationCheck && (currentGeneration >= generation || bestFitnessPerGeneration[currentGeneration] >= fitnessBoundary)) { return true; }
	return false;
};
#pragma endregion
/*
class tryTSPGACPU : public GA_Lib<short> {

public:
	float* coord;
	tryTSPGACPU() : GA_Lib((long)10, 38, 30, 0.15, 0.35,
		CrossoverType::Order1Crossover, MutationType::InversionMutation, SelectionType::RankSelection) {
		run();
		short* bestChromosome = getBestChromosome();
		printf("City Index: %i", bestChromosome[0]);
		for (int i = 1; i < getSize(); i++) {
			printf(",%i", bestChromosome[i]);
		}
		printf("\n\n");
	}

	void doInitialization() {
		coord = new float[2 * getSize()];
		short* newChromosome = this->getChromosome();

		//read file
		ifstream file("D:\\Materi Kuliah\\Semester 7\\[TA]\\[VERSION 2]\\[CUDA_LIB]\\TSP txt\\dj38.tsp.txt");
		string line;
		bool startRead = false;
		int i = 0;
		while (getline(file, line)) {
			stringstream linestream(line);
			if (startRead) {
				int temp;
				float c1, c2;
				linestream >> temp >> c1 >> c2;
				coord[i] = c1;
				coord[i + 1] = c2;
				i += 2;
			}
			if (line == "NODE_COORD_SECTION") startRead = true;
		}
		for (int i = 0; i < getChromosomePerGeneration(); i++) {
			short* a = new short[getSize()];
			randomChromosome(a);
			for (int j = 0; j < getSize(); j++) {
				newChromosome[i * getSize() + j] = a[j];
			}
			delete a;
		}
		setChromosome(newChromosome);
	}

	void doFitnessCheck(long chromosomeAmount) {
		short* chromosomeTemp = this->getChromosome();
		float* fitnessTemp = this->getFitness();
		for (int i = 0; i < chromosomeAmount; i++) {
			cudaDeviceSynchronize();
			float nilai = 0;
			int i_d = i*getSize();
			for (int j = 1; j < getSize(); j++) {
				int xDist = coord[chromosomeTemp[i_d + j] * 2] - coord[chromosomeTemp[i_d + (j - 1)] * 2];
				int yDist = coord[(chromosomeTemp[i_d + j] * 2) + 1] - coord[(chromosomeTemp[i_d + (j - 1)] * 2) + 1];
				nilai += (sqrtf((xDist*xDist) + (yDist*yDist)));
			}
			int xDist = coord[chromosomeTemp[i_d] * 2] - coord[chromosomeTemp[i_d + (getSize() - 1)] * 2];
			int yDist = coord[(chromosomeTemp[i_d] * 2) + 1] - coord[(chromosomeTemp[i_d + (getSize() - 1)] * 2) + 1];
			nilai += (sqrtf((xDist*xDist) + (yDist*yDist)));
			fitnessTemp[i] = 99999.0f - nilai;
		}
		setFitness(fitnessTemp);
		//callFitnessCheckGPU(81, chromosomeTemp, fitnessTemp, chromosomeAmount);
	};

	void randomChromosome(short* newChromosome) {
		short* temp = new short[getSize()];
		for (short i = 0; i < getSize(); i++) temp[i] = i;
		for (short i = 0; i < 300; i++) {
			short index1 = randomInt(getSize() - 1), index2 = randomInt(getSize() - 1);
			short temp2 = temp[index1];
			temp[index1] = temp[index2];
			temp[index2] = temp2;
		}
		for (int i = 0; i < getSize(); i++) {
			newChromosome[i] = temp[i];
		}
		delete temp;
	};
};

int main() {
	tryTSPGACPU try1;
	system("pause");
}
*/