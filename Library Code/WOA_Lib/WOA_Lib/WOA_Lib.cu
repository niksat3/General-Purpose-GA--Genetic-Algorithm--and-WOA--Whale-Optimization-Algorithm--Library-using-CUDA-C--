#include "WOA_Lib.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand_kernel.h>
#include <chrono>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
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
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#pragma region GPU method

__global__ void assignSequencedValueWOA(int* data) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	data[i] = i;
};

__global__ void assignSearchAgent(int* indexData, float* searchAgent, int size, int numberOfSearchAgent) {
	extern __shared__ float sFloat[];
	float* tempSearchAgent = sFloat;
	int i = threadIdx.x;
	int index = indexData[i];

	for (int j = 0; j < size; j++) {
		tempSearchAgent[i * size + j] = searchAgent[index * size + j];
	}
	__syncthreads();
	for (int j = 0; j < size; j++) {
		searchAgent[i * size + j] = tempSearchAgent[i * size + j];
	}
};

__global__ void copyArray(float* newSearchAgent, float* searchAgent) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	searchAgent[(i + 1) * blockDim.x + j] = newSearchAgent[i * blockDim.x + j];
};

__global__ void spiralUpdatingPosition(int size, float* searchAgent, float* newSearchAgent, bool* genChangedOption) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	
	curandState state;
	curand_init((unsigned long)clock() + i, 0, 0, &state);
	float t = (curand_uniform(&state) * 2) - 1;
	if (genChangedOption[j]) {
		float D = abs(searchAgent[j] - searchAgent[(i+1) * size + j]);
		newSearchAgent[i * size + j] = D * exp(t) * cos(2 * M_PI * t) + searchAgent[j];
	}
	else newSearchAgent[i * size + j] = searchAgent[(i+1) * size + j];
};

__global__ void huntPrey(int size, float* searchAgent, float* newSearchAgent, float C, float A, int indexOtherWhale, bool* genChangedOption){
	int i = blockIdx.x;
	int j = threadIdx.x;

	if (genChangedOption[j]) {
		float D = abs(C * searchAgent[indexOtherWhale * size + j] - searchAgent[(i+1) * size + j]);
		newSearchAgent[i * size + j] = searchAgent[indexOtherWhale * size + j] - A * D;
	}
	else newSearchAgent[i * size + j] = searchAgent[(i+1) * size + j];
};
#pragma endregion

#pragma region Constructor
WOA_Lib::WOA_Lib(int size) {
	this->size = size;
	numberOfSearchAgent = 200;
	generation = 10000;
	doInitializationClass();
};

WOA_Lib::WOA_Lib(long generation, int size, long numberOfSearchAgent) {
	this->size = size;
	this->generation = generation;
	this->numberOfSearchAgent = numberOfSearchAgent;
	doInitializationClass();
};

WOA_Lib::~WOA_Lib() {
}
#pragma endregion

#pragma region List of Accessor and Mutator
long WOA_Lib::getGeneration() { return generation; };

void WOA_Lib::setGeneration(long generation) { this->generation = generation; };

long WOA_Lib::getNumberOfSearchAgent() { return numberOfSearchAgent; };

void WOA_Lib::setNumberOfSearchAgent(long numberOfSearchAgent) { this->numberOfSearchAgent = numberOfSearchAgent; };

float* WOA_Lib::getLeader() { return leader; };

//Used in Initialization for Setting Search Agent
void WOA_Lib::setSearchAgent(float* searchAgentArray) { searchAgent = searchAgentArray; };

float* WOA_Lib::getSearchAgent() { return searchAgent; };

float* WOA_Lib::getFitness() { return fitness; };

float WOA_Lib::getLastBestFitness() { return lastBestFitness; };

void WOA_Lib::setFitness(float* fitnessArray) { fitness = fitnessArray; };

int WOA_Lib::getSize() { return size; };

void WOA_Lib::setSize(int size) { this->size = size; };

float WOA_Lib::getTotalTime() { return totalTime; };

long WOA_Lib::getLastGeneration() { return currentGeneration; };

#pragma endregion

#pragma region Public Method

void WOA_Lib::run() {
	t1 = high_resolution_clock::now();
	doInitialize();
	while (!checkStoppingCriteria()) {
		if(currentGeneration%100==0) {
			t2 = high_resolution_clock::now();

			auto duration = duration_cast<milliseconds>(t2 - t1).count();
			totalTime = (float)duration / 1000.00;
			printf("%f\n", totalTime);
		}
		currentGeneration++;
		doLoopInitialization();
		doGPUOperation();
		doSaveHistory();
	}
	doFreeMemory();
	doPrintResults();
};

//Return Best Fitness In Every Generation
float* WOA_Lib::generateBestFitnessHistory() { return bestFitnessPerGeneration.data(); };

//Return Average Fitness In Every Generation
float* WOA_Lib::generateAverageFitnessHistory() { return averageFitnessPerGeneration.data(); };
#pragma endregion

__host__ __device__ float WOA_Lib::randomUniform() {
	#ifdef __CUDA_ARCH__
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);
		return curand_uniform(&state);
	#else
		return rand() / (RAND_MAX);
	#endif
};

__host__ __device__ int WOA_Lib::randomInt(int max) {
	#ifdef __CUDA_ARCH__
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);
		return curand_uniform(&state) * max;
	#else
		return rand() % (max + 1);
	#endif
};

__host__ __device__ int WOA_Lib::randomInt(int min, int max) {
	#ifdef __CUDA_ARCH__
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);
		return (curand_uniform(&state) * (max - min)) + min;
	#else
		return (rand() % (max + 1 - min)) + min;
	#endif
};

__host__ __device__ float WOA_Lib::randomFloat(float max) {
	#ifdef __CUDA_ARCH__
		int i = blockDim.x * blockIdx.x + threadIdx.x;
		curandState state;
		curand_init((unsigned long)clock() + i, 0, 0, &state);
		return curand_uniform(&state) * max;
	#else
		return (((float)rand()) / (float)RAND_MAX) * max;
	#endif
};

__host__ __device__ float WOA_Lib::randomFloat(float min, float max) {
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

//Initialization After Constructor
void WOA_Lib::doInitializationClass() {
	printf("Do initialization Class..\n");
	srand(time(NULL));
	cudaSetDevice(0);
	cudaGetDeviceProperties(&deviceProp, 0);
	printf("Finished Do initialization Class...\n");
};

//Initialization Before Looping
void WOA_Lib::doInitialize() {
	printf("Do initialization..\n");
	currentGeneration = -1;
	lastBestFitness = 0.0f;
	
	cudaMallocManaged(&fitness, sizeof(float)*numberOfSearchAgent);
	cudaMallocManaged(&searchAgent, sizeof(float)*size*numberOfSearchAgent);
	cudaMallocManaged(&genChangedOption, sizeof(bool)*size);
	cudaMallocHost(&leader, sizeof(float)*size);

	doCountVariable();

	this->doInitialization();
	this->setGenChangedOption(genChangedOption);
	this->doFitnessCheck(numberOfSearchAgent);
	cudaDeviceSynchronize();
	
	doSortingAndAveraging(numberOfSearchAgent);
	
	counterStop = 0;
	printf("Finished Do Initialization...\n");
};

void WOA_Lib::doLoopInitialization() {
	doCountVariable();
};

void WOA_Lib::doCountVariable() {
	float r1 = (float)(rand() / (float)RAND_MAX), r2 = (float)(rand() / (float)RAND_MAX);
	if (currentGeneration == -1) a = 2.0;
	else a -= (2.0 / generation);
	A = 2.0 * a * r1 - a;
	C = 2.0 * r2;
	p = (float)(rand() / (float)RAND_MAX);
};

//All Cuda Operation Inside Class
void WOA_Lib::doGPUOperation() {
	float* newSearchAgent;

	cudaMallocManaged(&newSearchAgent, (numberOfSearchAgent-1) * sizeof(float) * size);

	updateSearchAgentPosition(searchAgent, newSearchAgent);
	cudaDeviceSynchronize();
	copyArray<<<numberOfSearchAgent-1,size>>>(newSearchAgent,searchAgent);
	cudaDeviceSynchronize();
	this->doFitnessCheck(numberOfSearchAgent);
	cudaDeviceSynchronize();
	doSortingAndAveraging(numberOfSearchAgent);
	cudaDeviceSynchronize();

	cudaFree(newSearchAgent);
};

void WOA_Lib::updateSearchAgentPosition(float* searchAgent, float* newSearchAgent) {
	if (p < 0.5) {
		if (abs(A) >= 1) huntPrey<< <numberOfSearchAgent-1, size >> > (size, searchAgent, newSearchAgent, C, A, 0, genChangedOption);
		else huntPrey<<<numberOfSearchAgent-1,size>>> (size, searchAgent, newSearchAgent, C, A, rand() % numberOfSearchAgent, genChangedOption);
	}
	else spiralUpdatingPosition<<<numberOfSearchAgent-1,size>>>(size, searchAgent, newSearchAgent, genChangedOption);
};

//Sorting Every searchAgent Based on Its Fitness's
void WOA_Lib::doSortingAndAveraging(int fitnessSize) {
	//printf("Sorting And Averaging..\n");

	int *resultedIndexSearchAgent;
	cudaMallocManaged(&resultedIndexSearchAgent, sizeof(int)*(fitnessSize));

	int numBlocks = (int)ceil(numberOfSearchAgent*1.0f / deviceProp.maxThreadsPerBlock*1.0f);
	int threadPerBlocks = numberOfSearchAgent* 1.0f / numBlocks * 1.0f;
	assignSequencedValueWOA<<<numBlocks, threadPerBlocks>>>(resultedIndexSearchAgent);
	cudaDeviceSynchronize();
	
	if (currentGeneration != -1) {
		averageFitnessThisGeneration = thrust::reduce(fitness, fitness + numberOfSearchAgent, 0, thrust::plus<float>());
		cudaDeviceSynchronize();
		averageFitnessThisGeneration /= numberOfSearchAgent;
	}
	//Do sort_by_key using thrust, then return it to origin fitness and chromosome
	cudaDeviceSynchronize();
	thrust::sort_by_key(fitness, fitness + fitnessSize, resultedIndexSearchAgent, thrust::greater<float>());
	cudaDeviceSynchronize();

	numBlocks = 1;
	threadPerBlocks = fitnessSize;

	assignSearchAgent<<<numBlocks, threadPerBlocks, sizeof(float) * size * fitnessSize >>> (resultedIndexSearchAgent, searchAgent, size, numberOfSearchAgent);
	cudaDeviceSynchronize();
	cudaFree(resultedIndexSearchAgent);

};

//Save History of Best and Average Fitness
void WOA_Lib::doSaveHistory() {
	this->bestFitnessPerGeneration.push_back(fitness[0]);
	this->averageFitnessPerGeneration.push_back(this->averageFitnessThisGeneration);
	for (int i = 0; i < size; i++) {
		leader[i] = searchAgent[i];
	}
};

void WOA_Lib::doPrintResults() {
	t2 = high_resolution_clock::now();

	auto duration = duration_cast<milliseconds>(t2 - t1).count();
	totalTime = (float)duration / 1000.00;
	printf("Operation Finished..\n");
	printf("Total Execution Time: %f s\n", totalTime);
	printf("Operation finished in generation %d...\n", currentGeneration);
	printf("Best Fitness in last generation: %f\n", lastBestFitness);
}

void WOA_Lib::doFreeMemory() {
	cudaFree(fitness);
	cudaFree(searchAgent);
	cudaFree(genChangedOption);
};

//Function to Check Stopping Criteria
bool WOA_Lib::checkStoppingCriteria() {
	//If in 100 generation the best fitness is the same, stop the process
	if (lastBestFitness == fitness[0]) {
		if (counterStop >= 100000) {
			return true;
		}
		counterStop++;
	}
	else counterStop = 0;
	lastBestFitness = fitness[0];

	if (currentGeneration >= generation) { return true; }
	
	return false;
};
#pragma endregion