#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

#pragma once
template <typename T>
class __declspec(dllexport) GA_Lib {
public:
#pragma region Enumeration
	enum StopCriteria : short {
		GenerationComplete = 0,
		FitnessBoundaryAchieved = 1,
		FitnessAndGenerationCheck = 2
	};
	enum CrossoverType : short {
		OnePointCrossover = 0,
		MultiPointCrossover = 1,
		UniformCrossover = 2,
		PMXCrossover = 3,
		Order1Crossover = 4
	};
	enum MutationType : short {
		RandomResetting = 0,
		SwapMutation = 1,
		ScrambleMutation = 2,
		InversionMutation = 3
	};
	enum SelectionType : short {
		RouletteWheelSelection = 0,
		TournamentSelection = 1,
		RankSelection = 2
	};
#pragma endregion

#pragma region Constructor
	GA_Lib(int size);
	GA_Lib(long generation, int size, long chromosomePerGeneration, float mutationRate, float crossoverRate
		, CrossoverType crossoverType, MutationType mutationType
		, SelectionType selectionType);
	GA_Lib(float fitnessBoundary, int size, long chromosomePerGeneration, float mutationRate, float crossoverRate
		, CrossoverType crossoverType, MutationType mutationType
		, SelectionType selectionType);
	GA_Lib(long generation, float fitnessBoundary, int size, long chromosomePerGeneration, float mutationRate
		, float crossoverRate, CrossoverType crossoverType, MutationType mutationType
		, SelectionType selectionType);
	~GA_Lib();
#pragma endregion

#pragma region List of Accessor and Mutator
	long getGeneration();
	void setGeneration(long generation);
	float getFitnessBoundary();
	void setFitnessBoundary(float fitnessBoundary);
	long getChromosomePerGeneration();
	void setChromosomePerGeneration(long chromosomePerGeneration);
	float getCrossoverRate();
	void setCrossoverRate(float crossoverRate);
	float getMutationRate();
	void setMutationRate(float mutationRate);
	StopCriteria getStopCriteria();
	void setStopCriteria(StopCriteria stopCriteria);
	CrossoverType getCrossoverType();
	void setCrossoverType(CrossoverType crossoverType);
	MutationType getMutationType();
	void setMutationType(MutationType mutationType);
	SelectionType getSelectionType();
	void setSelectionType(SelectionType selectionType);
	T* getBestChromosome();
	//Used in Fitness Check
	T* getChromosome();
	//Used in Initialization for Setting Chromosome
	void setChromosome(T* chromosomeArray);
	float* getFitness();
	float getLastBestFitness();
	void setFitness(float* fitnessArray);
	int getSize();
	void setSize(int size);
	float getTotalTime();
	long getLastGeneration();
#pragma endregion

#pragma region Protected Method
	void run();
	//Return Best Fitness In Every Generation
	float* generateBestFitnessHistory();
	//Return Average Fitness In Every Generation
	float* generateAverageFitnessHistory();
	//Random Float From 0 Until 1
	__host__ __device__ float randomUniform();
	//Inclusive Random Integer With Maximum Integer As Parameter (Starting From 0 until Maximum Value)
	__host__ __device__ int randomInt(int max);
	//Inclusive Random Integer With Minimum and Maximum Integer As Parameter (Starting From Minimum until Maximum Value)
	__host__ __device__ int randomInt(int min, int max);
	//Inclusive Random Float With Maximum Float As Parameter (Starting From 0 until Maximum Value)
	__host__ __device__ float randomFloat(float max);
	//Inclusive Random Float With Minimum and Maximum Float As Parameter (Starting From Minimum until Maximum Value)
	__host__ __device__ float randomFloat(float min, float max);
#pragma region Abstract Function (must be overridden if the virtual method = 0, otherwise can be overriden only)
	//Initialize GA
	virtual void doInitialization() = 0;
	//Set Gen Changed Option Of Chromosome, doesn't have to be overriden (true if the gen inside chromosome can be changed and vice versa)
	virtual void setGenChangedOption(bool* genChangedOption) {
		for (int i = 0; i < size; i++) {
			genChangedOption[i] = true;
		}
	};
	virtual void randomChromosome(T* newChromosome) {};
	virtual void doFitnessCheck(long chromosomeAmount) = 0;
#pragma endregion

private:
#pragma region Properties
	int size;
	high_resolution_clock::time_point t1;
	high_resolution_clock::time_point t2;
	//Total Generation
	long generation;
	long currentGeneration;
	//Fitness Boundary for Stopping Criteria
	float fitnessBoundary;
	//Total Chromosome Per Generation After Elimination
	long chromosomePerGeneration;
	StopCriteria stopCriteria;
	CrossoverType crossoverType;
	MutationType mutationType;
	SelectionType selectionType;
	float crossoverRate;
	float mutationRate;
	//Chromosome Solution in Current Generation
	T* chromosome;
	T* bestChromosome;
	T* randChromosome;
	//Option If Gen in Chromosome Can Be Changed Or Not
	bool* genChangedOption;
	//Stop the Process If The Best Fitness Doesn't Change In 100 Generation
	float lastBestFitness;
	short counterStop;
	//Fitness in Current Generation
	float* fitness;
	//For Performance Reason
	float averageFitnessThisGeneration;
	vector<float> bestFitnessPerGeneration;
	vector<float> averageFitnessPerGeneration;
	//Storing CUDA Capability
	cudaDeviceProp deviceProp;
	float totalTime;
#pragma endregion

#pragma region Private Method
	//Initialization After Constructor
	void doInitializationClass();
	//Initialization Before Looping
	void doInitialize();
	//Initialization in Every Loop
	void doLoopInitialization();
	//All Cuda Operation Inside Class
	void doGPUOperation();
	void doGeneticOperation();
	//Do Selection for Crossover With 1 Exception Chromosome
	long doSelection(long exceptChromosome);
	//Sorting Every Chromosome Based on Its Fitness's
	void doSortingAndAveraging(long fitnessSize);
	//Save History of Best and Average Fitness
	void doSaveHistory();
	void doFreeMemory();
	void doPrintResults();
	//Function to Check Stopping Criteria
	bool checkStoppingCriteria();
#pragma endregion
};

template class __declspec(dllexport) GA_Lib<bool>;
template class __declspec(dllexport) GA_Lib<char>;
template class __declspec(dllexport) GA_Lib<short>;
template class __declspec(dllexport) GA_Lib<int>;
template class __declspec(dllexport) GA_Lib<float>;
template class __declspec(dllexport) GA_Lib<double>;
template class __declspec(dllexport) GA_Lib<long>;
template class __declspec(dllexport) GA_Lib<long long>;
