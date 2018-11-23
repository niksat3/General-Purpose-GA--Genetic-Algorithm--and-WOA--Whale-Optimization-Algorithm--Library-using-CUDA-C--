
#define _USE_MATH_DEFINES

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

class __declspec(dllexport) WOA_Lib {
public:
	#pragma region Constructor
	WOA_Lib(int size);
	WOA_Lib(long generation, int size, long numberOfSearchAgent);
	~WOA_Lib();
	#pragma endregion

	#pragma region List of Accessor and Mutator
	long getGeneration();
	void setGeneration(long generation);
	long getNumberOfSearchAgent();
	void setNumberOfSearchAgent(long numberOfSearchAgent);
	float* getLeader();
	void setSearchAgent(float* searchAgentArray);
	float* getSearchAgent();
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
	#pragma endregion

	#pragma region Abstract Function (must be overridden if the virtual method = 0, otherwise can be overriden only)
	//Initialize WOA
	virtual void doInitialization() = 0;
	//Set Gen Changed Option Of Chromosome, doesn't have to be overriden (true if the gen inside chromosome can be changed and vice versa)
	virtual void setGenChangedOption(bool* genChangedOption) {
		for (int i = 0; i < size; i++) {
			genChangedOption[i] = true;
		}
	};
	virtual void doFitnessCheck(long searchAgentAmount) = 0;
	virtual void randomSearchAgent(float* randomSearchAgentValue) = 0;
	#pragma endregion

private:
	#pragma region Properties
	int size;
	high_resolution_clock::time_point t1;
	high_resolution_clock::time_point t2;
	//Total Generation
	long generation;
	long currentGeneration;
	//Total searchAgent Per Generation After Elimination
	long numberOfSearchAgent;
	//searchAgent Solution in Current Generation
	float* searchAgent;
	float* leader;
	bool* genChangedOption;
	//Stop the Process If The Best Fitness Doesn't Change In 100 Generation
	float lastBestFitness;
	short counterStop;
	//Fitness in Current Generation
	float* fitness;
	float a, A, C, p;
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
	void doLoopInitialization();
	void doCountVariable();
	//All Cuda Operation Inside Class
	void doGPUOperation();
	void updateSearchAgentPosition(float* searchAgent, float* newSearchAgent);
	//Sorting Every searchAgent Based on Its Fitness's
	void doSortingAndAveraging(int fitnessSize);
	//Save History of Best and Average Fitness
	void doSaveHistory();
	void doPrintResults();
	void doFreeMemory();
	//Function to Check Stopping Criteria
	bool checkStoppingCriteria();
	#pragma endregion
};