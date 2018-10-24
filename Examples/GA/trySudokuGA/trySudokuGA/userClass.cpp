#include "userClass.h"
#include "userFitness.h"
#include <string>
#include <sstream>
#include <fstream>
#include <Windows.h>

userClass::userClass() : GA_Lib((long)100, 81, 100, 0.15, 0.35,
	CrossoverType::OnePointCrossover, MutationType::RandomResetting, SelectionType::RankSelection) {
	run();
	int* bestChromosome = this->getBestChromosome();
	printf("Best Chromosome:\n");
	for (int i = 0; i < getSize(); i++) {
		if (i % 9 == 0) printf("\n");
		printf("%i ", bestChromosome[i]);
	}
	printf("\n");
}

void userClass::doInitialization() {
	problem = new int[getSize()];
	ifstream file;
	file.open("..\\..\\..\\..\\Resources\\Sudoku File\\Sudoku Easy.txt");
	int i = 0;
	while (i < getSize() && file >> problem[i]) i++;
	file.close();
	int* newChromosome = this->getChromosome();
	for (int i = 0; i < getChromosomePerGeneration(); i++)
	{
		int* a = new int[getSize()];
		for (int j = 0; j < 81; j++) a[j] = problem[j];
		randomChromosome(a);
		for (int j = 0; j < 81; j++) newChromosome[(i * 81) + j] = a[j];
		delete a;
	}
	setChromosome(newChromosome);
	delete problem;
}

void userClass::setGenChangedOption(bool* genChangedOption) {
	for (int i = 0; i < 81; i++) {
		if (problem[i] == 0) genChangedOption[i] = true;
		else genChangedOption[i] = false;
	}
}

void userClass::doFitnessCheck(long chromosomeAmount) {
	int* chromosomeTemp = this->getChromosome();
	float* fitnessTemp = this->getFitness();
	callFitnessCheckGPU_SudokuGA(81, chromosomeTemp, fitnessTemp, chromosomeAmount);
};

void userClass::randomChromosome(int* newChromosome) {
	for (int i = 0; i < 9; i++)
	{
		int temp[] = { 1,2,3,4,5,6,7,8,9 };
		for (int j = 0; j < 100; j++)
		{
			int index1 = rand() % 9;
			int index2 = rand() % 9;
			int temp2 = temp[index1];
			temp[index1] = temp[index2];
			temp[index2] = temp2;
		}
		for (int j = 0; j < 9; j++)
		{
			if (newChromosome[i * 9 + j] == 0) newChromosome[i * 9 + j] = temp[j];
		}
	}
};