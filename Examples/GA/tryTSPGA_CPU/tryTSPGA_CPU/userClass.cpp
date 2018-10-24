#include "userClass.h"
#include <string>
#include <sstream>
#include <fstream>

userClass::userClass() : GA_Lib((long)100, 38, 100, 0.15, 0.35,
	CrossoverType::PMXCrossover, MutationType::InversionMutation, SelectionType::TournamentSelection) {
	run();
	short* bestChromosome = this->getBestChromosome();
	printf("Best Chromosome:\n");
	for (int i = 0; i < getSize(); i++) {
		printf("%i ",bestChromosome[i]);
	}
	printf("\n");
}

void userClass::doInitialization() {
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
	}
	setChromosome(newChromosome);
}

void userClass::doFitnessCheck(long chromosomeAmount) {
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

void userClass::randomChromosome(short* newChromosome) {
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
