#include "userClass.h"
#include <string>
#include <sstream>
#include <fstream>

userClass::userClass() : GA_Lib((long)100, 38, 100, 0.15, 0.35,
	CrossoverType::PMXCrossover, MutationType::InversionMutation, SelectionType::TournamentSelection) {
	cudaMallocManaged(&coord, sizeof(float) * 2 * getSize());
	run();
	short* bestChromosome = this->getBestChromosome();
	printf("Best Chromosome:\n");
	for (int i = 0; i < getSize(); i++) {
		printf("%i ", bestChromosome[i]);
	}
	printf("\n");
}

void userClass::doInitialization() {
	short* newChromosome = this->getChromosome();

	//read file
	ifstream file("..\\..\\..\\..\\Resources\\TSP File\\dj38.tsp.txt");
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
	callFitnessCheckGPU_TSPGA(getSize(), chromosomeTemp, fitnessTemp, chromosomeAmount, coord);
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
