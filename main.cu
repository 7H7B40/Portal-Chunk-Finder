#include <cmath>
#include <ctime>
#include <string>
#include <stdio.h>
#define M_PI 3.14159265358978 //Hello !
#define ll long long
#define ull unsigned long long
#define next(N) N*=25214903917;N+=11;N%=281474976710656
#define numBlocks 1024
#define numThreadsInBlock 1024
#define search_from 0
#define search_to 50000000000
using namespace std;

__device__ int getEyesFromChunkseed(ull chunkseed);
__global__ void checkSeeds();

int main() {
	checkSeeds<<<numBlocks,numThreadsInBlock>>>();
	cudaDeviceSynchronize();
	return 0;
}
__global__ void checkSeeds() {

	ull stepSize = ((ull)numBlocks)*numThreadsInBlock;
	ull seed, RNGseed, chunkseed;
	ll var8, var10;
	int baseX, baseZ, chunkX, chunkZ, nbEyes;
	double angle, dist;
	for (seed = search_from+blockIdx.x+threadIdx.x*numBlocks; seed <search_to; seed += stepSize) {
		RNGseed = seed ^ 25214903917;
		next(RNGseed);
		var8 = (RNGseed >> 16) << 32;
		angle = (RNGseed / 140737488355328.0) * M_PI;
		next(RNGseed);
		var8 += (int) (RNGseed >> 16); //Don't ask me why there is a conversion to int here, I don't know either.
		var8 = var8 / 2 * 2 + 1;
		next(RNGseed);
		var10 = (RNGseed >> 16) << 32;
		dist = 40 + (RNGseed / 8796093022208.0);
		next(RNGseed);
		var10 += (int) (RNGseed >> 16);
		var10 = var10 / 2 * 2 + 1;
		baseX = round(cos(angle) * dist);
		baseZ = round(sin(angle) * dist);
		for (chunkX = min(baseX - 6, baseX + 6);
				chunkX <= max(baseX - 6, baseX + 6); chunkX++) {
			for (chunkZ = min(baseZ - 6, baseZ + 6);
					chunkZ <= max(baseZ - 6, baseZ + 6); chunkZ++) {
				chunkseed = (var8 * chunkX + var10 * chunkZ) ^ seed;
				nbEyes = getEyesFromChunkseed(chunkseed);
				if (nbEyes >= 11) {
					printf("%d %d %d %d\n",seed,nbEyes,chunkX,chunkZ);
				}
			}
		}
	}
}

__device__ int getEyesFromChunkseed(ull chunkseed) //This function is full of Azelef math magic
		{
	int iEye, nbEyes(0);
	chunkseed = chunkseed ^ 25214903917; //This is the equivalent of starting a new Java RNG
	chunkseed *= 124279299069389; //This line and the one after it simulate 761 calls to next() (761 was determined by CrafterDark)
	chunkseed += 17284510777187;
	chunkseed %= 281474976710656;
	if (chunkseed > 253327479039590) {
		next(chunkseed);
		if (chunkseed > 253327479039590) {
			nbEyes = 2;
			for (iEye = 2; iEye < 12; iEye++) //This is the same as calling nextFloat() 10 times and comparing it to 0.9
					{
				next(chunkseed);
				if (chunkseed > 253327479039590) {
					nbEyes++;
				}
			}
		}
	}
	return nbEyes;
}
