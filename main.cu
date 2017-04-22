#include <cmath>
#include <string>
#include <stdio.h>
#define M_PI 3.14159265358978 //Hello !
#define ll long long
#define ull unsigned long long
#define next(N) N*=25214903917;N+=11;N%=281474976710656
#define numBlocks 1024
#define numThreadsPerBlock 256
#define numSeedsPerThread 1
#define search_from 400000000000
#define search_to   1000000000000
#define eye_count 11
using namespace std;

__device__ int getEyesFromChunkseed(ull chunkseed);
__global__ void checkSeeds(ull* start);

int main() {
for(ull i = search_from; i<search_to+numBlocks*numThreadsPerBlock*numSeedsPerThread;i+=numBlocks*numThreadsPerBlock*numSeedsPerThread){
	ull* d_seed;
	cudaMalloc((void**)&d_seed,sizeof(ull));
	cudaMemcpy(d_seed,&i,sizeof(ull),cudaMemcpyHostToDevice);
	checkSeeds<<<numBlocks,numThreadsPerBlock>>>(d_seed);
	cudaDeviceSynchronize();
	cudaFree(d_seed);
}
	return 0;
}
__global__ void checkSeeds(ull* start) {
	ull seed, RNGseed, chunkseed;
	ll var8, var10;
	int baseX, baseZ, chunkX, chunkZ, nbEyes;
	double angle, dist;
	seed=*start+threadIdx.x+blockIdx.x*numThreadsPerBlock;
	for(int i = 0; i<numSeedsPerThread;i++){
		if(seed<search_to){
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
			for (chunkX = min(baseX - 6, baseX + 6); chunkX <= max(baseX - 6, baseX + 6); chunkX++) {
				for (chunkZ = min(baseZ - 6, baseZ + 6); chunkZ <= max(baseZ - 6, baseZ + 6); chunkZ++) {
					chunkseed = (var8 * chunkX + var10 * chunkZ) ^ seed;
					nbEyes = getEyesFromChunkseed(chunkseed);
					if (nbEyes >= eye_count) {
						printf("%llu %d %d %d\n",seed,nbEyes,chunkX,chunkZ);
					}
				}
			}
		}
		seed+=numThreadsPerBlock*numBlocks;
	}
}

__device__ int getEyesFromChunkseed(ull chunkseed) {
	int iEye, nbEyes(0);
	chunkseed = chunkseed ^ 25214903917; //This is the equivalent of starting a new Java RNG
	chunkseed *= 124279299069389; //This line and the one after it simulate 761 calls to next() (761 was determined by CrafterDark)
	chunkseed += 17284510777187;
	chunkseed %= 281474976710656;
	for (iEye = 0; iEye < 12; iEye++) //This is the same as calling nextFloat() 10 times and comparing it to 0.9
		{
		next(chunkseed);
		if (chunkseed <= 253327479039590) {
			nbEyes++;
			if(nbEyes++>12-eye_count){
				return 0;
			}
		}
	}
	return 12-nbEyes;
}
