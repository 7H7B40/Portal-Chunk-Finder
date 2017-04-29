#include <cmath>
#include <string>
#include <stdio.h>
#include <time.h>
#define M_PI 3.14159265358978 //Hello !
#define ll long long
#define ull unsigned long long
#define next(N) N*=25214903917;N+=11;N%=281474976710656
#define numBlocks 1024
#define numThreadsPerBlock 256
#define numSeedsPerThread 1
#define search_from 0
#define search_to   1000000000000
#define eye_count 11
using namespace std;
 
__device__ bool getEyesFromChunkseed(ull chunkseed);
__global__ void checkSeeds(ull* start, int* d_sinLUT, int* d_cosLUT);
 
int sinLUT[1024];
int cosLUT[1024];
 
void calculateLUTs(){ //Thanks to jacobsjo for adding a Look-Up Table
	for (int i = 0 ; i< 1024 ; i++){
		sinLUT[i] = round(sin((i* M_PI) / 512.0)*2048);
		cosLUT[i] = round(cos((i* M_PI) / 512.0)*2048);
	}
}
 
int main() {
	time_t timer;
	time(&timer);
	
	int *d_sinLUT, *d_cosLUT;
	ull* d_seed = 0;
	
	calculateLUTs();
	
	cudaMalloc((void**)&d_sinLUT,sizeof(int)*1024);
	cudaMalloc((void**)&d_cosLUT,sizeof(int)*1024);
	cudaMalloc((void**)&d_seed,sizeof(ull));
	
	cudaMemcpy(d_sinLUT,&sinLUT,sizeof(int)*1024,cudaMemcpyHostToDevice);
	cudaMemcpy(d_cosLUT,&cosLUT,sizeof(int)*1024,cudaMemcpyHostToDevice);
	
	for(ull i = search_from; i<search_to+numBlocks*numThreadsPerBlock*numSeedsPerThread;i+=numBlocks*numThreadsPerBlock*numSeedsPerThread){
		cudaMemcpy(d_seed,&i,sizeof(ull),cudaMemcpyHostToDevice);
		checkSeeds<<<numBlocks,numThreadsPerBlock>>>(d_seed, d_sinLUT, d_cosLUT);
		cudaDeviceSynchronize();
	}
	
	cudaFree(d_seed);
	cudaFree(d_sinLUT);
	cudaFree(d_cosLUT);
	
	time_t endtime;
	time(&endtime);
	
	double seconds = difftime(timer,endtime);
	printf("%.f seconds to calculate",seconds);
	return 0;
}
__global__ void checkSeeds(ull* start, int* d_sinLUT, int* d_cosLUT) { // This function is full of Azelef math magic
	ull seed, RNGseed, chunkseed,initialxor;
	ll var8, var10;
	int baseX, baseZ, chunkX, chunkZ, angle;
	double dist;
	seed=*start+threadIdx.x+blockIdx.x*numThreadsPerBlock;
	for(int i = 0; i<numSeedsPerThread;i++){
		if(seed<search_to){
			RNGseed = seed ^ 25214903917;
			next(RNGseed);
			var8 = (RNGseed >> 16) << 32;
			angle = RNGseed/274877906944;
			next(RNGseed);
			var8 += (int) (RNGseed >> 16); //Don't ask me why there is a conversion to int here, I don't know either.
			var8 = var8 / 2 * 2 + 1;
			next(RNGseed);
			var10 = (RNGseed >> 16) << 32;
			dist = 160+(RNGseed/2199023255552);
			next(RNGseed);
			var10 += (int) (RNGseed >> 16);
			var10 = var10 / 2 * 2 + 1;
			baseX = (*(d_cosLUT+angle) * dist) / 8192;
			baseZ = (*(d_sinLUT+angle) * dist) / 8192;
			initialxor = seed ^ 25214903917;
			for (chunkX = baseX - 6; chunkX <= baseX + 6; chunkX++) {
				for (chunkZ =baseZ - 6; chunkZ <= baseZ + 6; chunkZ++) {
					chunkseed = (var8 * chunkX + var10 * chunkZ) ^ initialxor;
					if (getEyesFromChunkseed(chunkseed)) {
						printf("%llu %d %d %d\n",seed,eye_count,chunkX,chunkZ);
					}
				}
			}
		}
		seed+=numThreadsPerBlock*numBlocks;
	}
}
 
__device__ bool getEyesFromChunkseed(ull chunkseed) {
	//chunkseed = chunkseed ^ 25214903917; //This is the equivalent of starting a new Java RNG // this was moved to the checkSeeds method so it's only computed once
	chunkseed *= 124279299069389; //This line and the one after it simulate 761 calls to next() (761 was determined by CrafterDark)
	chunkseed += 17284510777187;
	chunkseed %= 281474976710656;
	
	//Xero's branch-reduced code
	//This code is better than Azelef's because it gets all the 11-eye ones
	//instead of discarding a sixth of the seeds
	
	
	int failcount = (chunkseed <= 253327479039590);//eye 1
	//if(failcount > 12-eye_count)return false; //commenting out eye 1, 10, 11 is optimal for
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 2
	if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 3
	if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 4
	if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 5
	if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 6
	if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 7
	if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 8
	if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 9
	if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 10
	//if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 11
	//if(failcount > 12-eye_count)return false;
	
	next(chunkseed);
	failcount += (chunkseed <= 253327479039590);//eye 12
	return failcount == 12-eye_count;
}
