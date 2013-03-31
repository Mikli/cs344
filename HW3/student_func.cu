/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

__global__ void find_max_min(const float* const logLuminance,
							 float* const maxs_per_block,
							 float* const mins_per_block,
							 const size_t numRows,
							 const size_t numCols)
{
	extern __shared__ float this_block[];
	float *this_block_max = &this_block[0];
	float *this_block_min = &this_block[blockDim.x];
	int col = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;

	if (col >= numRows * numCols)
		return;

	this_block_max[tid] = this_block_min[tid] = logLuminance[col];
	
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s>>=1)
	{
		if (tid < s)
		{
			this_block_max[tid] = fmaxf(this_block_max[tid],this_block_max[tid+s]);
			this_block_min[tid] = fminf(this_block_min[tid],this_block_min[tid+s]);
		}
		__syncthreads();
	}

	// after previous alg, the result is in the first index
	if (tid == 0)
	{
		maxs_per_block[blockIdx.x] = this_block_max[0];
		mins_per_block[blockIdx.x] = this_block_min[0];
	}


}

__global__ void reduce_max_linear(float* const input_of_maxs)
{
	extern __shared__ float block_memory[];
	int tid = threadIdx.x;
	unsigned int size = blockDim.x;

	if (size % 2 != 0)
	{
		size++;
	}

	block_memory[tid] = input_of_maxs[tid];

	__syncthreads();

	for (unsigned int s = size / 2; s > 0; s>>=1)
	{
		if (tid < s)
		{
			block_memory[tid] = fmaxf(block_memory[tid],block_memory[tid+s]);
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		input_of_maxs[0] = block_memory[0];
	}
}

__global__ void reduce_min_linear(float* const input_of_mins)
{
	extern __shared__ float block_memory[];
	int tid = threadIdx.x;
	unsigned int size = blockDim.x;

	if (size % 2 != 0)
	{
		size++;
	}
	block_memory[tid] = input_of_mins[tid];
	
	__syncthreads();

	for (unsigned int s = size / 2; s > 0; s>>=1)
	{
		if (tid < s)
		{
			block_memory[tid] = fminf(block_memory[tid],block_memory[tid+s]);
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		input_of_mins[0] = block_memory[0];
	}
}

float *d_maxs_for_blocks, *d_mins_for_blocks;

#include<stdio.h>

static float find_max_ref(const float* const tab, unsigned int no_of_elem)
{
	float m = 0.0;
	unsigned int max_index = 0;

	for(unsigned int i = 0; i<no_of_elem; i++)
	{
		if (tab[i] > m) m = tab[i];
		max_index = i;
	}
	return m;
}

static float find_min_ref(const float* const tab, unsigned int no_of_elem)
{
	float m = 0.0;
	unsigned int max_index = 0;

	for(unsigned int i = 0; i<no_of_elem; i++)
	{
		if (tab[i] < m)
		{
			m = tab[i];
			max_index = i;
		}
	}
	return m;
}

__global__ void make_histo(unsigned int* const d_hist,
						   const float* const d_logLuminance,
						   const float lumMin,
						   const float lumRange,
						   const size_t numBins,
						   const size_t numRows,
						   const size_t numCols)
{
extern __shared__ unsigned int block_bins[];
	
	int col = threadIdx.x + blockDim.x * blockIdx.x;
		
	if (col >= (numRows * numCols))
		return;

  	unsigned int bins_per_thread = numBins/blockDim.x;
	
    unsigned int i = threadIdx.x * bins_per_thread;

    while (i < ((threadIdx.x + 1) * bins_per_thread))
	{
        block_bins[i] = 0;
        i++;
    }

    __syncthreads();

	unsigned int bin = ((d_logLuminance[col] - lumMin) / lumRange) * numBins - 1;

	atomicAdd(&(block_bins[bin]),1);
	
	__syncthreads();
	i = threadIdx.x * bins_per_thread;
	/* now block_bins is ready, count how many threads we have and let them add to global bins */
	while (i < ((threadIdx.x + 1) * bins_per_thread))
	{
        atomicAdd(&(d_hist[i]),block_bins[i]);
        i++;
    }
	
}

__global__ void scan_hillis_steele(const unsigned int* const d_histogram,
                               unsigned int* const d_cdf, 
                               const size_t cdf_size)
{
    extern __shared__ unsigned int scan[];
    unsigned int offset;
    unsigned int steps = log2((float)cdf_size);

    unsigned int col = threadIdx.x;

    scan[col] = d_histogram[col];
    
    __syncthreads();       
    unsigned int new_value = 0;
    for(unsigned int i = 0; i < steps; i++)
    {
       offset = 1 << i;
       if (col < offset)
       {
           //just get the same value
           new_value = scan[col];
       }
       else
       {
           new_value = scan[col] + scan[col-offset];
       }   
       __syncthreads();
       
       scan[col] = new_value;
       __syncthreads();
    }

    // make it inclusive
    d_cdf[col] = scan[col] - d_histogram[col];
}




bool testHisto(const unsigned int* const histo, size_t len, unsigned int expected_value)
{
	unsigned int sum = 0;
	for(unsigned int s = 0; s < len; s++)
		sum += histo[s];

	if (sum == expected_value)
		return true;
	else
		return false;

}


void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{

	size_t BLOCK_WIDTH = 512 ;//* filterWidth;
	size_t BLOCK_HEIGHT = 1 ;//* filterWidth;
	
	const dim3 blockSize(BLOCK_WIDTH,BLOCK_HEIGHT,1);
	const dim3 gridSize(numRows * numCols / BLOCK_WIDTH + 1, 1, 1);

	float *maxs_for_blocks = new float[gridSize.x];
	float *mins_for_blocks = new float[gridSize.x];


	//float *logLum = new float[numRows * numCols];

	unsigned int *d_histogram;
	//unsigned int *h_histogram = new unsigned int[numBins];

	//checkCudaErrors(cudaMemcpy(logLum,d_logLuminance,sizeof(float) * numRows * numCols,cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMalloc(&d_maxs_for_blocks,  sizeof(float) * gridSize.x));
	checkCudaErrors(cudaMemset(d_maxs_for_blocks, 0, sizeof(float) * gridSize.x));
	checkCudaErrors(cudaMalloc(&d_mins_for_blocks,  sizeof(float) * gridSize.x));
	checkCudaErrors(cudaMemset(d_mins_for_blocks, 0, sizeof(float) * gridSize.x));

	checkCudaErrors(cudaMalloc(&d_histogram,  sizeof(unsigned int) * numBins));
	checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int) * numBins));


	//printf("%d, %d, %d\n", blockSize.x, blockSize.y, blockSize.z);
	//printf("%d, %d, %d\n", gridSize.x, gridSize.y, gridSize.z);

	//float float_min = find_min_ref(logLum, numRows * numCols);
    //float float_max = find_max_ref(logLum, numRows * numCols);

	find_max_min<<<gridSize,blockSize, sizeof(float) * BLOCK_WIDTH * 2>>>(d_logLuminance, d_maxs_for_blocks, d_mins_for_blocks, numRows, numCols);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	//checkCudaErrors(cudaMemcpy(maxs_for_blocks,d_maxs_for_blocks,sizeof(float) * gridSize.x,cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(mins_for_blocks,d_mins_for_blocks,sizeof(float) * gridSize.x,cudaMemcpyDeviceToHost));

	//printf("%f, %f\n", find_min_ref(mins_for_blocks, gridSize.x),find_max_ref(maxs_for_blocks, gridSize.x));

	// threads are as many as blocks were
	int threads_for_reduce = numRows * numCols / BLOCK_WIDTH + 1;

	reduce_max_linear<<<1,threads_for_reduce,sizeof(float) * threads_for_reduce + 4>>>(d_maxs_for_blocks);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaMemcpy(maxs_for_blocks,d_maxs_for_blocks,sizeof(float) * gridSize.x,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&max_logLum,d_maxs_for_blocks,sizeof(float),cudaMemcpyDeviceToHost));

	reduce_min_linear<<<1,threads_for_reduce,sizeof(float) * threads_for_reduce + 4>>>(d_mins_for_blocks);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaMemcpy(mins_for_blocks,d_mins_for_blocks,sizeof(float) * gridSize.x,cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(&min_logLum,d_mins_for_blocks,sizeof(float),cudaMemcpyDeviceToHost));
    
	//min_logLum = mins_for_blocks[0];
	//max_logLum = maxs_for_blocks[0];

	float lumRange = max_logLum - min_logLum;

	make_histo<<<gridSize,blockSize,sizeof(unsigned int) * numBins>>>(d_histogram,
	                d_logLuminance, min_logLum, lumRange, numBins, numRows, numCols);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaMemcpy(h_histogram,d_histogram,sizeof(unsigned int) * numBins,cudaMemcpyDeviceToHost));
	//testHisto(h_histogram, numBins, numRows * numCols);
	//printf("%f, %f\n", maxs_for_blocks[0], mins_for_blocks[0]);
	
    scan_hillis_steele<<<1,numBins,sizeof(unsigned int) * numBins>>>(d_histogram,
                                                                    d_cdf,
                                                                    numBins);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    //checkCudaErrors(cudaMemcpy(h_histogram,d_cdf,sizeof(unsigned int) * numBins,cudaMemcpyDeviceToHost));
	//testHisto(h_histogram, numBins, numRows * numCols);

	//TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */




  /****************************************************************************
  * You can use the code below to help with debugging, but make sure to       *
  * comment it out again before submitting your assignment for grading,       *
  * otherwise this code will take too much time and make it seem like your    *
  * GPU implementation isn't fast enough.                                     *
  *                                                                           *
  * This code generates a reference cdf on the host by running the            *
  * reference calculation we have given you.  It then copies your GPU         *
  * generated cdf back to the host and calls a function that compares the     *
  * the two and will output the first location they differ.                   *
  * ************************************************************************* */

  /*float *h_logLuminance = new float[numRows * numCols];
  unsigned int *h_cdf   = new unsigned int[numBins];
  unsigned int *h_your_cdf = new unsigned int[numBins];
  checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance, numCols * numRows * sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_your_cdf, d_cdf, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

  referenceCalculation(h_logLuminance, h_cdf, numRows, numCols, numBins);

  //compare the results of the CDF
  checkResultsExact(h_cdf, h_your_cdf, numBins);
 
  delete[] h_logLuminance;
  delete[] h_cdf; 
  delete[] h_your_cdf;*/

	delete[] maxs_for_blocks;
	delete[] mins_for_blocks;
	checkCudaErrors(cudaFree(d_maxs_for_blocks));
	checkCudaErrors(cudaFree(d_mins_for_blocks));
}
