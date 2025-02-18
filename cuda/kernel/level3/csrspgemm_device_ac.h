#ifndef CSRSPGEMM_DEVICE_AC
#define CSRSPGEMM_DEVICE_AC

#include "alphasparse.h"
#include "alphasparse/types.h" 
#include <bitset>
#include <memory>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define LZCNT __builtin_clzll

#include "ac/MultiplyKernels.h"
#include "ac/consistent_gpu_memory.h"
#include "ac/consistent_memory.h"
#include "ac/memory.h"
#include "ac/stream.h"
#include "ac/MergeCaseOffsets.h"
#include "ac/meta_utils.h"
#include "ac/Multiply.h"
#include "ac/acSpGEMM_DetermineBlockStarts.cuh"
#include "ac/acSpGEMM_SpGEMM.cuh"
#include "ac/acSpGEMM_MergeSimple.cuh"
#include "ac/acSpGEMM_MergeMaxChunks.cuh"
#include "ac/acSpGEMM_MergeGeneralized.cuh"
#include "ac/acSpGEMM_ChunksToCSR.cuh"
#include "ac/HelperFunctions.cuh"
#include "ac/CustomExceptions.h"
#include "ac/default_scheduling_traits.h"

void startTimer(cudaEvent_t& start, CUstream stream = 0)
{
	HANDLE_ERROR(cudaEventRecord(start, stream));
}

float recordTimer(cudaEvent_t& start, cudaEvent_t& end, CUstream stream = 0)
{
	float time;
	HANDLE_ERROR(cudaEventRecord(end, stream));
	HANDLE_ERROR(cudaEventSynchronize(end));
	HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));
	return time;
}

namespace CU
{
	unique_ptr allocMemory(std::size_t size)
	{
		CUdeviceptr ptr;
		cudaMalloc(reinterpret_cast<void**>(&ptr), size);
		return unique_ptr(ptr);
	}
	
	unique_ptr allocMemoryPitched(std::size_t& pitch, std::size_t row_size, std::size_t num_rows, unsigned int element_size)
	{
		CUdeviceptr ptr;
		cudaMallocPitch(reinterpret_cast<void**>(&ptr), &pitch, row_size, num_rows);
		return unique_ptr(ptr);
	}
	
	pitched_memory allocMemoryPitched(std::size_t row_size, std::size_t num_rows, unsigned int element_size)
	{
		CUdeviceptr ptr;
		std::size_t pitch;
		cudaMallocPitch(reinterpret_cast<void**>(&ptr), &pitch, row_size, num_rows);
		return pitched_memory(unique_ptr(ptr), pitch);
	}
}

using OffsetType = uint32_t;

namespace ACSpGEMM {
	template <typename IndexType, typename DataType, uint32_t threads, uint32_t blocks_per_mp, uint32_t nnz_per_thread, uint32_t input_elements_per_thread, uint32_t retain_elements_per_thread, uint32_t merge_max_chunks, uint32_t generalized_merge_max_path_options, uint32_t merge_max_path_options, bool DEBUG_MODE>
	void MultiplyImplementation(
								alphasparseHandle_t handle,
								alphasparseOperation_t opA,
								alphasparseOperation_t opB,
								const DataType alpha,
								alphasparseSpMatDescr_t matA,
								alphasparseSpMatDescr_t matB,
								const DataType beta,
								alphasparseSpMatDescr_t matC,
								char * externalBuffer2,
								const GPUMatrixMatrixMultiplyTraits& traits)
	{
		using ConsistentGPUMemory = ConsistentMemory<MemorySpace::device>;
		
		// the magic numbers to make it run smoother
		const float OverallocationFactor = 1.25f;
		const int ChunkPointerOverestimationFactor = 4;
		const float ChunkOverallocationFactor = 1.0f;
		using UintBitSet = std::bitset<sizeof(uint32_t)>;

		if(DEBUG_MODE)
		{
			std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
			std::cout << "THREADS: " << threads << " | NNZPerThread: " << nnz_per_thread << " | InputElementsPerThreads: " << input_elements_per_thread << " | RetainElementsPerThreads: " << retain_elements_per_thread;
			std::cout << " | MaxChunks: " << merge_max_chunks << " | MergePathOptions: " << merge_max_path_options << "| ChunkpointerOverestimationFactor: " << ChunkPointerOverestimationFactor << "\n";
			std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
		}

		// Helper variables
		size_t memory_usage_in_Bytes{ 0 };
		const size_t chunckAllocationsSize{ 256 };
		const size_t numFlags{ 128 };
		const size_t numCounters{ 3 };
		const size_t mergeTypeCounters{ 4 };
		static size_t maxExpectedNNZ{ 500000000 }; //limit allocation...
		static size_t minExpectedNNZ{ 10000000 }; //limit allocation...
		static float lastChunckBufferRequirementRatio{ 1.0f };
		const uint32_t nnzperblock{ threads * nnz_per_thread };
		size_t run{ 0 }, chunk_pointer_restart_run{ 0 };
		bool completed{ false };
		bool rowmerging{ false };
		MergeCaseOffsets mergeBlocks;
		uint32_t* currentCounters, *currentChunckAllocation, *currentFlag;
		uint32_t numSharedRows;
		size_t size_to_allocate;
		size_t upper_limit{ 3LL * 1024 * 1024 * 1024 };

		// Kernels
		AcSpGEMMKernels spgemm(threads);

		// Matrix information
		size_t Arows = matA->rows;
		size_t Acols = matA->cols;
		size_t Brows = matB->rows;
		size_t Bcols = matB->cols;
		size_t Crows = Arows;
		size_t Ccols = Bcols;

		if (Acols != Brows)
			throw std::runtime_error("Unable to multiply matrix with matrix - invalid dimensions");

		// Matrix Output estimation
		double a_avg_row = matA->nnz / static_cast<double>(Arows);
		double b_avg_row = matB->nnz / static_cast<double>(Brows);
		double avg_row_overlap = b_avg_row / Bcols;
		// note geometric sequence
		double output_estimate = OverallocationFactor*Arows*b_avg_row * (1.0 - pow(1.0 - avg_row_overlap, a_avg_row)) / (avg_row_overlap);

		// chunks might get created earlier
		double single_chunk_estimate = b_avg_row;
		double current_overlap = avg_row_overlap;
		double merges;
		for (merges = 1; merges < static_cast<size_t>(a_avg_row + 1.0); ++merges)
		{
			if (single_chunk_estimate >= retain_elements_per_thread*threads)
				break;
			single_chunk_estimate += (1 - current_overlap)*b_avg_row;
			current_overlap = current_overlap + (1 - current_overlap)*avg_row_overlap;
		}
		double intermediate_estimate = OverallocationFactor * a_avg_row / std::min(merges, a_avg_row) * single_chunk_estimate * Arows;
		double mergepointer_estimate = std::max(intermediate_estimate, output_estimate) / (retain_elements_per_thread*threads) + 16 * 1024;
		size_t expectedNNZ = std::max(minExpectedNNZ, std::min(maxExpectedNNZ, static_cast<size_t>(lastChunckBufferRequirementRatio*std::max(intermediate_estimate, output_estimate))));
		size_to_allocate = (sizeof(DataType) + sizeof(IndexType))*expectedNNZ*ChunkOverallocationFactor;
		size_t free, total;
		cudaMemGetInfo(&free, &total);
		upper_limit = std::min(upper_limit, free / 3);
		if (size_to_allocate > upper_limit)
			size_to_allocate = upper_limit;
		if(DEBUG_MODE)
		{
			std::cout << "A: " << Arows << "x" << Acols << " NNZ: " << matA->nnz << " avg row: " << a_avg_row << "  " << "B: " << Brows << "x" << Bcols << " NNZ: " << matB->nnz << " avg row: " << b_avg_row << "\n";
			std::cout << "expected row overlap: " << avg_row_overlap << " overallocation: " << OverallocationFactor << "\n";
			std::cout << "expected nnz: " << static_cast<size_t>(round(output_estimate)) << " expected temp: " << static_cast<size_t>(round(intermediate_estimate)) << " mem alloc: " << expectedNNZ << "\n";
			std::cout << "mergepointer alloc " << static_cast<size_t>(ChunkPointerOverestimationFactor*mergepointer_estimate) << " mergepointer estimate: " << mergepointer_estimate << "\n";
		}

		// CUDA variables
		cudaStream_t stream = handle->streams[0];;
		int blockSize = 256;
		int gridSize(divup<int>(Arows + 1, blockSize));
		const int number_merge_streams = 3;
		cudaStream_t mergeStreams[number_merge_streams];
		for (int i = 0; i < number_merge_streams; ++i)
		{
			// if(stats.measure_all)
			// 	mergeStreams[i] = stream;
			// else
			mergeStreams[i] = handle->streams[i+1];
				// cudaStreamCreate(&mergeStreams[i]);
		}

		// cudaEvent_t ce_start, ce_stop, individual_start, individual_stop;
		// cudaEventCreate(&ce_start); cudaEventCreate(&ce_stop); cudaEventCreate(&individual_start); cudaEventCreate(&individual_stop);

		// GPU Memory Helper structures - general
		static ConsistentGPUMemory chunckPointers;
		static ConsistentGPUMemory combinedGeneralMemory;
		static ConsistentGPUMemory chunk_counter_cptr;
		uint32_t* chunckAllocations{ nullptr };
		uint32_t* blockStarts{ nullptr };
		uint32_t* sharedRowTracker{ nullptr };
		void** outputRowListHead{ nullptr };
		uint32_t* outputRowChunkCounter{ nullptr };
		uint32_t* completion_status{ nullptr };
		uint32_t* chunk_counter{ nullptr };
		void* prefixSumTemp{ nullptr };

		// GPU Memory Helper structures - merge stage allocation
		static ConsistentGPUMemory combineBlockOffsets; // SIZE: combineBlockOffsetsSize * sizeof(IndexType)

		static ConsistentGPUMemory chunk_indices_cptr; // SIZE:  ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * 8
		static ConsistentGPUMemory chunk_values_cptr; // SIZE: ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * 8
		static ConsistentGPUMemory chunk_multiplier_cptr; // SIZE: ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * 8

		static ConsistentGPUMemory combinedMergeStageMemory;
		static uint32_t* shared_rows_handled{ nullptr };
		static uint32_t* restart_completion{ nullptr };
		static uint32_t* chunkElementConsumedAndPath{ nullptr };
		uint32_t* num_chunks{ nullptr };
		uint32_t* chunkElementCountDataOffset{ nullptr };
		uint32_t* sample_offset{ nullptr };
		static IndexType** chunk_indices{ nullptr };
		static DataType** chunk_values{ nullptr };
		static DataType* chunk_multiplier{ nullptr };
		

		// CPU Memory Helper structures
		static RegisteredMemoryVar<size_t> chunkPointerSize(0);
		static RegisteredMemoryVar<size_t> outputRowInfoSize(0);
		static RegisteredMemoryVar<size_t> prefixSumTempMemSize;
		static RegisteredMemoryVar<size_t> combineBlockOffsetsSize(0);
		static RegisteredMemoryVar<size_t> mergeBlocksAlloc(0);
		static RegisteredMemoryVar<size_t> lastSharedRows(0);
		static RegisteredMemoryVar<size_t> merge_simple_rows(0);
		static RegisteredMemoryVar<size_t> merge_max_chunks_rows(0);
		static RegisteredMemoryVar<size_t> merge_generalized_rows(0);
		uint32_t flagsAndListAllocCounters[numFlags + numCounters];
		size_t tempChunkBufferSizes[256];
		CU::unique_ptr tempChunkBuffers[256];
		tempChunkBufferSizes[0] = alignment(size_to_allocate, 16);
		//
		// TSOPF_RS_b300_c2.mtx shows very weird results if this is done here??
		//
		// Allocate temporary memory for chunks
		tempChunkBuffers[0] = CU::allocMemory(tempChunkBufferSizes[0]);
		
		cudaDeviceSynchronize();
		// ##############################
		// startTimer(ce_start, stream);
		// ##############################
		// if(stats.measure_all)
		// 	startTimer(individual_start, stream);
		

		// Allocate memory for block offsets
		uint32_t requiredBlocks = divup<uint32_t>(matA->nnz, nnzperblock);

		// Allocate memory for chunk and shared row tracker
		if (outputRowInfoSize < Crows)
		{
			//----------------------------------------------------------
			prefixSumTempMemSize = spgemm.tempMemSize<IndexType>(Crows);
			//----------------------------------------------------------
			outputRowInfoSize = Crows;
		}

		// Allocate combined general memory
		size_t combinedGeneralMemory_size =
			/*chunckAllocations*/alignment((chunckAllocationsSize + numFlags + numCounters + mergeTypeCounters) * sizeof(uint32_t), 8) +
			/*blockStarts*/ alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) +
			/*completion_status*/ alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) +
			///*chunk_counter*/ alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) +
			/*outputRowListHead*/ alignment(Crows * sizeof(void*), 8) +
			/*outputRowChunkCounter*/ alignment(Crows * sizeof(uint32_t), 8) +
			/*sharedRowTracker*/ alignment(Crows * sizeof(uint32_t), 8) +
			/*prefixSumTemp*/ alignment(static_cast<size_t>(prefixSumTempMemSize), 8);
		combinedGeneralMemory.assure(combinedGeneralMemory_size);
		memory_usage_in_Bytes += combinedGeneralMemory_size;

		// Place pointers in correct positions
		outputRowListHead = combinedGeneralMemory.get<void*>();
		chunckAllocations = reinterpret_cast<uint32_t*>(outputRowListHead + (alignment(Crows * sizeof(void*), 8) / sizeof(void*)));
		completion_status = chunckAllocations + alignment((chunckAllocationsSize + numFlags + numCounters + mergeTypeCounters) * sizeof(uint32_t), 8) / sizeof(uint32_t);
		/*chunk_counter = completion_status + (alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) / sizeof(uint32_t));*/
		blockStarts = completion_status + (alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) / sizeof(uint32_t));
		outputRowChunkCounter = blockStarts + (alignment((requiredBlocks + 2) * sizeof(uint32_t), 8) / sizeof(uint32_t));
		sharedRowTracker = outputRowChunkCounter + (alignment(Crows * sizeof(uint32_t), 8) / sizeof(uint32_t));
		prefixSumTemp = reinterpret_cast<void*>(sharedRowTracker + (alignment(Crows * sizeof(uint32_t), 8) / sizeof(uint32_t)));

		// TODO: Move back in, currently sometimes produces crashes for whatever reason
		chunk_counter_cptr.assure((requiredBlocks + 2) * sizeof(uint32_t));
		chunk_counter = chunk_counter_cptr.get<uint32_t>();
		
		// Allocate memory for chunk pointers
		size_t targetChunkPointerSize = ChunkPointerOverestimationFactor*mergepointer_estimate;
		if (chunkPointerSize < targetChunkPointerSize)
		{
			chunkPointerSize = targetChunkPointerSize;
			chunckPointers.assure((targetChunkPointerSize) * sizeof(void*));
			memory_usage_in_Bytes += (targetChunkPointerSize) * sizeof(void*);
		}

		// Allocate memory for offsets
		CU::unique_ptr newmat_offsets;
		if (matC->rows != Crows)
		{
			newmat_offsets = CU::allocMemory((Crows + 1) * sizeof(OffsetType));
			memory_usage_in_Bytes += (Crows + 1) * sizeof(OffsetType);
		}
		else
		{
			newmat_offsets.consume(reinterpret_cast<CUdeviceptr>((IndexType*)matC->row_data));
			matC->row_data = nullptr;
		}

		spgemm.setLaunchDimensions(gridSize, stream, blockSize);
		//----------------------------------------------------------
		spgemm.h_DetermineBlockStarts<OffsetType, threads*nnz_per_thread>(
			Arows,
			reinterpret_cast<IndexType*>(matA->row_data),
			blockStarts,
			reinterpret_cast<uint64_t*>(outputRowListHead),
			outputRowChunkCounter,
			newmat_offsets.get<uint32_t>(),
			requiredBlocks,
			completion_status,
			(chunckAllocationsSize + numFlags + numCounters + mergeTypeCounters),
			chunckAllocations,
			(lastSharedRows),
			shared_rows_handled,
			restart_completion,
			chunk_counter,
			(lastSharedRows) * (generalized_merge_max_path_options + helper_overhead),
			chunkElementConsumedAndPath
			);
		//----------------------------------------------------------
		// if(stats.measure_all)
		// 	stats.duration_blockstarts = recordTimer(individual_start, individual_stop, stream);

		do
		{
			currentChunckAllocation = chunckAllocations + (2 * run);
			currentFlag = chunckAllocations + (chunckAllocationsSize + run + chunk_pointer_restart_run);
			currentCounters = chunckAllocations + (chunckAllocationsSize + numFlags);
			if (!rowmerging)
			{
				if(DEBUG_MODE)
				{
					std::cout << "################################################\n";
					std::cout << "Start spgemm stage with " << requiredBlocks<<  " and run: " << run << "\n";
				}
				// if(stats.measure_all)
				// 	startTimer(individual_start, stream);

				// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
				// Stage 2 - Compute SpGEMM
				// $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
				spgemm.setLaunchDimensions(requiredBlocks, stream, threads);
				if (Arows < 0x10000 && Bcols < 0x10000)
				{
				if(DEBUG_MODE)
				{
					std::cout << "Case 1:\n";
				}
					//we can just use 16bit
					//----------------------------------------------------------
					spgemm.h_computeSpgemmPart<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_path_options, DataType, DataType, DataType, IndexType, OffsetType, 0>(
						(DataType*)matA->val_data, (IndexType*)matA->col_data, (IndexType*)matA->row_data,
						(DataType*)matB->val_data, (IndexType*)matB->col_data, (IndexType*)matB->row_data, alpha,
						blockStarts, matA->nnz, Arows,
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, currentChunckAllocation + 1, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						newmat_offsets.get<OffsetType>(), outputRowListHead, outputRowChunkCounter,
						sharedRowTracker, currentCounters + 1, avg_row_overlap, 1.0f / avg_row_overlap,
						currentFlag, completion_status, chunk_counter, currentCounters + 2);
					//----------------------------------------------------------
				}
				else if (Bcols < (1 << LZCNT(nnz_per_thread*threads)) - 1)
				{
					if(DEBUG_MODE)
					{
						std::cout << "Case 2:\n";
					}
					//remap every local row to reduce bit count and use remaining for col ids
					//----------------------------------------------------------
					spgemm.h_computeSpgemmPart<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_path_options, DataType, DataType, DataType, IndexType, OffsetType, 1>(
						(DataType*)matA->val_data, (IndexType*)matA->col_data, (IndexType*)matA->row_data,
						(DataType*)matB->val_data, (IndexType*)matB->col_data, (IndexType*)matB->row_data, alpha,
						blockStarts, matA->nnz, Arows,
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, currentChunckAllocation + 1, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						newmat_offsets.get<OffsetType>(), outputRowListHead, outputRowChunkCounter,
						sharedRowTracker, currentCounters + 1, avg_row_overlap, 1.0f / avg_row_overlap,
						currentFlag, completion_status, chunk_counter, currentCounters + 2);
					//----------------------------------------------------------
				}
				else
				{
					if(DEBUG_MODE)
					{
						std::cout << "Case 3:\n";
					}
					//----------------------------------------------------------
					spgemm.h_computeSpgemmPart<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_path_options, DataType, DataType, DataType, IndexType, OffsetType, 2>(
						(DataType*)matA->val_data, (IndexType*)matA->col_data, (IndexType*)matA->row_data,
						(DataType*)matB->val_data, (IndexType*)matB->col_data, (IndexType*)matB->row_data, alpha,
						blockStarts, matA->nnz, Arows,
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, currentChunckAllocation + 1, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						newmat_offsets.get<OffsetType>(), outputRowListHead, outputRowChunkCounter,
						sharedRowTracker, currentCounters + 1, avg_row_overlap, 1.0f / avg_row_overlap,
						currentFlag, completion_status, chunk_counter, currentCounters + 2);
					//----------------------------------------------------------
				}
				// if (cudaDeviceSynchronize() != cudaSuccess) {
				// 	throw SpGEMMException();
				// }
				// if(stats.measure_all)
				// 	stats.duration_spgemm += recordTimer(individual_start, individual_stop, stream);
			}
			else
			{
				if(DEBUG_MODE)
				{
					std::cout << "################################################\n";
					std::cout << "Start Merge Stage\n";
				}
				uint32_t simple_restart_offset = 0;
				uint32_t max_chunks_restart_offset = mergeBlocks.shared_rows_simple;
				uint32_t generalized_restart_offset = mergeBlocks.shared_rows_simple + mergeBlocks.shared_rows_max_chunks;
				// Simple Case -> Output fits in shared
				if (mergeBlocks.shared_rows_simple)
				{
					// if(stats.measure_all)
					// 	startTimer(individual_start, mergeStreams[0]);

					spgemm.setLaunchDimensions(mergeBlocks.shared_rows_simple, mergeStreams[0], threads);
					if (Bcols < 1 << LZCNT(threads - 1))
					{
						if (DEBUG_MODE)
						{
							std::cout << "Case: 1\n";
						}
						//----------------------------------------------------------
						spgemm.h_mergeSharedRowsSimple< nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_chunks, merge_max_path_options, DataType, IndexType, OffsetType, false>(
							combineBlockOffsets.get<uint32_t>() + (3 * numSharedRows), combineBlockOffsets.get<uint32_t>(), outputRowListHead,
							newmat_offsets.get<OffsetType>(), alpha,
							tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, NULL, tempChunkBufferSizes[run],
							chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
							currentFlag, restart_completion, shared_rows_handled, simple_restart_offset, currentCounters + 2
							);
						//----------------------------------------------------------
					}
					else
					{
						if (DEBUG_MODE)
						{
							std::cout << "Case: 2\n";
						}
						//----------------------------------------------------------
						spgemm.h_mergeSharedRowsSimple< nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_chunks, merge_max_path_options, DataType, IndexType, OffsetType, true>(
							combineBlockOffsets.get<uint32_t>() + (3 * numSharedRows), combineBlockOffsets.get<uint32_t>(), outputRowListHead,
							newmat_offsets.get<OffsetType>(), alpha,
							tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, NULL, tempChunkBufferSizes[run],
							chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
							currentFlag, restart_completion, shared_rows_handled, simple_restart_offset, currentCounters + 2
							);
						//----------------------------------------------------------
					}
					// if (cudaDeviceSynchronize() != cudaSuccess) {
					// 	throw MergeSimpleCaseException();
					// }
					// if(stats.measure_all)
					// 	stats.duration_merge_simple += recordTimer(individual_start, individual_stop, mergeStreams[0]);
				}

				// Complex Case -> Output gets merged through paths over MAX_CHUNKS
				if (mergeBlocks.shared_rows_max_chunks)
				{
					// if(stats.measure_all)
					// 	startTimer(individual_start, mergeStreams[1]);
					spgemm.setLaunchDimensions(mergeBlocks.shared_rows_max_chunks, mergeStreams[1], threads);
					//----------------------------------------------------------
					spgemm.h_mergeSharedRowsMaxChunks<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, merge_max_chunks, merge_max_path_options, DataType, IndexType, OffsetType>(
						NULL, combineBlockOffsets.get<uint32_t>() + (1 * numSharedRows), outputRowListHead,
						newmat_offsets.get<OffsetType>(), alpha,
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, NULL, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						currentFlag, restart_completion, shared_rows_handled,
						chunk_indices, chunk_values, chunk_multiplier,
						chunkElementCountDataOffset, max_chunks_restart_offset, num_chunks, currentCounters + 2);
					//----------------------------------------------------------
					// if (cudaDeviceSynchronize() != cudaSuccess) {
					// 	throw MergeMaxChunksCaseException();
					// }
					// if(stats.measure_all)
					// 	stats.duration_merge_max += recordTimer(individual_start, individual_stop, mergeStreams[1]);
				}

				// General Case -> Handles cases with more than MAX_CHUNKS chunks
				if (mergeBlocks.shared_rows_generalized)
				{
					// if(stats.measure_all)
					// 	startTimer(individual_start, mergeStreams[2]);
					spgemm.setLaunchDimensions(mergeBlocks.shared_rows_generalized, mergeStreams[2], threads);
					//----------------------------------------------------------
					spgemm.h_mergeSharedRowsGeneralized<nnz_per_thread, threads, blocks_per_mp, input_elements_per_thread, retain_elements_per_thread, generalized_merge_max_path_options, merge_max_path_options, DataType, IndexType, OffsetType>(
						NULL, combineBlockOffsets.get<uint32_t>() + (2 * numSharedRows), outputRowListHead,
						newmat_offsets.get<OffsetType>(), alpha,
						tempChunkBuffers[run].get<uint32_t>(), currentChunckAllocation, NULL, tempChunkBufferSizes[run],
						chunckPointers.get<void*>(), currentCounters, chunkPointerSize,
						currentFlag, restart_completion, shared_rows_handled,
						sample_offset, chunkElementConsumedAndPath, generalized_restart_offset, currentCounters + 2
						);
					//----------------------------------------------------------
					// if (cudaDeviceSynchronize() != cudaSuccess) {
					// 	throw MergeGeneralizedCaseException();
					// }
					// if(stats.measure_all)
					// 	stats.duration_merge_generalized += recordTimer(individual_start, individual_stop, mergeStreams[2]); 
				}
			}

			// Copy back flags
			HANDLE_ERROR(cudaMemcpy(&flagsAndListAllocCounters[0], chunckAllocations + chunckAllocationsSize, (numFlags + numCounters) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
			completed = flagsAndListAllocCounters[run + chunk_pointer_restart_run] == 0;

			if (!completed)
			{
				// if (stats.measure_all && stats.duration_merge_simple + stats.duration_merge_max + stats.duration_merge_generalized > 10000)
				// 	throw MergeLoopingException();


				uint32_t return_value = flagsAndListAllocCounters[run + chunk_pointer_restart_run];
				if (UintBitSet(return_value).test(0))
				{
					if (DEBUG_MODE)
					{
						std::cout << "Chunk Memory Restart allocating space for " << tempChunkBufferSizes[run] / (sizeof(DataType) + sizeof(IndexType)) << " elements\n";
					}
					// Get more chunk memory
					auto new_buffer_size = tempChunkBufferSizes[run];
					tempChunkBufferSizes[run+1] = new_buffer_size;
					tempChunkBuffers[run+1] = CU::allocMemory(new_buffer_size);
					if (++run == chunckAllocationsSize / 2)
						throw RestartOutOfMemoryException();
				}
				if (UintBitSet(return_value).test(1))
				{
					if (DEBUG_MODE)
					{
						std::cout << "Chunk Pointer Restart allocating " << targetChunkPointerSize << " new pointers\n";
					}
					// Get more chunk pointers
					chunkPointerSize += targetChunkPointerSize;
					chunckPointers.increaseMemRetainData((targetChunkPointerSize) * 8);
					targetChunkPointerSize *= 2;
					if (++chunk_pointer_restart_run == chunckAllocationsSize / 2)
						throw RestartOutOfChunkPointerException();
					HANDLE_ERROR(cudaMemcpy(currentCounters, currentCounters + 2, sizeof(uint32_t), cudaMemcpyDeviceToDevice));
				}
			}
			if (completed && !rowmerging)
			{
				numSharedRows = flagsAndListAllocCounters[numFlags + 1];
				if (numSharedRows > 0)
				{
					// if(stats.measure_all)
					// 	startTimer(individual_start, stream);

					if (combineBlockOffsetsSize < 4 * (numSharedRows + 1))
					{
						combineBlockOffsetsSize = 4 * (numSharedRows + 1024);
						combineBlockOffsets.assure(combineBlockOffsetsSize * sizeof(IndexType));
						memory_usage_in_Bytes += combineBlockOffsetsSize * sizeof(IndexType);
					}
					CUdeviceptr mergeTypeCounters = reinterpret_cast<CUdeviceptr>(chunckAllocations) + 4 * (chunckAllocationsSize + numFlags + numCounters);

					//----------------------------------------------------------
					mergeBlocks = spgemm.assignCombineBlocks<IndexType, merge_max_chunks, 2 * threads * input_elements_per_thread, threads>(numSharedRows, prefixSumTemp, prefixSumTempMemSize, sharedRowTracker, newmat_offsets, outputRowChunkCounter, combineBlockOffsets, mergeTypeCounters, stream);
					//----------------------------------------------------------

					completed = false;
					rowmerging = true;

					if(DEBUG_MODE)
					{
						std::cout << "################################################\n";
						std::cout << "Assigned " << numSharedRows << " shared rows to blocks, starting \n\t\t"
							<< mergeBlocks.shared_rows_simple << " simple merges for " << mergeBlocks.shared_rows_simple_rows << " rows,\n\t\t"
							<< mergeBlocks.shared_rows_max_chunks << " max chunk mergers, and\n\t\t"
							<< mergeBlocks.shared_rows_generalized << " general mergers\n";
					}

					// Set merge stage row stats
					// stats.shared_rows = numSharedRows;
					// stats.simple_mergers = mergeBlocks.shared_rows_simple;
					// stats.simple_rows = mergeBlocks.shared_rows_simple_rows;
					// stats.complex_rows = mergeBlocks.shared_rows_max_chunks;
					// stats.generalized_rows = mergeBlocks.shared_rows_generalized;
					merge_simple_rows = mergeBlocks.shared_rows_simple;
					merge_max_chunks_rows = mergeBlocks.shared_rows_max_chunks;
					merge_generalized_rows = mergeBlocks.shared_rows_generalized;

					// Allocate memory for all helper data structures
					size_t combinedMergeStageMemory_size =
						/*shared_rows_handled*/((numSharedRows) * sizeof(uint32_t)) +
						/*restart_completion*/((numSharedRows) * sizeof(uint32_t)) +
						/*chunkElementConsumedAndPath*/((numSharedRows) * (generalized_merge_max_path_options + helper_overhead) * sizeof(uint32_t)) +
						/*chunkElementCountDataOffset*/(((numSharedRows) * merge_max_chunks) * sizeof(uint32_t)) +
						/*num_chunks*/((numSharedRows) * sizeof(uint32_t)) +
						/*sample_offset*/(((numSharedRows) * (threads) * sizeof(uint32_t))); //+
						///* chunk_indices*/(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(IndexType*)) +
						///*chunk_values*/(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(DataType*)) +
						///*chunk_multiplier*/(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(DataType));
					combinedMergeStageMemory.assure(combinedMergeStageMemory_size);
					memory_usage_in_Bytes += combinedMergeStageMemory_size;

					//// Place pointers in memory allocation
					shared_rows_handled = combinedMergeStageMemory.get<uint32_t>();
					restart_completion = shared_rows_handled + (numSharedRows);
					chunkElementConsumedAndPath = restart_completion + (numSharedRows);
					chunkElementCountDataOffset = chunkElementConsumedAndPath + (numSharedRows) * (generalized_merge_max_path_options + helper_overhead);
					num_chunks = chunkElementCountDataOffset + ((numSharedRows) * merge_max_chunks);
					sample_offset = num_chunks + (numSharedRows);

					// TODO: Why does this work??????????????????????????
					chunk_indices_cptr.assure(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(IndexType*));
					chunk_indices = chunk_indices_cptr.get<IndexType*>();
					chunk_values_cptr.assure(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(DataType*));
					chunk_values = chunk_values_cptr.get<DataType*>();
					chunk_multiplier_cptr.assure(((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(DataType));
					chunk_multiplier = chunk_multiplier_cptr.get<DataType>();
					

					// TODO: Why does this NOT work??????????????????????????
					/*chunk_indices = reinterpret_cast<IndexType**>(chunk_multiplier + ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks));*/
					/*chunk_values = reinterpret_cast<DataType**>(chunk_indices + ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks));*/
					// chunk_multiplier = reinterpret_cast<DataType*>(sample_offset + ((numSharedRows) * (threads)));

					memory_usage_in_Bytes += ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(IndexType*);
					memory_usage_in_Bytes += ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(DataType*);
					memory_usage_in_Bytes += ((mergeBlocks.shared_rows_max_chunks) * merge_max_chunks) * sizeof(DataType);

					if (numSharedRows > lastSharedRows)
					{
						cudaMemset(combinedMergeStageMemory.get(), 0,
							/*chunkElementConsumedAndPath*/((numSharedRows) * (generalized_merge_max_path_options + helper_overhead) * sizeof(uint32_t)) +
							/*shared_rows_handled*/((numSharedRows) * sizeof(uint32_t)) +
							/*restart_completion*/((numSharedRows) * sizeof(uint32_t))
						);
						lastSharedRows = numSharedRows;
					}
					// if(stats.measure_all)
					// 	stats.duration_merge_case_computation = recordTimer(individual_start, individual_stop, stream);
				}
			}
		} while (!completed);

		// Let's write the chunks out to a csr matrix
		// if(stats.measure_all)
		// 	startTimer(individual_start, stream);

		//----------------------------------------------------------
		spgemm.computeRowOffsets<IndexType>(Crows, prefixSumTemp, prefixSumTempMemSize, newmat_offsets, stream);
		//----------------------------------------------------------
		
		// Allocate output matrix
		IndexType matrix_elements;
		CUdeviceptr offs = newmat_offsets;
		offs += sizeof(IndexType) * Crows;
		HANDLE_ERROR(cudaMemcpy(&matrix_elements, reinterpret_cast<void*>(offs), sizeof(IndexType), cudaMemcpyDeviceToHost));

		if (matC->nnz != matrix_elements)
		{
			//std::cout << "Reallocation HERE ################" << matC.nnz << " | " << matrix_elements <<"\n";
			// matC.alloc(Crows, Ccols, matrix_elements, true);
			matC->nnz = matrix_elements;
			cudaMalloc(&matC->val_data, sizeof(DataType)*matrix_elements);
			cudaMalloc(&matC->col_data, sizeof(IndexType)*matrix_elements);
		}
		matC->row_data = (int *)std::move(newmat_offsets.getRelease<IndexType>());

		//----------------------------------------------------------
		spgemm.h_copyChunks<DataType, IndexType, OffsetType>(chunckPointers.get<void*>(), currentCounters, 
			(DataType*)matC->val_data, (IndexType*)matC->col_data, (IndexType*)matC->row_data, alpha);
		//----------------------------------------------------------
		// if(stats.measure_all)
		// 	stats.duration_write_csr = recordTimer(individual_start, individual_stop, stream);

		// if (stats.measure_all)
		// {
		// 	stats.mem_allocated_chunks = tempChunkBufferSizes[0] * (run + 1);
		// 	uint32_t* d_current_chunk_allocation = chunckAllocations + (2 * run);
		// 	uint32_t h_current_chunk_allocation = 0;
		// 	HANDLE_ERROR(cudaMemcpy(&h_current_chunk_allocation, d_current_chunk_allocation, sizeof(uint32_t), cudaMemcpyDeviceToHost));
		// 	stats.mem_used_chunks = tempChunkBufferSizes[0] * run + h_current_chunk_allocation;
		// }
		// stats.restarts = run + chunk_pointer_restart_run;

		// ##############################
		// stats.duration = recordTimer(ce_start, ce_stop, stream);
		// ##############################
		
		// Stream cleanup
		// if (!(stats.measure_all))
		// {
		// 	for (int i = 0; i < number_merge_streams; ++i)
		// 		cudaStreamDestroy(mergeStreams[i]);
		// }

		return;
	}
}
#endif