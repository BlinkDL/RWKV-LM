// FL sept 2024... tested on rpi4
// apt install libopenblas-dev/jammy
// gcc -o sgemv_example sgemv_example.c -lopenblas

// Usage: ./sgemv_example [-in-mem | -mmap | -fread]

// https://chatgpt.com/share/66f44750-58ec-8004-8379-d1acbc4248d3


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#define FILENAME "/tmp/testmat.bin"         // to save matrix A 

// Function to fill an array with random float values
void fill_random(float *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random values between -1 and 1
    }
}

// fixed values 
void fill_fixed(float *array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = (i / size) * 2.0f - 1.0f;  // Random values between -1 and 1
    }
}

// Function to get the current time in seconds
double get_time_in_seconds() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}


// Function to save matrix A to a disk file
void save_matrix_to_file(const char *filename, float *A, int M, int N) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        printf("Error opening file for writing!\n");
        exit(1);
    }
    fwrite(A, sizeof(float), M * N, file);
    fclose(file);
    printf("saved to %s\n", filename);
}


// Function to load matrix A from a disk file using mmap
float* load_matrix_from_file_mmap(const char *filename, int M, int N) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        printf("Error opening file for reading!\n");
        exit(1);
    }

    // Get the size of the file (M * N * sizeof(float))
    size_t size = M * N * sizeof(float);

    // Use mmap to map the file to memory
    float *mapped_A = (float *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_A == MAP_FAILED) {
        printf("Error mapping the file!\n");
        close(fd);
        exit(1);
    }

    printf("%s\n", __func__); 
    for (int i = 0; i < 10; i++) {
        printf("%f ", mapped_A[i]);
    }


    close(fd);
    return mapped_A;
}

// do mmap, then madvise() to mark sparse regions
float* load_matrix_from_file_mmap_madvise(const char *filename, int M, int N) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        printf("Error opening file for reading!\n");
        exit(1);
    }

    // Get the size of the file (M * N * sizeof(float))
    size_t size = M * N * sizeof(float);

    // Use mmap to map the file to memory
    float *mapped_A = (float *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_A == MAP_FAILED) {
        printf("Error mapping the file!\n");
        close(fd);
        exit(1);
    }

    int page_count = size / 4096; 
    size_t advise_offset = 2 * 4096;  
    size_t advise_length = (page_count-2) * 4096; 

    if (madvise((char*)mapped_A + advise_offset, advise_length, MADV_DONTNEED) == -1) {
        perror("Error with madvise");
        munmap(mapped_A, size);
        close(fd);
        return 0;
    }

    printf("Marked region from page %lu to %lu as unneeded using MADV_DONTNEED.\n", 
           (unsigned long)advise_offset/4096, (unsigned long)(advise_offset + advise_length)/4096);

    // check 
    printf("%s\n", __func__); 
    for (int i = 0; i < 10; i++) {
        printf("%f ", mapped_A[i]);
    } // should have values
    printf("\n--------------------\n");
    for (int i = advise_offset; i < advise_offset + 10; i++) {
        printf("%f ", mapped_A[i]);
    } // should be zeros

    close(fd);
    return mapped_A;
}

// do mmap, then create anonymous mapping atop it 
float* load_matrix_from_file_mmap_overlay(const char *filename, int M, int N) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        printf("Error opening file for reading!\n");
        exit(1);
    }

    // Get the size of the file (M * N * sizeof(float))
    size_t size = M * N * sizeof(float);

    // Use mmap to map the file to memory
    float *mapped_A = (float *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_A == MAP_FAILED) {
        printf("Error mapping the file!\n");
        close(fd);
        exit(1);
    }

    int page_count = size / 4096; 
    size_t advise_offset = 2 * 4096;  
    size_t advise_length = (page_count-2) * 4096;   // ~99 sparisty
    // size_t advise_length = 2 * 4096; 

    if (mmap(mapped_A + advise_offset, advise_length, PROT_READ,
                            MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) == MAP_FAILED) {
        perror("Error with madvise");
        munmap(mapped_A, size);
        close(fd);
        return 0;
    }

    printf("Marked region from page %lu to %lu as unneeded using MADV_DONTNEED.\n", 
           (unsigned long)advise_offset/4096, (unsigned long)(advise_offset + advise_length)/4096);

    // check 
    printf("%s\n", __func__); 
    for (int i = 0; i < 10; i++) {
        printf("%f ", mapped_A[i]);
    } // should have values
    printf("\n--------------------\n");
    for (int i = advise_offset; i < advise_offset + 10; i++) {
        printf("%f ", mapped_A[i]);
    } // should be zeros

    close(fd);
    return mapped_A;
}



// Function to load matrix A from a disk file using read()
float* load_matrix_from_file_read(const char *filename, int M, int N) {
    // Open the file in read-only mode
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        printf("Error opening file for reading!\n");
        exit(1);
    }

    // Allocate memory for the matrix
    float *A = (float *)malloc(M * N * sizeof(float));
    if (A == NULL) {
        printf("Memory allocation failed!\n");
        close(fd);
        exit(1);
    }

    // Read the matrix from the file into the allocated memory
    ssize_t bytes_read = read(fd, A, M * N * sizeof(float));
    if (bytes_read == -1) {
        printf("Error reading the file!\n");
        free(A);  // Free the memory before exiting
        close(fd);
        exit(1);
    }

    if (bytes_read != M * N * sizeof(float)) {
        printf("File size mismatch!\n");
        free(A);
        close(fd);
        exit(1);
    }

    printf("%s\n", __func__); 
    for (int i = 0; i < 10; i++) {
        printf("%f ", A[i]);
    }    

    // Close the file after reading
    close(fd);

    return A;
}


const int M = 4096;  // Number of rows in matrix A
const int N = 1024;  // Number of columns in matrix A

float *A, *X, *Y; 


void alloc_and_fill_A(int save_file) {
    A = (float *)malloc(M * N * sizeof(float));
    fill_random(A, M * N);          // slow 
    if (save_file)
        save_matrix_to_file(FILENAME, A, M, N);  // optional 
}

int main(int argc, char *argv[]) {
    // Argument check
    if (argc != 2) {
        printf("Usage: %s [-in-mem | -mmap | -fread | -sparse]\n", argv[0]);
        return 1;
    }

    // Store the mode based on argv[1]
    enum { IN_MEM, MMAP, FREAD, SPARSE } mode;

    if (strcmp(argv[1], "-in-mem") == 0) {
        mode = IN_MEM;
    } else if (strcmp(argv[1], "-mmap") == 0) {
        mode = MMAP;
    } else if (strcmp(argv[1], "-fread") == 0) {
        mode = FREAD;
    } else if (strcmp(argv[1], "-sparse") == 0) {
        mode = SPARSE;
    } else {
        printf("Usage: %s [-in-mem | -mmap | -fread | - sparse]\n", argv[0]);
        return 1;
    }

    double start_time, end_time;

    //////////  A -- weight  ////////////////////
    start_time = get_time_in_seconds();
    if (mode == IN_MEM)
        alloc_and_fill_A(0/*don't save to file*/); // one way to do it 
    else if (mode == MMAP)
        A=load_matrix_from_file_mmap(FILENAME, M, N);     // the mmap way 
    else if (mode == FREAD)
        A=load_matrix_from_file_read(FILENAME, M, N);       // the read() way
    else if (mode == SPARSE) 
        // A=load_matrix_from_file_mmap_madvise(FILENAME, M, N);       // sparse -- bad
        A=load_matrix_from_file_mmap_overlay(FILENAME, M, N);       // sparse


    end_time = get_time_in_seconds();

    printf("creation A. time: %.2f ms\n", 1000*(end_time - start_time));

    //////////  X and Y. cheap  ////////////////////
    float *X = (float *)malloc(N * sizeof(float));
    float *Y = (float *)malloc(M * sizeof(float));

    if (A == NULL || X == NULL || Y == NULL) {
        printf("Memory allocation failed!\n");
        return 1;
    }
    srand(time(NULL));  // Seed the random number generator
    // fill_random(X, N);    
    fill_fixed(X, N);    
    // Initialize Y to zero
    for (int i = 0; i < M; i++) {
        Y[i] = 0.0f;
    }

    //////////  computation ////////////////////

    // Scalar values for alpha and beta
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Measure the start time
     start_time = get_time_in_seconds();

    // Perform matrix-vector multiplication Y = alpha * A * X + beta * Y
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, alpha, A, N, X, 1, beta, Y, 1);

    // Measure the end time
    end_time = get_time_in_seconds();

    // Print the execution time
    printf("cblas_sgemv time of cblas_sgemv: %.2f ms\n", 1000*(end_time - start_time));

    // Optionally print a small part of the result (for debugging purposes)
    for (int i = 0; i < 10; i++) {
        printf("%f ", A[i]);
    }
    printf("Sample of resulting vector Y:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", Y[i]);
    }

    // Free the dynamically allocated memory
    // free(A);         for testing, no need 
    printf("warning: should free A\n");
    free(X);
    free(Y);

    return 0;
}


/*
    results: rpi4
    FL: -- rpi4 measurement                 the kernel computation
    4k,1k       A          -in-mem         5-6ms        (could be few/no cache miss 
    4K,1k       A          -mmap            5.75ms      -- the OS is doing some kind of async page fault? 
    4K,1k       A          -fread            6.55ms        (fread time: 50ms)
    4K,1k       A          -sparse            4.3ms        (20% faster than dense, in-mem)

    2k,1k       A          -in-mem         ????  (what will happen??? -- TBD
*/