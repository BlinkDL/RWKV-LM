// dd if=/dev/random of=/tmp/existing_file.bin bs=1M count=4

/* When you create an anonymous mapping on top of an existing memory-mapped
region (such as one backed by a file), you essentially replace the original
mapping with the new one */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>

void print_memory_content(void *ptr, size_t size) {
    unsigned int *p = (unsigned int *)ptr;
    size_t count = size / sizeof(unsigned int);  // Number of unsigned ints to print
    // size_t row_length = 8;  // Number of unsigned ints per row (adjusted for typical console width)
    size_t row_length = 16;  // Number of unsigned ints per row (adjusted for typical console width)

    for (size_t i = 0; i < count; i++) {
        // Print the memory content as unsigned int
        printf("%08x ", p[i]);
        
        // Print a newline after every row_length elements
        if ((i + 1) % row_length == 0) {
            printf("\n");
        }
    }

    // If there are remaining elements that didn't complete a full row, add a newline
    if (count % row_length != 0) {
        printf("\n");
    }
}

void mmap_and_anonymous_mappings(const char *file_path, int N) {
    int fd;
    struct stat sb;
    void *map;
    size_t file_size;
    struct timespec start, end;

    // Open the file
    fd = open(file_path, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Get the file size
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        close(fd);
        exit(EXIT_FAILURE);
    }
    file_size = sb.st_size;

    // Memory-map the file
    map = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        exit(EXIT_FAILURE);
    }

    // Measure the start time
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Create N anonymous mappings atop the file mmap region
    N = (N < file_size / 4096) ? N : file_size / 4096;  // Limit N to the number of 4KB pages in the file

    for (int i = 0; i < N; i++) {
        void *new_mapping = mmap((char *)map + i * 4096, 4096, PROT_READ | PROT_WRITE,
                                 MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (new_mapping == MAP_FAILED) {
            perror("anonymous mmap failed");
            munmap(map, file_size);
            close(fd);
            exit(EXIT_FAILURE);
        }
    }

    // Measure the end time
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate the total time taken
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Total time taken for %d anonymous mappings: %.2f ms %.2f ms per page\n", N, 1000*time_taken, 1000*time_taken/N);

    // Cleanup
    if (munmap(map, file_size) == -1) {
        perror("Error unmapping the file");
    }
    close(fd);
}

// MP version. wont help much.  maybe already too fast (2ms for 100 mappings)
void mmap_and_anonymous_mappings_smp(const char *file_path, int N) {
    int fd;
    struct stat sb;
    void *map;
    size_t file_size;
    struct timespec start, end;

    // Open the file
    fd = open(file_path, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Get the file size
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        close(fd);
        exit(EXIT_FAILURE);
    }
    file_size = sb.st_size;

    // Memory-map the file
    map = mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        exit(EXIT_FAILURE);
    }

    // Measure the start time
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Create N anonymous mappings atop the file mmap region
    N = (N < file_size / 4096) ? N : file_size / 4096;  // Limit N to the number of 4KB pages in the file

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        void *new_mapping = mmap((char *)map + i * 4096, 4096, PROT_READ | PROT_WRITE,
                                 MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (new_mapping == MAP_FAILED) {
            perror("anonymous mmap failed");
            #pragma omp critical
            {
                munmap(map, file_size);
                close(fd);
                exit(EXIT_FAILURE);
            }
        }
    }

    // Measure the end time
    clock_gettime(CLOCK_MONOTONIC, &end);

    // Calculate the total time taken
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Total time taken for %d anonymous mappings: %.2f ms %.2f ms per page\n", N, 1000*time_taken, 1000*time_taken/N);

    // Cleanup
    if (munmap(map, file_size) == -1) {
        perror("Error unmapping the file");
    }
    close(fd);
}

void test_mmap_overlay(const char *filename, size_t file_size) {
    int fd;
    char *map;

    // 1. Open the existing file
    fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        return;
    }

    // 2. Get the size of the file (using fstat or assume fixed size)
    // In this case, we assume the file size is known (4KB), but you can also use fstat()
    
    // 3. Memory-map the file
    map = (char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        return;
    }

    // 5. Use madvise() to mark a part of the region as unneeded (MADV_DONTNEED)
    size_t map_offset = 4096;  // Start from 1024 bytes offset (within the 4KB range)
    size_t map_length = 4096;  // A 1KB region

    printf("Accessing the mmap region BEFORE ... : ");
    print_memory_content(map, file_size); 

    // Now, let's say we want to make the first 4KB of the mapping return zeros.
    // We remap the first 4KB with an anonymous mapping.
    void *new_mapping = mmap(map + map_offset, map_length, PROT_READ | PROT_WRITE,
                            MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    // Check if the new mmap() succeeded
    if (new_mapping == MAP_FAILED) {
        perror("anonymous mmap failed");
        exit(EXIT_FAILURE);
    }

    // The first 4KB of mapped_region now points to zero-initialized memory,
    // while the rest still reflects the file's content.

    printf("Marked region from byte %lu to %lu as unneeded using MADV_DONTNEED.\n", 
           (unsigned long)map_offset, (unsigned long)(map_offset + map_length));

    print_memory_content(map, file_size); 

    // 9. Cleanup
    if (munmap(map, file_size) == -1) {
        perror("Error unmapping the file");
    }
    close(fd);
}


int main(int argc, char *argv[]) {
    const char *default_file_path = "/tmp/existing_file.bin";
    const char *file_path;
    const char *option;

    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <-test|-bench> [file_path]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    option = argv[1];
    file_path = (argc == 3) ? argv[2] : default_file_path;

    if (strcmp(option, "-test") == 0) {
        test_mmap_overlay(file_path, 4096 * 2);  // Assuming file size is >8KB for the test
    } else if (strcmp(option, "-bench") == 0) {
        int N = 800;  // Default value for N
        if (argc == 3) {
            N = atoi(argv[2]);
        }
        mmap_and_anonymous_mappings(file_path, N);
        // mmap_and_anonymous_mappings_smp(file_path, N);
    } else {
        fprintf(stderr, "Invalid option: %s. Use -test or -bench.\n", option);
        exit(EXIT_FAILURE);
    }

    return 0;
}


/*
    rpi4

    (myenv) robot@rpi4:~/workspace-rwkv/RWKV-LM/RWKV-v5/src/sparse-test$ ./test-mmap-overlay -bench

    Total time taken for 800 anonymous mappings: 7.73 ms 0.01 ms per page
    (about 3x faster than pytorch)

*/