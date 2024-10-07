// dd if=/dev/random of=/tmp/existing_file.bin bs=4K count=2

/* When you create an anonymous mapping on top of an existing memory-mapped
region (such as one backed by a file), you essentially replace the original
mapping with the new one */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

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


int main() {
    const char *filename = "/tmp/existing_file.bin";
    size_t file_size = 4096*2; 
    int fd;
    char *map;

    // 1. Open the existing file
    fd = open(filename, O_RDWR);
    if (fd == -1) {
        perror("Error opening file");
        return 1;
    }

    // 2. Get the size of the file (using fstat or assume fixed size)
    // In this case, we assume the file size is known (4KB), but you can also use fstat()
    
    // 3. Memory-map the file
    map = (char *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        return 1;
    }

    // 5. Use madvise() to mark a part of the region as unneeded (MADV_DONTNEED)
    size_t advise_offset = 4096;  // Start from 1024 bytes offset (within the 4KB range)
    size_t advise_length = 4096;  // A 1KB region

    printf("Accessing the advised region BEFORE ... : ");
    // for (size_t i = advise_offset; i < advise_offset + 16; i++) {
    //     printf("%02x ", map[i]);  // Print as hex values
    // }    

    print_memory_content(map, file_size); 


    // Now, let's say we want to make the first 4KB of the mapping return zeros.
    // We remap the first 4KB with an anonymous mapping.
    void *new_mapping = mmap(map + advise_offset, advise_length, PROT_READ | PROT_WRITE,
                            MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);


    // Check if the new mmap() succeeded
    if (new_mapping == MAP_FAILED) {
        perror("anonymous mmap failed");
        exit(EXIT_FAILURE);
    }

    // The first 4KB of mapped_region now points to zero-initialized memory,
     // while the rest still reflects the file's content.

    printf("Marked region from byte %lu to %lu as unneeded using MADV_DONTNEED.\n", 
           (unsigned long)advise_offset, (unsigned long)(advise_offset + advise_length));

    print_memory_content(map, file_size); 

    // printf("Accessing the advised region (should return zeros or cleared data): ");
    // for (size_t i = advise_offset; i < advise_offset + 16; i++) {
    //     printf("%02x ", map[i]);  // Print as hex values
    // }
    // printf("\n");

    /* Unmapping: If you later munmap() the anonymous region, you can remap the
    file to that region again to restore access to the file's contents. */


    // 9. Cleanup
    if (munmap(map, file_size) == -1) {
        perror("Error unmapping the file");
    }
    close(fd);

    return 0;
}
