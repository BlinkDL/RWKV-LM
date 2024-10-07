// dd if=/dev/random of=/tmp/existing_file.bin bs=4K count=2

// WONT WORK -- B/C madvise(..MADV_DONTNEED) only returns 0 on anonymous mapping.
// it will still load contents from disk 

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

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
    map = (char *)mmap(NULL, file_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        return 1;
    }

    // 5. Use madvise() to mark a part of the region as unneeded (MADV_DONTNEED)
    size_t advise_offset = 4096;  // Start from 1024 bytes offset (within the 4KB range)
    size_t advise_length = 4096;  // A 1KB region

    printf("Accessing the advised region BEFORE ... : ");
    for (size_t i = advise_offset; i < advise_offset + 16; i++) {
        printf("%02x ", map[i]);  // Print as hex values
    }    

    if (madvise(map + advise_offset, advise_length, MADV_DONTNEED) == -1) {
        perror("Error with madvise");
        munmap(map, file_size);
        close(fd);
        return 1;
    }

    printf("Marked region from byte %lu to %lu as unneeded using MADV_DONTNEED.\n", 
           (unsigned long)advise_offset, (unsigned long)(advise_offset + advise_length));

    // 6. Accessing the region after madvise() with MADV_DONTNEED
    printf("Accessing the advised region (should return zeros or cleared data): ");
    for (size_t i = advise_offset; i < advise_offset + 16; i++) {
        printf("%02x ", map[i]);  // Print as hex values
    }
    printf("\n");

    // 7. Write new data to the region after MADV_DONTNEED (which re-allocates pages)
    strcpy(map + advise_offset, "New data after MADV_DONTNEED.");

    // 8. Read back the new data
    printf("New data in the advised region: %s\n", map + advise_offset);

    // 9. Cleanup
    if (munmap(map, file_size) == -1) {
        perror("Error unmapping the file");
    }
    close(fd);

    return 0;
}
