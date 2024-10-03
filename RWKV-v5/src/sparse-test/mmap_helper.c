
// gcc -shared -o mmap_helper.so -fPIC mmap_helper.c

#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <stdint.h>

// batch mmap(), to be called from python code 

// Function to mmap each address from the passed list

int mmap_addresses0(uintptr_t *addresses, size_t num_addresses, size_t length, int prot, int flags, long *offsets, int fd) {
    for (int i = 0; i < num_addresses; i++) {
        void *addr = (void *)(addresses[i] & ~(0xFFF));
        long offset = offsets[i] & ~(0xFFF);

        void *mapped_addr = mmap(addr, length, prot, flags, fd, offset);
        
        if (mapped_addr == MAP_FAILED) {
            perror("mmap_addresses: mmap failed");
            printf("i %d, addr %p, length %ld, prot %d, flags %d, fd %d, offset %ld\n", i, addr, length, prot, flags, fd, offsets[i]);
            return -1; 
        } else {
            // printf("Memory mapped at: %p for requested address: %p with offset: %ld\n", mapped_addr, addr, offsets[i]);
            ;
        }
    }
    return 0; 
}


int mmap_addresses(uintptr_t *addresses, size_t num_addresses, size_t length, int prot, int flags, long *offsets, int fd) {
    void *last_mapped_addr = NULL;
    for (int i = 0; i < num_addresses; i++) {
        void *addr = (void *)(addresses[i] & ~(0xFFF));
        long offset = offsets[i] & ~(0xFFF);

        if (addr != last_mapped_addr) {
            void *mapped_addr = mmap(addr, length, prot, flags, fd, offset);
            
            if (mapped_addr == MAP_FAILED) {
                perror("mmap_addresses: mmap failed");
                printf("i %d, addr %p, length %ld, prot %d, flags %d, fd %d, offset %ld\n", i, addr, length, prot, flags, fd, offsets[i]);
                return -1; 
            } else {
                // printf("Memory mapped at: %p for requested address: %p with offset: %ld\n", mapped_addr, addr, offsets[i]);
                last_mapped_addr = addr;
            }
        }
    }
    return 0; 
}


// idea1: assuing addresses are sorted ascending; check if the page is already mapped. if so, 
// skip the mmap() call.

// idea2: don't pass in both addresses & offsets, just pass in the base address, and the offsets