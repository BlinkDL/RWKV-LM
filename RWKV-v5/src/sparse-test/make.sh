gcc -o sgemv_example sgemv_example.c -lopenblas -O2 -g

# -rt for high precision timer
gcc -o test-mmap-overlay test-mmap-overlay.c -lrt  -O2 -g
# ./mmap_test <file_path> <N>

gcc -shared -o mmap_helper.so -fPIC mmap_helper.c -O2 -g
