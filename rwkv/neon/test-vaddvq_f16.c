// gcc -march=armv8.2-a+fp16 test-vaddvq_f16.c

#include <arm_neon.h>

float16x8_t a = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
float16_t sum = vaddvq_f16(a);  // Sum all elements in the vector

// FL: no such a thing like vaddvq_f16
// cf: https://developer.arm.com/architectures/instruction-sets/intrinsics/#q=vaddvq

printf("Sum of vector elements: %f\n", (float)sum);
