__global__ void scale(float* x, int n) {
    int i = threadIdx.x;
    x[i] = x[i] * 2.0f;
}
