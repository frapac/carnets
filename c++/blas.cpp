#include <stdio.h>
#include <cstdint>
#include <ctime>
#include <iostream>
#include "cblas-openblas.h"

template<class T>
void Init(T* &A, T* &x, T* &y, int size)
{
    A = new T[size*size];

    x = new T[size];
    y = new T[size];

    for(int j=0; j<size; j++)
    for(int i=0; i<size; i++)
    {
        A[i + j*size] = 0.234;
    }

    for(int i=0; i<size; i++)
    {
        x[i] = 1.234;
        y[i] = 0;
    }
}

template<class T>
void Destroy(T* &A, T* &x, T* &y)
{
    delete[] A;
    delete[] y;
    delete[] x;
}

template<class T>
void classic_blas(int size, int num)
{
    T *A, *x, *y;
    Init(A, x, y, size);
    clock_t time1;
    time1 = clock();
    for(int i=0; i < num; i++)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, size, size, 1.0, A, size, x, 1, 0.0, y, 1);
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC * num));
    printf("y0: %.4f\n", y[0]);
    Destroy(A, x, y);
    return ;
}

int main(int argc, char** argv) {
    openblas_set_num_threads(1);
    int size = 2048;
    int num = 1000;
    classic_blas<double>(size, num);
}
