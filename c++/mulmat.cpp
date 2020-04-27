#include <stdio.h>
#include <cstdint>
#include <ctime>
#include <iostream>

#define prefetch(mem) \
         __asm__ __volatile__ ("prefetchnta %0" : : "m" (*((char *)(mem))))

static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

template<class T>
void Init(T** &A, T* &x, T* &y, int size)
{
    T *p;
    A = new T*[size];
    p = new T[size*size];
    for(int i=0; i<size; i++) A[i] = &p[i*size];

    x = new T[size];
    y = new T[size];

    for(int j=0; j<size; j++)
    for(int i=0; i<size; i++)
    {
        A[i][j] = 0.234;
    }

    for(int i=0; i<size; i++)
    {
        x[i] = 1.234;
        y[i] = 0;
    }
}

template<class T>
void Destroy(T** &A, T* &x, T* &y)
{
    delete[] &A[0][0];
    delete[] y;
    delete[] x;
}

template<class T>
void classic1(int size, int num)
{
    clock_t time1;
    T **A, *x, *y;
    Init(A, x, y, size);
    time1 = clock();
    for(int k=0; k<num; k++)
    {
        for(int i=0; i<size; i++)
        {
            y[i] = 0;
            for(int j=0; j<size; j++)
                y[i] += A[i][j]*x[j];
        }
    }
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC * num));
    Destroy(A, x, y);
    return ;
}

template<class T>
void classic2(int size, int num)
{
    T **A, *x, *y;
    Init(A, x, y, size);
    T ytemp;
    clock_t time1;
    time1 = clock();
    for(int k=0; k<num; k++)
    {
        T *Apos = &A[0][0];
        for(int i=0; i<size; i++)
        {
            T *xpos = &x[0];
            ytemp = 0;
            for(int j=0; j<size; j++)
                ytemp += (*Apos++) * (*xpos++);
            y[i] = ytemp;
        }
    }
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC * num));
    Destroy(A, x, y);
    return ;
}

template<class T>
void classic3(int size, int num)
{
    T **A, *x, *y;
    Init(A, x, y, size);
    clock_t time1;
    time1 = clock();
    for(int k=0; k<num; k++)
    {
        T *Apos1 = &A[0][0];
        T *Apos2 = &A[1][0];
        T *ypos = &y[0];
        for(int i=0; i<size/2; i++)
        {
            T ytemp1 = 0;
            T ytemp2 = 0;
            T *xpos = &x[0];
            for(int j=0; j<size; j++) {
                ytemp1 += (*Apos1++) * (*xpos++);
                ytemp2 += (*Apos2++) * (*xpos++);
                xpos++;
            }
            *ypos = ytemp1;
            ypos++;
            *ypos = ytemp2;
            ypos++;
            Apos1 += size;
            Apos2 += size;
        }
    }
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC * num));
    Destroy(A, x, y);
    return ;
}

template<class T>
void classic4(int size, int num)
{
    T **A, *x, *y;
    Init(A, x, y, size);
    clock_t time1;
    time1 = clock();
    for(int k=0; k<num; k++)
    {
        T *Apos1 = &A[0][0];
        T *Apos2 = &A[1][0];
        T *Apos3 = &A[2][0];
        T *Apos4 = &A[3][0];
        T *ypos = &y[0];
        for(int i=0; i<size/4; i++)
        {
            T ytemp1 = 0;
            T ytemp2 = 0;
            T ytemp3 = 0;
            T ytemp4 = 0;

            T *xpos = &x[0];
            for(int j=0; j<size; j++) {
                ytemp1 += (*Apos1++) * (*xpos++);
                ytemp2 += (*Apos2++) * (*xpos++);
                ytemp3 += (*Apos3++) * (*xpos++);
                ytemp4 += (*Apos4++) * (*xpos++);
                xpos++;
            }
            *ypos = ytemp1;
            ypos++;
            *ypos = ytemp2;
            ypos++;
            *ypos = ytemp3;
            ypos++;
            *ypos = ytemp4;
            ypos++;
            Apos1 += 3*size;
            Apos2 += 3*size;
            Apos3 += 3*size;
            Apos4 += 3*size;
        }
    }
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC * num));
    Destroy(A, x, y);
    return ;
}

template<class T>
void classic5(int size, int num)
{
    T **A, *x, *y;
    Init(A, x, y, size);
    clock_t time1;
    time1 = clock();
    for(int k=0; k<num; k++)
    {
        T *Apos1 = &A[0][0];
        T *Apos2 = &A[1][0];
        T *ypos = &y[0];
        for(int i=0; i<size/4; i++)
        {
            register T ytemp1 = 0;
            register T ytemp2 = 0;
            register T *xpos = &x[0];
            int blocksize = size / 8;

            T x0 = xpos[0];
            T x1 = xpos[1];

            for(int j=0; j<blocksize; j++) {
                prefetch(Apos1+64);
                prefetch(Apos2+64);
                ytemp1 += x0 * (Apos1[0]);
                ytemp2 += x0 * (Apos2[0]);
                x0 = xpos[2];
                ytemp1 += x1 * (Apos1[1]);
                ytemp2 += x1 * (Apos2[1]);
                x1 = xpos[3];

                ytemp1 += x0 * (Apos1[2]);
                ytemp2 += x0 * (Apos2[2]);
                x0 = xpos[4];

                ytemp1 += x1 * (Apos1[3]);
                ytemp2 += x1 * (Apos2[3]);
                x1 = xpos[5];

                ytemp1 += x0 * (Apos1[4]);
                ytemp2 += x0 * (Apos2[4]);
                x0 = xpos[6];

                ytemp1 += x1 * (Apos1[5]);
                ytemp2 += x1 * (Apos2[5]);
                x0 = xpos[7];

                xpos += 8;
                ytemp1 += x0 * (Apos1[6]);
                ytemp2 += x0 * (Apos2[6]);
                x0 = xpos[0];

                ytemp1 += x1 * (Apos1[7]);
                Apos1 += 8;
                ytemp2 += x1 * (Apos2[7]);
                x1 = xpos[1];
                Apos2 += 8;
            }
            ypos[0] = ytemp1;
            ypos[1] = ytemp2;
            ypos+=2;
            Apos1 += size;
            Apos2 += size;
        }
    }
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC * num));
    Destroy(A, x, y);
    return ;
}

int main(int argc, char** argv) {
    int ntime;
    if(argc > 1)
        ntime = atoi(argv[1]);
    else
        ntime = 100;
    int size = 1024;
    double dtime;
    classic1<double>(size, ntime);
    classic2<double>(size, ntime);
    classic3<double>(size, ntime);
    classic4<double>(size, ntime);
    classic5<double>(size, ntime);
}
