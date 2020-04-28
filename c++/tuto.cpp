#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <ctime>
#include <vector>
#include <x86intrin.h>

using namespace std;
constexpr float SINFTY = std::numeric_limits<float>::infinity();
typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));
/* Vectorization */
static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }

static float8_t* float8_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(float8_t), sizeof(float8_t) * n)) {
        throw std::bad_alloc();
    }
    return (float8_t*)tmp;
}

constexpr float8_t f8infty {
    SINFTY, SINFTY, SINFTY, SINFTY, SINFTY, SINFTY, SINFTY, SINFTY
};

static inline float hmin8(float8_t vv) {
    float v = SINFTY;
    for (int i=0; i < 8; ++i) {
        v = std::min(vv[i], v);
    }
    return v;
}

static inline float8_t min8(float8_t x, float8_t y) {
    return x < y ? x : y;
}

void step(float* r, const float* d, int n) {
    std::vector<float> t(n*n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            t[j*n + i] = d[i*n + j];

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float v = std::numeric_limits<float>::infinity();
            for (int k = 0; k < n; ++k) {
                float x = d[n*i + k];
                float y = t[n*j + k];
                float z = x + y;
                v = std::min(v, z);
            }
            r[n*i + j] = v;
        }
    }
}

void step1(float* r, const float* d_, int n) {
    constexpr int nb = 4;
    int na = (n + nb - 1) / nb;
    int nab = na * nb;
    /* Padding */
    std::vector<float> d(n*nab, SINFTY);
    std::vector<float> t(n*nab, SINFTY);

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            d[j*nab + i] = d_[j*n + i];
            t[j*nab + i] = d_[i*n + j];
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float vv[nb];
            for (int kb=0; kb < nb; ++kb)
                vv[kb] = SINFTY;

            for (int ka=0; ka < na; ++ka) {
                for (int kb = 0; kb < nb; ++kb) {
                    float x = d[nab*i + ka * nb + kb];
                    float y = t[nab*j + ka * nb + kb];
                    float z = x + y;
                    vv[kb] = std::min(vv[kb], z);
                }
            }
            float v = SINFTY;
            for (int kb = 0; kb < nb; ++kb) {
                v = std::min(vv[kb], v);
            }
            r[n*i + j] = v;
        }
    }
}

void step2(float* r, const float* d_, int n) {
    constexpr int nb = 8;
    int na = (n + nb - 1) / nb;

    float8_t* vd = float8_alloc(n*na);
    float8_t* vt = float8_alloc(n*na);

    #pragma omp parallel for
    for (int j=0; j < n; ++j) {
        for (int ka=0; ka < na; ++ka) {
            for (int kb=0; kb < nb; ++kb) {
                int i = ka*nb + kb;
                vd[na*j + ka][kb] = i < n ? d_[n*j + i] : SINFTY ;
                vt[na*j + ka][kb] = i < n ? d_[n*i + j] : SINFTY ;
            }
        }
    }

    #pragma omp parallel for
    for (int i=0; i < n; ++i) {
        for (int j=0; j < n; ++j) {
            float8_t vv = f8infty;
            for (int ka=0; ka < na; ++ka) {
                float8_t x = vd[na*i + ka];
                float8_t y = vt[na*j + ka];
                float8_t z = x + y;
                vv = min8(vv, z);
            }
            r[n*i + j] = hmin8(vv);
        }
    }

    std::free(vt);
    std::free(vd);
}

void step3(float* r, const float* d_, int n) {
    /* Elements per vector */
    constexpr int nb = 8;
    int na = (n + nb - 1) / nb;

    /* Block size */
    constexpr int nd = 3;
    int nc = (n + nd - 1) / nd;
    int ncd = nc * nd;

    float8_t* vd = float8_alloc(ncd*na);
    float8_t* vt = float8_alloc(ncd*na);

    #pragma omp parallel for
    for (int j=0; j < n; ++j) {
        for (int ka=0; ka < na; ++ka) {
            for (int kb=0; kb < nb; ++kb) {
                int i = ka*nb + kb;
                vd[na*j + ka][kb] = i < n ? d_[n*j + i] : SINFTY ;
                vt[na*j + ka][kb] = i < n ? d_[n*i + j] : SINFTY ;
            }
        }
    }
    /* padding */
    for (int j=n; j < ncd; ++j) {
        for (int ka=0; ka < na; ++ka) {
            for (int kb=0; kb < nb; ++kb) {
                vd[na*j + ka][kb] = SINFTY;
                vt[na*j + ka][kb] = SINFTY;
            }
        }
    }

    #pragma omp parallel for
    for (int ic=0; ic < nc; ++ic) {
        for (int jc=0; jc < nc; ++jc) {
            float8_t vv[nd][nd];
            for (int id=0; id < nd; ++id) {
                for (int jd=0; jd < nd; ++jd) {
                    vv[id][jd] = f8infty;
                }
            }
            for (int ka=0; ka < na; ++ka) {
                float8_t y0 = vt[na*(jc*nd + 0) + ka];
                float8_t y1 = vt[na*(jc*nd + 1) + ka];
                float8_t y2 = vt[na*(jc*nd + 2) + ka];
                float8_t x0 = vd[na*(ic*nd + 0) + ka];
                float8_t x1 = vd[na*(ic*nd + 1) + ka];
                float8_t x2 = vd[na*(ic*nd + 2) + ka];
                vv[0][0] = min8(vv[0][0], x0 + y0);
                vv[0][1] = min8(vv[0][1], x0 + y1);
                vv[0][2] = min8(vv[0][2], x0 + y2);
                vv[1][0] = min8(vv[1][0], x1 + y0);
                vv[1][1] = min8(vv[1][1], x1 + y1);
                vv[1][2] = min8(vv[1][2], x1 + y2);
                vv[2][0] = min8(vv[2][0], x2 + y0);
                vv[2][1] = min8(vv[2][1], x2 + y1);
                vv[2][2] = min8(vv[2][2], x2 + y2);
            }
            for (int id=0; id < nd; ++id) {
                for (int jd=0; jd < nd; ++jd) {
                    int i = ic * nd + id;
                    int j = jc * nd + jd;
                    if (i < n && j < n) {
                        r[n*i + j] = hmin8(vv[id][jd]);
                    }
                }
            }
        }
    }

    std::free(vt);
    std::free(vd);
}

void step4(float* r, const float* d_, int n) {
    // vectors per input column
    int na = (n + 8 - 1) / 8;

    // input data, padded, converted to vectors
    float8_t* vd = float8_alloc(na*n);
    // input data, transposed, padded, converted to vectors
    float8_t* vt = float8_alloc(na*n);

    #pragma omp parallel for
    for (int ja = 0; ja < na; ++ja) {
        for (int i = 0; i < n; ++i) {
            for (int jb = 0; jb < 8; ++jb) {
                int j = ja * 8 + jb;
                vd[n*ja + i][jb] = j < n ? d_[n*j + i] : SINFTY;
                vt[n*ja + i][jb] = j < n ? d_[n*i + j] : SINFTY;
            }
        }
    }

    #pragma omp parallel for
    for (int ia = 0; ia < na; ++ia) {
        for (int ja = 0; ja < na; ++ja) {
            float8_t vv000 = f8infty;
            float8_t vv001 = f8infty;
            float8_t vv010 = f8infty;
            float8_t vv011 = f8infty;
            float8_t vv100 = f8infty;
            float8_t vv101 = f8infty;
            float8_t vv110 = f8infty;
            float8_t vv111 = f8infty;
            for (int k = 0; k < n; ++k) {
                constexpr int PF = 20;
                __builtin_prefetch(&vd[n*ia + k + PF]);
                __builtin_prefetch(&vt[n*ja + k + PF]);
                float8_t a000 = vd[n*ia + k];
                float8_t b000 = vt[n*ja + k];
                float8_t a100 = swap4(a000);
                float8_t a010 = swap2(a000);
                float8_t a110 = swap2(a100);
                float8_t b001 = swap1(b000);
                vv000 = min8(vv000, a000 + b000);
                vv001 = min8(vv001, a000 + b001);
                vv010 = min8(vv010, a010 + b000);
                vv011 = min8(vv011, a010 + b001);
                vv100 = min8(vv100, a100 + b000);
                vv101 = min8(vv101, a100 + b001);
                vv110 = min8(vv110, a110 + b000);
                vv111 = min8(vv111, a110 + b001);
            }
            float8_t vv[8] = { vv000, vv001, vv010, vv011, vv100, vv101, vv110, vv111 };
            for (int kb = 1; kb < 8; kb += 2) {
                vv[kb] = swap1(vv[kb]);
            }
            for (int jb = 0; jb < 8; ++jb) {
                for (int ib = 0; ib < 8; ++ib) {
                    int i = ib + ia*8;
                    int j = jb + ja*8;
                    if (j < n && i < n) {
                        r[n*i + j] = vv[ib^jb][jb];
                    }
                }
            }
        }
    }

    std::free(vt);
    std::free(vd);
}

int main() {
    bool print = false;
    constexpr int n = 2000;
    float * d = new float[n*n];
    for (int i = 0; i < n*n; ++i)
        d[i] = rand();
    float * r = new float[n*n];
    clock_t time1;
    time1 = clock();
    step1(r, d, n);
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC));
    printf("r0: %.4f\n", r[0]);
    time1 = clock();
    step2(r, d, n);
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC));
    printf("r0: %.4f\n", r[0]);
    time1 = clock();
    step3(r, d, n);
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC));
    printf("r0: %.4f\n", r[0]);
    time1 = clock();
    step4(r, d, n);
    time1 = clock() - time1;
    printf("Time: %.6f\n", (float) time1 / (CLOCKS_PER_SEC));
    printf("r0: %.4f\n", r[0]);

    delete[] d;
    delete[] r;
    if(!print)
        return 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << r[i*n + j] << " ";
        }
        std::cout << "\n";
    }
}
