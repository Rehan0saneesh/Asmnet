#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* ---- Asmnet math library (System V AMD64 ABI) ---- */
long   m_add(long a, long b);
long   m_sub(long a, long b);
long   m_mul(long a, long b);
long   m_div(long a, long b);
long   m_mod(long a, long b);
long   m_neg(long a);
long   m_pow_i(long base, long exp);
long   m_sqrt(long n);
long   m_factorial(long n);
long   m_gcd(long a, long b);
long   m_lcm(long a, long b);
long   m_min(long a, long b);
long   m_max(long a, long b);
long   m_abs(long a);

double m_fadd(double a, double b);
double m_fsub(double a, double b);
double m_fmul(double a, double b);
double m_fdiv(double a, double b);
double m_fneg(double a);
double m_fabs(double a);

/* Matrix (double*, int, int, ...) */
void   m_matmul(double *A, int ra, int ca, double *B, int rb, int cb, double *C);
void   m_matadd(double *A, int r, int c, double *B, double *C);
void   m_matsub(double *A, int r, int c, double *B, double *C);
void   m_transpose(double *A, int r, int c, double *C);
double m_det2(double a, double b, double c, double d);
double m_det3(double a, double b, double c,
              double d, double e, double f,
              double g, double h, double i);

/* Calculus */
double m_integrate(double (*f)(double), double a, double b, int n);
double m_derivative(double (*f)(double), double x, double h);

/* Complex: re=xmm0, im=xmm1; operand2 re=xmm2, im=xmm3 */
void   m_cadd(double are, double aim, double bre, double bim, double *out_re, double *out_im);
void   m_csub(double are, double aim, double bre, double bim, double *out_re, double *out_im);
void   m_cmul(double are, double aim, double bre, double bim, double *out_re, double *out_im);
void   m_cdiv(double are, double aim, double bre, double bim, double *out_re, double *out_im);
double m_cabs(double are, double aim);
void   m_cconj(double are, double aim, double *out_re, double *out_im);
void   m_cneg(double are, double aim, double *out_re, double *out_im);

static int tests = 0, failures = 0;

static void check_int(const char *name, long got, long want) {
    tests++;
    if (got != want) {
        printf("FAIL %s: got %ld want %ld\n", name, got, want);
        failures++;
    } else {
        printf("PASS %s\n", name);
    }
}

static void check_dbl(const char *name, double got, double want, double tol) {
    tests++;
    if (fabs(got - want) > tol) {
        printf("FAIL %s: got %g want %g\n", name, got, want);
        failures++;
    } else {
        printf("PASS %s\n", name);
    }
}

/* ---- calculus test functions ---- */
static double func_linear(double x) { return 2.0 * x; }   /* int 0..10 = 100 */
static double func_quad(double x)  { return x * x; }       /* d/dx at 3 = 6 */
static double func_cubic(double x) { return x * x * x; }   /* int 0..2 = 4 */

int main(void) {
    printf("=== Integer arithmetic ===\n");
    check_int("add",       m_add(2, 3), 5);
    check_int("sub",       m_sub(5, 3), 2);
    check_int("mul",       m_mul(4, 5), 20);
    check_int("div",       m_div(10, 3), 3);
    check_int("mod",       m_mod(10, 3), 1);
    check_int("neg",       m_neg(7), -7);
    check_int("pow_i",     m_pow_i(2, 10), 1024L);
    check_int("sqrt",      m_sqrt(25), 5L);
    check_int("factorial", m_factorial(5), 120L);
    check_int("gcd",       m_gcd(48, 18), 6L);
    check_int("lcm",       m_lcm(12, 18), 36L);
    check_int("min",       m_min(3, 7), 3L);
    check_int("max",       m_max(3, 7), 7L);
    check_int("abs",       m_abs(-5), 5L);

    printf("\n=== Floating point ===\n");
    check_dbl("fadd",   m_fadd(1.5, 2.5), 4.0, 1e-9);
    check_dbl("fsub",   m_fsub(5.0, 2.0), 3.0, 1e-9);
    check_dbl("fmul",   m_fmul(2.5, 4.0), 10.0, 1e-9);
    check_dbl("fdiv",   m_fdiv(10.0, 4.0), 2.5, 1e-9);
    check_dbl("fneg",   m_fneg(3.0), -3.0, 1e-9);
    check_dbl("fabs",   m_fabs(-3.5), 3.5, 1e-9);

    printf("\n=== Matrices ===\n");
    {
        double A[4] = {1, 2, 3, 4};       /* 2x2 */
        double B[4] = {5, 6, 7, 8};
        double C[4] = {0};
        /* [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]] */
        m_matmul(A, 2, 2, B, 2, 2, C);
        check_dbl("matmul[0]", C[0], 19, 1e-9);
        check_dbl("matmul[1]", C[1], 22, 1e-9);
        check_dbl("matmul[2]", C[2], 43, 1e-9);
        check_dbl("matmul[3]", C[3], 50, 1e-9);

        double M[4] = {1, 2, 3, 4};
        double N[4] = {5, 6, 7, 8};
        double R[4] = {0};
        m_matadd(M, 2, 2, N, R);            /* [[6,8],[10,12]] */
        check_dbl("matadd[0]", R[0], 6, 1e-9);
        check_dbl("matadd[1]", R[1], 8, 1e-9);
        check_dbl("matadd[2]", R[2], 10, 1e-9);
        check_dbl("matadd[3]", R[3], 12, 1e-9);

        m_matsub(M, 2, 2, N, R);            /* [[-4,-4],[-4,-4]] */
        check_dbl("matsub[0]", R[0], -4, 1e-9);
        check_dbl("matsub[1]", R[1], -4, 1e-9);
        check_dbl("matsub[2]", R[2], -4, 1e-9);
        check_dbl("matsub[3]", R[3], -4, 1e-9);

        double T[4] = {0};
        m_transpose(M, 2, 2, T);            /* [[1,3],[2,4]] */
        check_dbl("trans[0]", T[0], 1, 1e-9);
        check_dbl("trans[1]", T[1], 3, 1e-9);
        check_dbl("trans[2]", T[2], 2, 1e-9);
        check_dbl("trans[3]", T[3], 4, 1e-9);
    }

    printf("\n=== Determinants ===\n");
    check_dbl("det2", m_det2(1, 2, 3, 4), -2.0, 1e-9);   /* 1*4 - 2*3 = -2 */
    check_dbl("det3", m_det3(1,2,3, 4,5,6, 7,8,10), -3.0, 1e-9);

    printf("\n=== Calculus ===\n");
    check_dbl("integrate_linear",
              m_integrate(func_linear, 0.0, 10.0, 1000), 100.0, 1e-6);
    check_dbl("integrate_cubic",
              m_integrate(func_cubic, 0.0, 2.0, 10000), 4.0, 1e-4);
    check_dbl("derivative_quad",
              m_derivative(func_quad, 3.0, 1e-5), 6.0, 1e-6);

    printf("\n=== Complex ===\n");
    {
        double re, im;
        m_cadd(1, 2, 3, 4, &re, &im);
        check_dbl("cadd_re", re, 4, 1e-9);
        check_dbl("cadd_im", im, 6, 1e-9);

        m_csub(5, 6, 2, 1, &re, &im);
        check_dbl("csub_re", re, 3, 1e-9);
        check_dbl("csub_im", im, 5, 1e-9);

        /* (1+2i)(3+4i) = -5 + 10i */
        m_cmul(1, 2, 3, 4, &re, &im);
        check_dbl("cmul_re", re, -5, 1e-9);
        check_dbl("cmul_im", im, 10, 1e-9);

        /* (1+0i)/(0+1i) = 0 - 1i */
        m_cdiv(1, 0, 0, 1, &re, &im);
        check_dbl("cdiv_re", re, 0, 1e-9);
        check_dbl("cdiv_im", im, -1, 1e-9);

        check_dbl("cabs", m_cabs(3, 4), 5.0, 1e-9);

        m_cconj(3, 4, &re, &im);
        check_dbl("conj_re", re, 3, 1e-9);
        check_dbl("conj_im", im, -4, 1e-9);

        m_cneg(3, 4, &re, &im);
        check_dbl("cneg_re", re, -3, 1e-9);
        check_dbl("cneg_im", im, -4, 1e-9);
    }

    printf("\n=== Results: %d tests, %d failures ===\n", tests, failures);
    return failures ? 1 : 0;
}
