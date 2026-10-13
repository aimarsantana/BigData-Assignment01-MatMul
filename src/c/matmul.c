/*
 * Basic dense matrix multiplication C = A x B (triple loop, i-j-k order).
 *
 * Storage: one contiguous row-major block of n*n doubles per matrix plus an
 * array of n row pointers into that block, so the kernel uses the same
 * A[i][k] * B[k][j] indexing as the Python and Java versions.
 * Numeric type: double (IEEE-754 binary64).
 *
 * Build: gcc -O2 -std=c11 -Wall -o build/matmul src/c/matmul.c
 * Usage: ./build/matmul A.bin B.bin n warmup reps batch [C_out.bin]
 *
 * The timed region contains ONLY the triple loop.
 * Timer: clock_gettime(CLOCK_MONOTONIC).
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double **alloc_matrix(int n) {
    double *block = calloc((size_t)n * n, sizeof(double));
    double **rows = malloc((size_t)n * sizeof(double *));
    if (!block || !rows) {
        fprintf(stderr, "allocation failed for n=%d\n", n);
        exit(3);
    }
    for (int i = 0; i < n; i++) rows[i] = block + (size_t)i * n;
    return rows;
}

static void free_matrix(double **m) {
    free(m[0]);
    free(m);
}

/* Input files are little-endian binary64; x86-64 is little-endian. */
static double **read_matrix(const char *path, int n) {
    double **m = alloc_matrix(n);
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(4); }
    size_t got = fread(m[0], sizeof(double), (size_t)n * n, f);
    fclose(f);
    if (got != (size_t)n * n) {
        fprintf(stderr, "short read on %s\n", path);
        exit(5);
    }
    return m;
}

static void write_matrix(const char *path, double **m, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(6); }
    fwrite(m[0], sizeof(double), (size_t)n * n, f);
    fclose(f);
}

/* Straightforward O(n^3) triple loop, i-j-k order, scalar accumulator. */
static void matmul(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int k = 0; k < n; k++) {
                s += A[i][k] * B[k][j];
            }
            C[i][j] = s;
        }
    }
}

static double checksum(double **C, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        double rs = 0.0;
        for (int j = 0; j < n; j++) rs += C[i][j];
        total += rs;
    }
    return total;
}

/* Peak resident set size (VmHWM) of this process, from /proc/self/status (Linux). */
static long peak_rss_kib(void) {
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long kib = -1;
    if (!f) return -1;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            kib = strtol(line + 6, NULL, 10);
            break;
        }
    }
    fclose(f);
    return kib;
}

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s A.bin B.bin n warmup reps batch [C_out.bin]\n", argv[0]);
        return 2;
    }
    const char *a_path = argv[1], *b_path = argv[2];
    int n = atoi(argv[3]), warmup = atoi(argv[4]), reps = atoi(argv[5]), batch = atoi(argv[6]);
    const char *out_path = argc > 7 ? argv[7] : NULL;

#ifdef __VERSION__
    printf("INFO,c,compiler,gcc %s\n", __VERSION__);
#endif
#ifdef CFLAGS_STR
    printf("INFO,c,cflags,%s\n", CFLAGS_STR);
#endif
    double **A = read_matrix(a_path, n);
    double **B = read_matrix(b_path, n);
    double **C = alloc_matrix(n); /* allocated (and zeroed) outside the timer */

    int total = warmup + reps;
    for (int r = 0; r < total; r++) {
        const char *phase = r < warmup ? "warmup" : "measured";
        int idx = r < warmup ? r : r - warmup;
        long long t0 = now_ns();
        for (int b = 0; b < batch; b++) matmul(A, B, C, n);
        long long t1 = now_ns();
        double cs = checksum(C, n); /* consume the result (outside timer) */
        printf("SAMPLE,c,%d,%s,%d,%d,%.1f,%.17g\n", n, phase, idx, batch,
               (double)(t1 - t0) / batch, cs);
        fflush(stdout);
    }
    printf("MEM,c,%d,vmhwm_kib,%ld\n", n, peak_rss_kib());
    if (out_path) write_matrix(out_path, C, n);
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    return 0;
}
