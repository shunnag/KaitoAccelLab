// liblz4 の LZ4_decompress_safe を 16 thread（静的分割）で独立 block に適用する CPU 比較器。
// frame は accel-lab lz4-make-frame / lz4 -BI の独立 block、content size 付きを前提にする。
// 使い方: lz4-cpu-liblz4 <file.lz4> [threads] [rounds]
#include <lz4.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

typedef struct { uint32_t srcOff, srcLen, dstOff, dstLen, stored; } Block;
typedef struct { const uint8_t *src; uint8_t *dst; Block *blocks; int begin, end; int ok; } Job;

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static void *worker(void *p) {
    Job *j = p; j->ok = 1;
    for (int i = j->begin; i < j->end; i++) {
        Block *b = &j->blocks[i];
        if (b->stored) { memcpy(j->dst + b->dstOff, j->src + b->srcOff, b->srcLen); continue; }
        int n = LZ4_decompress_safe((const char *)j->src + b->srcOff, (char *)j->dst + b->dstOff, (int)b->srcLen, (int)b->dstLen);
        if (n != (int)b->dstLen) { j->ok = 0; }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file.lz4> [threads] [rounds]\n", argv[0]); return 1; }
    int threads = argc > 2 ? atoi(argv[2]) : 16, rounds = argc > 3 ? atoi(argv[3]) : 5;
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror("open"); return 1; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *src = malloc(len); if (fread(src, 1, len, f) != (size_t)len) { perror("read"); return 1; } fclose(f);
    // frame header
    if (len < 7 || src[0] != 0x04 || src[1] != 0x22 || src[2] != 0x4D || src[3] != 0x18) { fprintf(stderr, "not an LZ4 frame\n"); return 1; }
    uint8_t flg = src[4], bd = src[5]; size_t pos = 6;
    int independent = (flg >> 5) & 1, blockChecksum = (flg >> 4) & 1, hasSize = (flg >> 3) & 1, dictID = flg & 1;
    if (!independent || !hasSize) { fprintf(stderr, "need independent blocks with content size\n"); return 1; }
    uint64_t contentSize; memcpy(&contentSize, src + pos, 8); pos += 8;
    if (dictID) pos += 4;
    pos += 1; // header checksum
    size_t blockMax = 1u << (8 + 2 * ((bd >> 4) & 7));
    Block *blocks = malloc(sizeof(Block) * (len / 16 + 16)); int count = 0;
    while (pos + 4 <= (size_t)len) {
        uint32_t size; memcpy(&size, src + pos, 4); pos += 4;
        if (size == 0) break;
        Block *b = &blocks[count++];
        b->stored = size >> 31; b->srcLen = size & 0x7FFFFFFF; b->srcOff = (uint32_t)pos; pos += b->srcLen;
        if (blockChecksum) pos += 4;
    }
    // 展開後 size: 最終 block 以外は均一と仮定して検証する（lz4-make-frame の出力）。
    uint64_t uniform = (contentSize + count - 1) / count; uint64_t off = 0;
    for (int i = 0; i < count; i++) { blocks[i].dstOff = (uint32_t)off; uint64_t n = contentSize - off < uniform ? contentSize - off : uniform; blocks[i].dstLen = (uint32_t)n; off += n; }
    uint8_t *dst = malloc(contentSize); memset(dst, 0, contentSize);
    double *times = malloc(sizeof(double) * rounds);
    for (int r = 0; r < rounds; r++) {
        pthread_t th[64]; Job jobs[64];
        double t0 = now();
        for (int t = 0; t < threads; t++) {
            jobs[t] = (Job){ src, dst, blocks, count * t / threads, count * (t + 1) / threads, 1 };
            pthread_create(&th[t], NULL, worker, &jobs[t]);
        }
        int ok = 1;
        for (int t = 0; t < threads; t++) { pthread_join(th[t], NULL); ok &= jobs[t].ok; }
        times[r] = now() - t0;
        if (!ok) { fprintf(stderr, "decode error\n"); return 1; }
    }
    qsort(times, rounds, sizeof(double), cmpd);
    // 検証用 checksum（FNV-1a）
    uint64_t h = 1469598103934665603ULL; for (uint64_t i = 0; i < contentSize; i++) { h ^= dst[i]; h *= 1099511628211ULL; }
    printf("cpu-%dthread-liblz4\t%llu\t%.9f\t%.6f\tLZ4_decompress_safe per block; static partition; blocks=%d blockMax=%zu; fnv1a=%016llx\n",
           threads, (unsigned long long)contentSize, times[rounds / 2], contentSize / times[rounds / 2] / 1e9, count, blockMax, (unsigned long long)h);
    return 0;
}
