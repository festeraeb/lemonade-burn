// Reference harness (not part of lemonade-burn; links llama.cpp's ggml-base). Two modes:
//   iqref deq   <model.gguf> <tensor> <rows> <out_prefix>          dump the tensor's first rows (raw .bin) and to_float (.f32)
//   iqref quant <bf16.gguf> <tensor> <rows> <type> <out_prefix>    quantize the first rows of a BF16/F32 tensor to <type>
//                                                                  with ggml (uniform imatrix), dump .bin and to_float .f32
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void dump(const char *pfx, const char *ext, const void *p, size_t n) {
    char fn[1024]; snprintf(fn, sizeof fn, "%s.%s", pfx, ext); FILE *o = fopen(fn, "wb"); fwrite(p, 1, n, o); fclose(o);
}
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 2; }
    struct gguf_init_params p = { .no_alloc = true, .ctx = NULL };
    struct gguf_context *g = gguf_init_from_file(argv[2], p);
    if (!g) return 3;
    int64_t t = gguf_find_tensor(g, argv[3]);
    if (t < 0) { fprintf(stderr, "no tensor %s\n", argv[3]); return 4; }
    enum ggml_type ty = gguf_get_tensor_type(g, t);
    size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, t);
    long cols = atol(getenv("COLS")), rows = atol(argv[4]);
    size_t rb = ggml_row_size(ty, cols);
    FILE *f = fopen(argv[2], "rb"); fseeko(f, off, SEEK_SET);
    unsigned char *buf = malloc(rb * rows); if (fread(buf, 1, rb * rows, f) != rb * rows) return 5;
    float *x = malloc(sizeof(float) * cols * rows);
    ggml_get_type_traits(ty)->to_float(buf, x, cols * rows);
    if (!strcmp(argv[1], "deq")) {
        dump(argv[5], "bin", buf, rb * rows); dump(argv[5], "f32", x, sizeof(float) * cols * rows);
        printf("%s %s rows %ld cols %ld row_bytes %zu\n", argv[3], ggml_type_name(ty), rows, cols, rb);
        return 0;
    }
    enum ggml_type q = GGML_TYPE_COUNT;
    for (int i = 0; i < GGML_TYPE_COUNT; i++) if (ggml_type_name(i) && !strcmp(ggml_type_name(i), argv[5])) q = i;
    if (q == GGML_TYPE_COUNT) { fprintf(stderr, "type?\n"); return 6; }
    ggml_quantize_init(q);
    float *im = malloc(sizeof(float) * cols); for (long i = 0; i < cols; i++) im[i] = 1.0f;
    size_t qrb = ggml_row_size(q, cols);
    unsigned char *qb = malloc(qrb * rows);
    ggml_quantize_chunk(q, x, qb, 0, rows, cols, im);
    float *y = malloc(sizeof(float) * cols * rows);
    ggml_get_type_traits(q)->to_float(qb, y, cols * rows);
    dump(argv[6], "bin", qb, qrb * rows); dump(argv[6], "f32", y, sizeof(float) * cols * rows);
    printf("%s from %s -> %s rows %ld cols %ld row_bytes %zu\n", argv[3], ggml_type_name(ty), argv[5], rows, cols, qrb);
    return 0;
}
