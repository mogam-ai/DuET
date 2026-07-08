// Program: DuET v1.1.0
// Author: Sungho Lee, Jae-Won Lee
// Affiliation: MOGAM Institute for Biomedical Research
// Contact: https://github.com/mogam-ai/DuET/issues
// Citation: TBD

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <libgen.h>

int process_file(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Cannot open file");
        return 1;
    }
    
    size_t capacity = 100;
    size_t size = 0;
    int *lengths = malloc(capacity * sizeof(int));
    if (!lengths) {
        perror("Memory allocation failed");
        fclose(file);
        return 1;
    }

    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        if (!fgets(line, sizeof(line), file)) break;
        size_t length = strlen(line) - 1;
        if (line[length] == '\n') line[length] = '\0';
        
        if (size >= capacity) {
            capacity *= 2;
            lengths = realloc(lengths, capacity * sizeof(int));
            if (!lengths) {
                perror("Memory allocation failed");
                fclose(file);
                return 1;
            }
        }
        
        lengths[size++] = length;
        
        fgets(line, sizeof(line), file);
        fgets(line, sizeof(line), file);
    }

    fclose(file);

    double sum = 0;
    double sum_sq = 0;

    for (size_t i = 0; i < size; i++) {
        sum += lengths[i];
        sum_sq += lengths[i] * lengths[i];
    }

    double mean = sum / size;
    double variance = (sum_sq / size) - (mean * mean);
    double stddev = sqrt(variance);

    printf("%s\t%zu\t%.3f\t%.3f\n", basename((char *)filename), size, mean, stddev);

    free(lengths);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <fastq_file1> <fastq_file2> ...\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (process_file(argv[i]) != 0) {
            fprintf(stderr, "Error processing file %s\n", argv[i]);
            return 1;
        }
    }

    return 0;
}

