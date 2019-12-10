#include <stdio.h>
#include <string.h>

void rabin_karp_search(char * str, char * pattern, int d, int q) {
    int len_str = strlen(str);
    int len_pat = strlen(pattern);
    int i, h = 1;
    int hash_s = 0;
    int hash_p = 0;

    for (i = 0; i < len_pat - 1; i++)
        h = d * h % q;
    for (i = 0; i < len_pat; i++) {
        hash_p = (d * hash_p + pattern[i]) % q;
        hash_s = (d * hash_s + str[i]) % q;
    }

    for (i = 0; i <= len_str - len_pat; i++) {
        if (hash_p == hash_s) {
            int j;
            for (j = 0; j < len_pat; j++) {
                if (pattern[j] != str[i + j])
                    break;
            }
            if (len_pat == j)
                printf("--Pattern is found at: %d\n", i);
        }
        hash_s = (d * (hash_s - str[i] * h) + str[i + len_pat]) % q;
        if (hash_s < 0)
            hash_s = hash_s + q;
    }
}

int main() {
    char str[] = "AABCAB12AFAABCABFFEGABCAB";
    char pat1[] = "ABCAB";
    char pat2[] = "FFF";
    char pat3[] = "CAB";

    printf("String test: %s\n", str);
    printf("Test1: search pattern %s\n", pat1);
    rabin_karp_search(str, pat1, 256, 29);
    printf("Test2: search pattern %s\n", pat2);
    rabin_karp_search(str, pat2, 256, 29);
    printf("Test3: search pattern %s\n", pat3);
    rabin_karp_search(str, pat3, 256, 29);
    return 0;
}