#include <stdio.h>
#include <stdlib.h>

#define MAXELEMENTS 1000

struct Dict {
    void * elements[MAXELEMENTS];
    int number_of_elements;
};

struct Dict * create_dict(void);

int add_item_label(struct Dict * , char label[], void * );

int add_item_index(struct Dict * , int index, void * );

void * get_element_label(struct Dict * , char[]);

void * get_element_index(struct Dict * , int);

void destroy(struct Dict * );

struct Dict * create_dict(void) {
    struct Dict * p_dic = malloc(sizeof(struct Dict));
    if (p_dic) {
        p_dic -> number_of_elements = 0;

        for (int i = 0; i < MAXELEMENTS; i++) {
            p_dic -> elements[i] = NULL;
        }

        return p_dic;
    } else {
        printf("unable to create a dictionary\n");
        return NULL;
    }
}

int get_hash(char s[]) {
    unsigned int hash_code = 0;

    for (int counter = 0; s[counter] != '\0'; counter++) {
        hash_code = s[counter] + (hash_code << 6) + (hash_code << 16) - hash_code;
    }

    return hash_code % MAXELEMENTS;
}

int add_item_label(struct Dict * dic, char label[], void * item) {
    unsigned int index = get_hash(label);

    if (index < MAXELEMENTS) {
        dic -> elements[index] = item;
        return 0;
    }

    return -1;
}

int add_item_index(struct Dict * dic, int index, void * item) {
    if (!dic -> elements[index]) {
        dic -> elements[index] = item;
        return 0;
    }

    return -1;
}

void * get_element_label(struct Dict * dict, char s[]) {
    int index = get_hash(s);
    if (dict -> elements[index]) {
        return dict -> elements[index];
    }

    printf("None entry at given label\n");
    return NULL;
}

void * get_element_index(struct Dict * dict, int index) {
    if (index >= 0 && index < MAXELEMENTS) {
        return dict -> elements[index];
    }

    printf("index out of bounds!\n");
    return NULL;
}

void destroy(struct Dict * dict) {
    free(dict);
}

int main(void) {
    struct Dict * testObj1;
    struct Dict * testObj2;

    int value = 28;

    testObj1 = create_dict();
    testObj2 = create_dict();

    add_item_label(testObj1, "age", & value);
    add_item_label(testObj2, "name", "Christian");

    printf("My age is %d\n", *((int * ) get_element_label(testObj1, "age")));
    printf("My name is %s\n", get_element_label(testObj2, "name"));

    if (!add_item_index(testObj1, 0, & value)) {
        printf("My age at index %d is %d\n", 0, *((int * ) get_element_index(testObj1, 0)));
    }

    destroy(testObj1);
    destroy(testObj2);

    return 0;
}