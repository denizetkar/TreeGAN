#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALPHABET_SIZE 26

struct TrieNode {
	struct TrieNode* children[ALPHABET_SIZE];
	char character;
	int isEndOfWord;

};

struct TrieNode* createTrieNode() {
	struct TrieNode* node;
	node = malloc(sizeof(struct TrieNode));
	node->isEndOfWord = 0;
	int i = 0;
	while (i < ALPHABET_SIZE) {
		node->children[i] = NULL;
		i++;
	}
	return node;
}

void insert(struct TrieNode* root, char* word) {
	if ((strlen(word) - 1) != 0) {
		char character = *word;
		if (root->children[character - 97] == NULL) {
			struct TrieNode* node = NULL;
			node = createTrieNode();
			node->character = character;
			root->children[character - 97] = node;
		}
		word++;
		insert(root->children[character - 97], word);
	}
	else {
		root->isEndOfWord = 1;
	}
	return;
}

struct TrieNode* search(struct TrieNode* root, char* word) {
	struct TrieNode* temp;
	while (*word != '\0') {
		char character = *word;
		if (root->children[character - 97] != NULL) {
			temp = root->children[character - 97];
			word++;
			root = temp;
		}
		else {
			printf("No possible words!!\n");
			return NULL;
		}
	}
	return root;
}

void printArray(char chars[], int len) {
	int i;
	for (i = 0; i < len; i++) {
		printf("%c", chars[i]);
	}
	printf("\n");
}

void printPathsRecur(struct TrieNode* node, char prefix[], int filledLen) {
	if (node == NULL) return;

	prefix[filledLen] = node->character;
	filledLen++;

	if (node->isEndOfWord) {
		printArray(prefix, filledLen);
	}

	int i;
	for (i = 0; i < ALPHABET_SIZE; i++) {
		printPathsRecur(node->children[i], prefix, filledLen);
	}
}

void traverse(char prefix[], struct TrieNode* root) {
	struct TrieNode* temp = NULL;
	temp = search(root, prefix);
	int j = 0;
	while (prefix[j] != '\0') {
		j++;
	}
	printPathsRecur(temp, prefix, j - 1);
}

#define NUMBER_OF_WORDS (354935)
#define INPUT_WORD_SIZE (100)

char* receiveInput(char* s) {
	scanf("%99s", s);
	return s;
}

int main() {
	int word_count = 0;
	char* words[NUMBER_OF_WORDS];
	struct _iobuf* fp = fopen("dictionary.txt", "r");

	if (fp == 0) {
		fprintf(stderr, "Error while opening dictionary file");
		exit(1);
	}

	words[word_count] = malloc(INPUT_WORD_SIZE);

	while (fgets(words[word_count], INPUT_WORD_SIZE, fp)) {
		word_count++;
		words[word_count] = malloc(INPUT_WORD_SIZE);
	}

	struct TrieNode* root = NULL;
	root = createTrieNode();
	int i;
	for (i = 0; i < NUMBER_OF_WORDS; i++) {
		insert(root, words[i]);
	}

	while (1) {
		printf("Enter keyword: ");
		char str[100];
		receiveInput(str);
		printf("\n==========================================================\n");
		printf("\n********************* Possible Words ********************\n");

		traverse(str, root);

		printf("\n==========================================================\n");
	}
}