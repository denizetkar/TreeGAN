#include <stdio.h>

long long sdbm(char[]);
long long djb2(char[]);
char xor8(char[]);
int adler_32(char[]);
unsigned int crc32(char[]);

int main(void) {
	char s[] = "name";

	printf("sdbm: %s --> %lld\n", s, sdbm(s));
	printf("djb2: %s --> %lld\n", s, djb2(s));
	printf("xor8: %s --> %i\n", s, xor8(s));
	printf("adler_32: %s --> %i\n", s, adler_32(s));
	printf("crc32: %s --> %i\n", s, crc32(s));

	return 0;
}

long long sdbm(char s[]) {
	long long hash = 0;
	int i = 0;
	while (s[i] != '\0') {
		hash = s[i] + (hash << 6) + (hash << 16) - hash;
		i++;
	}
	return hash;
}

long long djb2(char s[]) {
	long long hash = 5381;
	int i = 0;
	while (s[i] != '\0') {
		hash = ((hash << 5) + hash) + s[i];
		i++;
	}
	return hash;
}

char xor8(char s[]) {
	int hash = 0;
	int i = 0;
	while (s[i] != '\0') {
		hash = (hash + s[i]) & 0xff;
		i++;
	}
	return (((hash ^ 0xff) + 1) & 0xff);
}

int adler_32(char s[]) {
	int a = 1;
	int b = 0;
	const int MODADLER = 65521;

	int i = 0;
	while (s[i] != '\0') {
		a = (a + s[i]) % MODADLER;
		b = (b + a) % MODADLER;
		i++;
	}
	return (b << 16) | a;
}

unsigned int crc32(char data[]) {
	int i = 0;
	unsigned int crc = 0xffffffff;
	while (data[i] != '\0') {
		unsigned char byte = data[i];
		crc = crc ^ byte;
		for (int j = 8; j > 0; --j)
			crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));

		i++;
	}
	return crc ^ 0xffffffff;
}