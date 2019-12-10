#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#ifndef min
#define min(a, b) ((a) <= (b) ? (a) : (b))
#endif

#define swap(a, b)      \
    {                   \
        char* t = x[a]; \
        x[a] = x[b];    \
        x[b] = t;       \
    }
#define i2c(i) x[i][depth]

void vecswap(int i, int j, int n, char* x[])
{
	while (n-- > 0) {
		swap(i, j);
		i++;
		j++;
	}
}

void ssort1(char* x[], int n, int depth)
{
	int a, b, c, d, r, v;
	if (n <= 1)
		return;
	a = rand() % n;
	swap(0, a);
	v = i2c(0);
	a = b = 1;
	c = d = n - 1;
	for (;;) {
		while (b <= c && (r = i2c(b) - v) <= 0) {
			if (r == 0) {
				swap(a, b);
				a++;
			}
			b++;
		}
		while (b <= c && (r = i2c(c) - v) >= 0) {
			if (r == 0) {
				swap(c, d);
				d--;
			}
			c--;
		}
		if (b > c)
			break;
		swap(b, c);
		b++;
		c--;
	}
	r = min(a, b - a);
	vecswap(0, b - r, r, x);
	r = min(d - c, n - d - 1);
	vecswap(b, n - r, r, x);
	r = b - a;
	ssort1(x, r, depth);
	if (i2c(r) != 0)
		ssort1(x + r, a + n - d - 1, depth + 1);
	r = d - c;
	ssort1(x + n - r, r, depth);
}

void ssort1main(char* x[], int n)
{
	ssort1(x, n, 0);
}

void vecswap2(char** a, char** b, int n)
{
	while (n-- > 0) {
		char* t = * a;
		* a++ = * b;
		* b++ = t;
	}
}

#define swap2(a, b)  \
    {                \
        t = *(a);    \
        *(a) = *(b); \
        *(b) = t;    \
    }
#define ptr2char(i) (*(*(i) + depth))

char** med3func(char** a, char** b, char** c, int depth)
{
	int va, vb, vc;
	if ((va = ptr2char(a)) == (vb = ptr2char(b)))
		return a;
	if ((vc = ptr2char(c)) == va || vc == vb)
		return c;
	return va < vb ? (vb < vc ? b : (va < vc ? c : a))
		: (vb > vc ? b : (va < vc ? a : c));
}
#define med3(a, b, c) med3func(a, b, c, depth)

void inssort(char** a, int n, int d)
{
	char** pi, ** pj, * s, * t;
	for (pi = a + 1; --n > 0; pi++)
		for (pj = pi; pj > a; pj--) {
			for (s = *(pj - 1) + d, t = * pj + d; * s == * t && * s != 0; s++, t++)
				;
			if (* s <= * t)
				break;
			swap2(pj, pj - 1);
		}
}

void ssort2(char** a, int n, int depth)
{
	int d, r, partval;
	char** pa, ** pb, ** pc, ** pd, ** pl, ** pm, ** pn, * t;
	if (n < 10) {
		inssort(a, n, depth);
		return;
	}
	pl = a;
	pm = a + (n / 2);
	pn = a + (n - 1);
	if (n > 30) {
		d = (n / 8);
		pl = med3(pl, pl + d, pl + 2 * d);
		pm = med3(pm - d, pm, pm + d);
		pn = med3(pn - 2 * d, pn - d, pn);
	}
	pm = med3(pl, pm, pn);
	swap2(a, pm);
	partval = ptr2char(a);
	pa = pb = a + 1;
	pc = pd = a + n - 1;
	for (;;) {
		while (pb <= pc && (r = ptr2char(pb) - partval) <= 0) {
			if (r == 0) {
				swap2(pa, pb);
				pa++;
			}
			pb++;
		}
		while (pb <= pc && (r = ptr2char(pc) - partval) >= 0) {
			if (r == 0) {
				swap2(pc, pd);
				pd--;
			}
			pc--;
		}
		if (pb > pc)
			break;
		swap2(pb, pc);
		pb++;
		pc--;
	}
	pn = a + n;
	r = min(pa - a, pb - pa);
	vecswap2(a, pb - r, r);
	r = min(pd - pc, pn - pd - 1);
	vecswap2(pb, pn - r, r);
	if ((r = pb - pa) > 1)
		ssort2(a, r, depth);
	if (ptr2char(a + r) != 0)
		ssort2(a + r, pa - a + pn - pd - 1, depth + 1);
	if ((r = pd - pc) > 1)
		ssort2(a + n - r, r, depth);
}

void ssort2main(char** a, int n) { ssort2(a, n, 0); }

struct tnode {
	char splitchar;
	struct tnode* lokid, * eqkid, * hikid;
};
struct tnode* root;

struct tnode* insert1(struct tnode* p, char* s)
{
	if (p == 0) {
		p = (struct tnode*)malloc(sizeof(struct tnode));
		p->splitchar = * s;
		p->lokid = p->eqkid = p->hikid = 0;
	}
	if (* s < p->splitchar)
		p->lokid = insert1(p->lokid, s);
	else if (* s == p->splitchar) {
		if (* s != 0)
			p->eqkid = insert1(p->eqkid, ++s);
	}
	else
		p->hikid = insert1(p->hikid, s);
	return p;
}

void cleanup1(struct tnode* p)
{
	if (p) {
		cleanup1(p->lokid);
		cleanup1(p->eqkid);
		cleanup1(p->hikid);
		free(p);
	}
}

#define BUFSIZE 1000

struct tnode* buf;
int bufn, freen;
void* freearr[10000];
int storestring = 0;

void insert2(char* s)
{
	int d;
	char* instr = s;

	struct tnode* pp, ** p;
	p = &root;
	while (pp = * p) {
		if ((d = * s - pp->splitchar) == 0) {
			if (* s++ == 0)
				return;
			p = &(pp->eqkid);
		}
		else if (d < 0)
			p = &(pp->lokid);
		else
			p = &(pp->hikid);
	}
	for (;;) {
		if (bufn-- == 0) {
			buf = (struct tnode*)malloc(BUFSIZE * sizeof(struct tnode));
			freearr[freen++] = (void*)buf;
			bufn = BUFSIZE - 1;
		}
		* p = buf++;
		pp = * p;
		pp->splitchar = * s;
		pp->lokid = pp->eqkid = pp->hikid = 0;
		if (* s++ == 0) {
			if (storestring)
				pp->eqkid = (struct tnode*)instr;
			return;
		}
		p = &(pp->eqkid);
	}
}
void cleanup2()
{
	int i;
	for (i = 0; i < freen; i++)
		free(freearr[i]);
}

int search1(char* s)
{
	struct tnode* p;
	p = root;
	while (p) {
		if (* s < p->splitchar)
			p = p->lokid;
		else if (* s == p->splitchar) {
			if (* s++ == 0)
				return 1;
			p = p->eqkid;
		}
		else
			p = p->hikid;
	}
	return 0;
}

int search2(char* s)
{
	int d, sc;
	struct tnode* p;
	sc = * s;
	p = root;
	while (p) {
		if ((d = sc - p->splitchar) == 0) {
			if (sc == 0)
				return 1;
			sc = *++s;
			p = p->eqkid;
		}
		else if (d < 0)
			p = p->lokid;
		else
			p = p->hikid;
	}
	return 0;
}

int nodecnt;
char* srcharr[100000];
int srchtop;

void pmsearch(struct tnode* p, char* s)
{
	if (!p)
		return;
	nodecnt++;
	if (* s == '.' || * s < p->splitchar)
		pmsearch(p->lokid, s);
	if (* s == '.' || * s == p->splitchar)
		if (p->splitchar && * s)
			pmsearch(p->eqkid, s + 1);
	if (* s == 0 && p->splitchar == 0)
		srcharr[srchtop++] = (char*)p->eqkid;
	if (* s == '.' || * s > p->splitchar)
		pmsearch(p->hikid, s);
}

void nearsearch(struct tnode* p, char* s, int d)
{
	if (!p || d < 0)
		return;
	nodecnt++;
	if (d > 0 || * s < p->splitchar)
		nearsearch(p->lokid, s, d);
	if (p->splitchar == 0) {
		if ((int)strlen(s) <= d)
			srcharr[srchtop++] = (char*)p->eqkid;
	}
	else
		nearsearch(p->eqkid, * s ? s + 1 : s,
		(* s == p->splitchar) ? d : d - 1);
	if (d > 0 || * s > p->splitchar)
		nearsearch(p->hikid, s, d);
}

#define NUMBER_OF_STRING 3

int main(int argc, char* argv[])
{

	char* arr[NUMBER_OF_STRING] = { "apple", "cat", "boy" };

	ssort1main(arr, NUMBER_OF_STRING);

	for (int i = 0; i < NUMBER_OF_STRING; i++) {
		printf("%s ", arr[i]);
	}
	return 0;
}
