
struct CR_Register {
    unsigned intint;
	unsigned int const*const* ptr;
	unsigned int prev : 8;
	unsigned int CLKDIV : 8;
	unsigned int after : 16;
};

int* main()
{
    struct CR_Register reg;
    long int const * ** const x;
    long int* * * const y, const*const* z;
    int i, j, rows;
    printf("Enter number of rows: ");
    scanf("%d",&rows);
    for(i=1; i<=rows; ++i)
    {
        for(j=1; j<=i; ++j)
        {
            printf("* ");
        }
        printf("\n");
    }
    return 0;
}