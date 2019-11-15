
#include<stdio.h>


#if(defined (_M_THUMB))
#define _ARMINTR_ASSERT_SAT_SHIFT(type, amt) \
    (_ARMINTR_ASSERT(((type) == _ARM_LSL) || ((type) == _ARM_ASR), "shift type must be _ARM_LSL or _ARM_ASR"), \
     _ARMINTR_ASSERT(((type) != _ARM_LSL) || _ARMINTR_IN_RANGE((amt), 0, 31), "shift is out of range '0 - 31'"), \
     _ARMINTR_ASSERT(((type) != _ARM_ASR) || _ARMINTR_IN_RANGE((amt), 1, 31), "shift is out of range '1 - 31'"))

#else
#define _ARMINTR_ASSERT_SAT_SHIFT(type, amt) \
    (_ARMINTR_ASSERT(((type) == _ARM_LSL) || ((type) == _ARM_ASR), "shift type must be _ARM_LSL or _ARM_ASR"), \
     _ARMINTR_ASSERT(((type) != _ARM_LSL) || _ARMINTR_IN_RANGE((amt), 0, 31), "shift is out of range '0 - 31'"), \
     _ARMINTR_ASSERT(((type) != _ARM_ASR) || _ARMINTR_IN_RANGE((amt), 1, 32), "shift is out of range '1 - 32'"))
#endif

#define _ARMINTR_ENCODE_SAT_SH(type, amt) ((((type) & 2U) << 4) | _ARMINTR_ENCODE_IMM5_7(amt))

#define _ARMINTR_BFC        0x0000F36FU

#define _arm_umaal(_RdLo, _RdHi, _Rn, _Rm)  (__arm_gen_u_Rdn_RdnRmRs(_ARMINTR_UMAAL, ((unsigned __int64)(_RdHi) << 32) | (_RdLo), (_Rn), (_Rm)))

#ifdef __cplusplus

#endif


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