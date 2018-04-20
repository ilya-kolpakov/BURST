#ifndef MM_SHUFFLE_EPI8_H
#define MM_SHUFFLE_EPI8_H

#include <immintrin.h>

#define SSP_FORCEINLINE __attribute__((always_inline))

/*
SSP_FORCEINLINE __m128i ssp_comge_epi8_SSE2(__m128i a, __m128i b)
{
        __m128i c;
        c = _mm_cmpgt_epi8( a, b );
        a = _mm_cmpeq_epi8( a, b );
        a = _mm_or_si128  ( a, c );
        return a;
}
 
SSP_FORCEINLINE __m128i ssp_shuffle_epi8_SSE2(__m128i a, __m128i mask)
{
        ssp_m128 A, B, MASK, maskZero;
        A.i        = a;
        maskZero.i = ssp_comge_epi8_SSE2( mask, _mm_setzero_si128()        );
        MASK.i     = _mm_and_si128      ( mask, _mm_set1_epi8( (char)0x0F) );
 
        B.s8[ 0] = A.s8[ (MASK.s8[ 0]) ];
        B.s8[ 1] = A.s8[ (MASK.s8[ 1]) ];
        B.s8[ 2] = A.s8[ (MASK.s8[ 2]) ];
        B.s8[ 3] = A.s8[ (MASK.s8[ 3]) ];
        B.s8[ 4] = A.s8[ (MASK.s8[ 4]) ];
        B.s8[ 5] = A.s8[ (MASK.s8[ 5]) ];
        B.s8[ 6] = A.s8[ (MASK.s8[ 6]) ];
        B.s8[ 7] = A.s8[ (MASK.s8[ 7]) ];
        B.s8[ 8] = A.s8[ (MASK.s8[ 8]) ];
        B.s8[ 9] = A.s8[ (MASK.s8[ 9]) ];
        B.s8[10] = A.s8[ (MASK.s8[10]) ];
        B.s8[11] = A.s8[ (MASK.s8[11]) ];
        B.s8[12] = A.s8[ (MASK.s8[12]) ];
        B.s8[13] = A.s8[ (MASK.s8[13]) ];
        B.s8[14] = A.s8[ (MASK.s8[14]) ];
        B.s8[15] = A.s8[ (MASK.s8[15]) ];
 
        B.i = _mm_and_si128( B.i, maskZero.i );
        return B.i;
}
*/

inline SSP_FORCEINLINE __m128i ssp_shuffle_epi8_SSE2_new(__m128i a, __m128i mask)
{
        __m128i m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15;
        __m128i t0, t1, t2, t3;
 
        m0  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 0));
        m1  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 1));
        m2  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 2));
        m3  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 3));
        m4  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 4));
        m5  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 5));
        m6  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 6));
        m7  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 7));
        m8  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 8));
        m9  = _mm_cmpeq_epi8(mask, _mm_set1_epi8( 9));
        m10 = _mm_cmpeq_epi8(mask, _mm_set1_epi8(10));
        m11 = _mm_cmpeq_epi8(mask, _mm_set1_epi8(11));
        m12 = _mm_cmpeq_epi8(mask, _mm_set1_epi8(12));
        m13 = _mm_cmpeq_epi8(mask, _mm_set1_epi8(13));
        m14 = _mm_cmpeq_epi8(mask, _mm_set1_epi8(14));
        m15 = _mm_cmpeq_epi8(mask, _mm_set1_epi8(15));
 
        t0  = _mm_and_si128(a, _mm_set1_epi32(0x000000ff));
        t1  = _mm_and_si128(a, _mm_set1_epi32(0x0000ff00));
        t2  = _mm_and_si128(a, _mm_set1_epi32(0x00ff0000));
        t3  = _mm_and_si128(a, _mm_set1_epi32(0xff000000));
        t0  = _mm_or_si128(t0, _mm_slli_epi32(t0,  8));
        t0  = _mm_or_si128(t0, _mm_slli_epi32(t0, 16));
        t1  = _mm_or_si128(t1, _mm_srli_epi32(t1,  8));
        t1  = _mm_or_si128(t1, _mm_slli_epi32(t1, 16));
        t2  = _mm_or_si128(t2, _mm_slli_epi32(t2,  8));
        t2  = _mm_or_si128(t2, _mm_srli_epi32(t2, 16));
        t3  = _mm_or_si128(t3, _mm_srli_epi32(t3,  8));
        t3  = _mm_or_si128(t3, _mm_srli_epi32(t3, 16));
 
        a =                 _mm_and_si128(m0,  _mm_shuffle_epi32(t0, _MM_SHUFFLE(0, 0, 0, 0)));
        a = _mm_or_si128(a, _mm_and_si128(m1,  _mm_shuffle_epi32(t1, _MM_SHUFFLE(0, 0, 0, 0))));
        a = _mm_or_si128(a, _mm_and_si128(m2,  _mm_shuffle_epi32(t2, _MM_SHUFFLE(0, 0, 0, 0))));
        a = _mm_or_si128(a, _mm_and_si128(m3,  _mm_shuffle_epi32(t3, _MM_SHUFFLE(0, 0, 0, 0))));
        a = _mm_or_si128(a, _mm_and_si128(m4,  _mm_shuffle_epi32(t0, _MM_SHUFFLE(1, 1, 1, 1))));
        a = _mm_or_si128(a, _mm_and_si128(m5,  _mm_shuffle_epi32(t1, _MM_SHUFFLE(1, 1, 1, 1))));
        a = _mm_or_si128(a, _mm_and_si128(m6,  _mm_shuffle_epi32(t2, _MM_SHUFFLE(1, 1, 1, 1))));
        a = _mm_or_si128(a, _mm_and_si128(m7,  _mm_shuffle_epi32(t3, _MM_SHUFFLE(1, 1, 1, 1))));
        a = _mm_or_si128(a, _mm_and_si128(m8,  _mm_shuffle_epi32(t0, _MM_SHUFFLE(2, 2, 2, 2))));
        a = _mm_or_si128(a, _mm_and_si128(m9,  _mm_shuffle_epi32(t1, _MM_SHUFFLE(2, 2, 2, 2))));
        a = _mm_or_si128(a, _mm_and_si128(m10, _mm_shuffle_epi32(t2, _MM_SHUFFLE(2, 2, 2, 2))));
        a = _mm_or_si128(a, _mm_and_si128(m11, _mm_shuffle_epi32(t3, _MM_SHUFFLE(2, 2, 2, 2))));
        a = _mm_or_si128(a, _mm_and_si128(m12, _mm_shuffle_epi32(t0, _MM_SHUFFLE(3, 3, 3, 3))));
        a = _mm_or_si128(a, _mm_and_si128(m13, _mm_shuffle_epi32(t1, _MM_SHUFFLE(3, 3, 3, 3))));
        a = _mm_or_si128(a, _mm_and_si128(m14, _mm_shuffle_epi32(t2, _MM_SHUFFLE(3, 3, 3, 3))));
        a = _mm_or_si128(a, _mm_and_si128(m15, _mm_shuffle_epi32(t3, _MM_SHUFFLE(3, 3, 3, 3))));
        return a;
}

#endif
