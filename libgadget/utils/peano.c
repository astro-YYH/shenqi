#include "peano.h"

/*  The following rewrite of the original function
 *  peano_hilbert_key_old() has been written by MARTIN REINECKE. 
 *  It is about a factor 2.3 - 2.5 faster than Volker's old routine!
 */
static const unsigned char rottable3[48][8] = {
    {36, 28, 25, 27, 10, 10, 25, 27},
    {29, 11, 24, 24, 37, 11, 26, 26},
    {8, 8, 25, 27, 30, 38, 25, 27},
    {9, 39, 24, 24, 9, 31, 26, 26},
    {40, 24, 44, 32, 40, 6, 44, 6},
    {25, 7, 33, 7, 41, 41, 45, 45},
    {4, 42, 4, 46, 26, 42, 34, 46},
    {43, 43, 47, 47, 5, 27, 5, 35},
    {33, 35, 36, 28, 33, 35, 2, 2},
    {32, 32, 29, 3, 34, 34, 37, 3},
    {33, 35, 0, 0, 33, 35, 30, 38},
    {32, 32, 1, 39, 34, 34, 1, 31},
    {24, 42, 32, 46, 14, 42, 14, 46},
    {43, 43, 47, 47, 25, 15, 33, 15},
    {40, 12, 44, 12, 40, 26, 44, 34},
    {13, 27, 13, 35, 41, 41, 45, 45},
    {28, 41, 28, 22, 38, 43, 38, 22},
    {42, 40, 23, 23, 29, 39, 29, 39},
    {41, 36, 20, 36, 43, 30, 20, 30},
    {37, 31, 37, 31, 42, 40, 21, 21},
    {28, 18, 28, 45, 38, 18, 38, 47},
    {19, 19, 46, 44, 29, 39, 29, 39},
    {16, 36, 45, 36, 16, 30, 47, 30},
    {37, 31, 37, 31, 17, 17, 46, 44},
    {12, 4, 1, 3, 34, 34, 1, 3},
    {5, 35, 0, 0, 13, 35, 2, 2},
    {32, 32, 1, 3, 6, 14, 1, 3},
    {33, 15, 0, 0, 33, 7, 2, 2},
    {16, 0, 20, 8, 16, 30, 20, 30},
    {1, 31, 9, 31, 17, 17, 21, 21},
    {28, 18, 28, 22, 2, 18, 10, 22},
    {19, 19, 23, 23, 29, 3, 29, 11},
    {9, 11, 12, 4, 9, 11, 26, 26},
    {8, 8, 5, 27, 10, 10, 13, 27},
    {9, 11, 24, 24, 9, 11, 6, 14},
    {8, 8, 25, 15, 10, 10, 25, 7},
    {0, 18, 8, 22, 38, 18, 38, 22},
    {19, 19, 23, 23, 1, 39, 9, 39},
    {16, 36, 20, 36, 16, 2, 20, 10},
    {37, 3, 37, 11, 17, 17, 21, 21},
    {4, 17, 4, 46, 14, 19, 14, 46},
    {18, 16, 47, 47, 5, 15, 5, 15},
    {17, 12, 44, 12, 19, 6, 44, 6},
    {13, 7, 13, 7, 18, 16, 45, 45},
    {4, 42, 4, 21, 14, 42, 14, 23},
    {43, 43, 22, 20, 5, 15, 5, 15},
    {40, 12, 21, 12, 40, 6, 23, 6},
    {13, 7, 13, 7, 41, 41, 22, 20}
};

static const unsigned char subpix3[48][8] = {
    {0, 7, 1, 6, 3, 4, 2, 5},
    {7, 4, 6, 5, 0, 3, 1, 2},
    {4, 3, 5, 2, 7, 0, 6, 1},
    {3, 0, 2, 1, 4, 7, 5, 6},
    {1, 0, 6, 7, 2, 3, 5, 4},
    {0, 3, 7, 4, 1, 2, 6, 5},
    {3, 2, 4, 5, 0, 1, 7, 6},
    {2, 1, 5, 6, 3, 0, 4, 7},
    {6, 1, 7, 0, 5, 2, 4, 3},
    {1, 2, 0, 3, 6, 5, 7, 4},
    {2, 5, 3, 4, 1, 6, 0, 7},
    {5, 6, 4, 7, 2, 1, 3, 0},
    {7, 6, 0, 1, 4, 5, 3, 2},
    {6, 5, 1, 2, 7, 4, 0, 3},
    {5, 4, 2, 3, 6, 7, 1, 0},
    {4, 7, 3, 0, 5, 6, 2, 1},
    {6, 7, 5, 4, 1, 0, 2, 3},
    {7, 0, 4, 3, 6, 1, 5, 2},
    {0, 1, 3, 2, 7, 6, 4, 5},
    {1, 6, 2, 5, 0, 7, 3, 4},
    {2, 3, 1, 0, 5, 4, 6, 7},
    {3, 4, 0, 7, 2, 5, 1, 6},
    {4, 5, 7, 6, 3, 2, 0, 1},
    {5, 2, 6, 1, 4, 3, 7, 0},
    {7, 0, 6, 1, 4, 3, 5, 2},
    {0, 3, 1, 2, 7, 4, 6, 5},
    {3, 4, 2, 5, 0, 7, 1, 6},
    {4, 7, 5, 6, 3, 0, 2, 1},
    {6, 7, 1, 0, 5, 4, 2, 3},
    {7, 4, 0, 3, 6, 5, 1, 2},
    {4, 5, 3, 2, 7, 6, 0, 1},
    {5, 6, 2, 1, 4, 7, 3, 0},
    {1, 6, 0, 7, 2, 5, 3, 4},
    {6, 5, 7, 4, 1, 2, 0, 3},
    {5, 2, 4, 3, 6, 1, 7, 0},
    {2, 1, 3, 0, 5, 6, 4, 7},
    {0, 1, 7, 6, 3, 2, 4, 5},
    {1, 2, 6, 5, 0, 3, 7, 4},
    {2, 3, 5, 4, 1, 0, 6, 7},
    {3, 0, 4, 7, 2, 1, 5, 6},
    {1, 0, 2, 3, 6, 7, 5, 4},
    {0, 7, 3, 4, 1, 6, 2, 5},
    {7, 6, 4, 5, 0, 1, 3, 2},
    {6, 1, 5, 2, 7, 0, 4, 3},
    {5, 4, 6, 7, 2, 3, 1, 0},
    {4, 3, 7, 0, 5, 2, 6, 1},
    {3, 2, 0, 1, 4, 5, 7, 6},
    {2, 5, 1, 6, 3, 4, 0, 7}
};

/*! This function computes a Peano-Hilbert key for an integer triplet (x,y,z),
 *  with x,y,z in the range between 0 and 2^bits-1.
 */
peano_t peano_hilbert_key(const int x, const int y, const int z, const int bits)
{
    int bit;
    unsigned char rotation = 0;
    peano_t key = 0;

    for(bit = (bits - 1); bit >= 0; bit -= 1)
    {
        const int mask = 1 << bit;
        const unsigned char pix = (4*((x & mask) >> bit)) | (2*((y & mask) >> bit)) | ((z & mask) >> bit);

        key <<= 3;
        key |= subpix3[rotation][pix];
        rotation = rottable3[rotation][pix];
    }

    return key;
}
