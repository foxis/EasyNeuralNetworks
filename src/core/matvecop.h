#if !defined(ENN_MATVECOP_H)
#define ENN_MATVECOP_H

#include <stdlib.h>
#include <math.h>
#include <limits>

#define ENN_BIAS (BIAS?1:0)

#include "arch/pure/mvo_array.h"
#include "arch/pure/mvo_vector.h"
#include "arch/pure/mvo_matrix.h"
#include "arch/pure/mvo_conv.h"
#include "arch/pure/mvo_rand.h"

#endif
