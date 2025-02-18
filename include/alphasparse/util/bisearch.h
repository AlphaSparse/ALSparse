#pragma once

#include "../types.h"

ALPHA_INT alpha_binary_search(const ALPHA_INT* start,const ALPHA_INT* end,const ALPHA_INT target);
const ALPHA_INT* alpha_lower_bound(const ALPHA_INT* start,const ALPHA_INT* end,const ALPHA_INT target);
const ALPHA_INT* alpha_upper_bound(const ALPHA_INT* start,const ALPHA_INT* end,const ALPHA_INT target);
