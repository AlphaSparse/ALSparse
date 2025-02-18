#include "alphasparse/types.h"
#include "alphasparse/util/bisearch.h"

ALPHA_INT alpha_binary_search(const ALPHA_INT* start,const ALPHA_INT* end,const ALPHA_INT target){
    ALPHA_INT left = 0;
    ALPHA_INT right = end - start - 1;
    while(left <= right){
        ALPHA_INT mid = (left + right) >> 1;
        ALPHA_INT val = *(start + mid);
        if(val == target){
            return mid;
        }else if(val < target){
            left = mid + 1;
        }else if(val > target){
            right = mid - 1;
        } 
    }
    return -1;
}


const ALPHA_INT* alpha_lower_bound(const ALPHA_INT* start,const ALPHA_INT* end,const ALPHA_INT target){
    ALPHA_INT left = 0;
    ALPHA_INT right = end - start;
    while (left < right) { 
        ALPHA_INT mid = (left + right) >> 1;
        ALPHA_INT val = *(start + mid);
        if (val >= target) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    return start + left;
}


const ALPHA_INT* alpha_upper_bound(const ALPHA_INT* start,const ALPHA_INT* end,const ALPHA_INT target){
    ALPHA_INT left = 0;
    ALPHA_INT right = end - start;
    while (left < right) { 
        ALPHA_INT mid = (left + right) >> 1;
        ALPHA_INT val = *(start + mid);
        if (val <= target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return start + left;
}
