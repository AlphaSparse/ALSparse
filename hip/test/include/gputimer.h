#pragma once

#define GPU_TIMER_START(elapsed_time, event_start, event_stop) \
  do                                                           \
  {                                                            \
    elapsed_time = 0.0;                                        \
    hipEventCreate(&event_start);                             \
    hipEventCreate(&event_stop);                              \
    hipEventRecord(event_start);                              \
  } while (0)

#define GPU_TIMER_END(elapsed_time, event_start, event_stop)      \
  do                                                              \
  {                                                               \
    hipEventRecord(event_stop);                                  \
    hipEventSynchronize(event_stop);                             \
    hipEventElapsedTime(&elapsed_time, event_start, event_stop); \
    elapsed_time *= 1000.0;                                       \
  } while (0)