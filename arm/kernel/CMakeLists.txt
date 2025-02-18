set(alphasparse_source
  kernel/level2/alphasparse_mv.cpp
  PARENT_SCOPE
)

set(ASM_SOURCES
    kernel/level2/mv/gemv/csrmv/gemv_template.S
    PARENT_SCOPE
)