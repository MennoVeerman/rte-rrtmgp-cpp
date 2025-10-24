#
# This file is part of a C++ interface to the Radiative Transfer for Energetics (RTE)
#
FILE(GLOB sourcefiles
        "../src_tilt/tilt_utils.*"
)

include_directories("../include" "../include_rt" "../include_tilt" "../include_test" SYSTEM ${INCLUDE_DIRS})

add_library(rte_rrtmgp_tilt STATIC ${sourcefiles})
