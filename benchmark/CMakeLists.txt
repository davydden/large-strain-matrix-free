##
#  Extra benchmark executables
##

function(ADD_BENCHMARK _name)

  ADD_EXECUTABLE(${_name}
      ${_name}.cc
  )
  DEAL_II_SETUP_TARGET(${_name})
  TARGET_LINK_LIBRARIES(${_name}
    ${TARGETLIB}
  )
endfunction()

#
# add benchmarks:
#
ADD_BENCHMARK("tensors")
