ALL:

CFLAGS          = -g -O2
FFLAGS          =
CPPFLAGS        = -g -O2 -std=c++17 -stdlib=libc++
CLEANFILES      =
BENCHCOMMS      = benchcomms
BENCHCOMMSOBJ   = benchcomms.o alias.o utils.o io.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

benchcomms: $(BENCHCOMMS)

$(BENCHCOMMS) : $(BENCHCOMMSOBJ)
			${CXXLINKER} -o $(BENCHCOMMS) $(BENCHCOMMSOBJ) ${PETSC_TAO_LIB}

.PHONY: clean

clean::
	rm -f *.o
	rm -f benchcomms
