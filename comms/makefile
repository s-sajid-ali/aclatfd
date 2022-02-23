ALL:

CFLAGS          = -g -O2
FFLAGS          =
CPPFLAGS        = -g -O2 -std=c++17 -stdlib=libc++
CLEANFILES      =
BASELINE        = baseline
BASELINEOBJ     = baseline.o
BENCHCOMMS      = benchcomms
BENCHCOMMSOBJ   = benchcomms.o alias.o utils.o io.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

benchcomms: $(BENCHCOMMS)
baseline: $(BASELINE)

$(BENCHCOMMS) : $(BENCHCOMMSOBJ)
			${CXXLINKER} -o $(BENCHCOMMS) $(BENCHCOMMSOBJ) ${PETSC_TAO_LIB}

$(BASELINE) : $(BASELINEOBJ)
			${CXXLINKER} -o $(BASELINE) $(BASELINEOBJ) ${PETSC_TAO_LIB}

.PHONY: clean

clean::
	rm -f *.o
	rm -f benchcomms
	rm -f baseline
