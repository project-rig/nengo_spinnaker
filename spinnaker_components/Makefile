# SpiNNaker Nengo Integration
# ----------------------------------------------------------------------------
# Authors:
#  Andrew Mundy <mundya@cs.man.ac.uk>
#  Terry Stewart
#
# February - April 2014
# ----------------------------------------------------------------------------
# Code derived from Andrew Rowley, University of Manchester

APPS = ensemble sdp_tx sdp_rx filter mc_player value_sink value_source
PROFILEABLE_APPS = ensemble

APP_OUTPUT_DIR = $(PWD)/../nengo_spinnaker/binaries
COMPLETE_SOURCE = ./*/*.c ./*/*.h

all :
	for a in $(APPS); do ( cd $$a; "$(MAKE)"  ) || exit $$?; done
	for a in $(PROFILEABLE_APPS); do ( cd $$a; "$(MAKE)" PROFILER_ENABLED=1 ) || exit $$?; done
docs : ${COMPLETE_SOURCE}
	doxygen

tidy :
	for a in $(APPS); do ( cd $$a; "$(MAKE)" tidy  ) || exit $$?; done
	for a in $(PROFILEABLE_APPS); do ( cd $$a; "$(MAKE)" tidy PROFILER_ENABLED=1 ) || exit $$?; done

clean :
	for a in $(APPS); do ( cd $$a; "$(MAKE)" clean  ) || exit $$?; done
	for a in $(PROFILEABLE_APPS); do ( cd $$a; "$(MAKE)" clean PROFILER_ENABLED=1 ) || exit $$?; done
	rm -rf ./docs/
