ROOTDIR         := /home/djsaunde/code/experiments/experiments/benchmark/GeNNworkspace
EXECUTABLE      := main
SOURCES         := main.cpp  objects.cpp  brianlib/randomkit/randomkit.cc

INCLUDE_FLAGS   := -I. -Ibrianlib/randomkit -Lbrianlib/randomkit
LINK_FLAGS := 
OPTIMIZATIONFLAGS := -w -O3 -ffast-math -fno-finite-math-only -march=native
NVCC_OPTIMIZATIONFLAGS := -O3
SIM_CODE := magicnetwork_model_CODE

include $(GENN_PATH)/userproject/include/makefile_common_gnu.mk