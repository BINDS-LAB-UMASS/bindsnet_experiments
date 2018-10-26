PROGRAM = main

SRCS = objects.cpp main.cpp network.cpp synapses_classes.cpp run.cpp brianlib/randomkit/randomkit.c
H_SRCS = objects.h network.h synapses_classes.h run.h brianlib/dynamic_array.h brianlib/clocks.h brianlib/common_math.h
OBJS = ${SRCS:.cpp=.o}
OBJS := ${OBJS:.c=.o}
CC = @g++
OPTIMISATIONS = -Ibrianlib/randomkit -I/usr/include -w -O3 -ffast-math -fno-finite-math-only -march=native
CFLAGS = -c -Wno-write-strings $(OPTIMISATIONS) -I.  
LFLAGS =  -Lbrianlib/randomkit -L/usr/lib -Wl,--enable-new-dtags,-R/usr/lib 
DEPS = make.deps

all: $(PROGRAM)

.PHONY: all clean

$(PROGRAM): $(OBJS) $(DEPS) makefile
	$(CC) $(OBJS) -o $(PROGRAM) $(LFLAGS)

clean:
	rm $(OBJS) $(PROGRAM) $(DEPS)

make.deps: $(SRCS) $(H_SRCS)
	$(CC) $(CFLAGS) -MM $(SRCS) > make.deps
	
ifneq ($(wildcard $(DEPS)), )
include $(DEPS)
endif

%.o : %.cpp makefile
	$(CC) $(CFLAGS) $< -o $@