CC = g++
CFLAGS = -g -Wall
SRCS = KeyFramesAnalysis.cpp main.cpp
OBJS = KeyFramesAnalysis.o main.o
PROG = main

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean:
	rm -f $(OBJS) $(PROG) *~