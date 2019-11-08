CC = g++
CFLAGS = -g -Wall
SRCS = siftTest.cpp
OBJS = siftTest.o
PROG = siftTest

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean:
	rm -f $(OBJS) $(PROG) *~