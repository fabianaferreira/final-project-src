CC = g++
CFLAGS = -g -Wall
SRCS = functions.cpp dictionaryCreator.cpp
OBJS = functions.o dictionaryCreator.o
PROG = training

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean:
	rm -f $(OBJS) $(PROG) *~