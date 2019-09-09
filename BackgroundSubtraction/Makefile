# the compiler
CC = g++
CFLAGS  = `pkg-config --cflags --libs opencv`
CFILES = functions.cpp backgroundSubTest.cpp
OBJFILES = functions.o backgroundSubTest.o

TARGET = backgroundSubTest

all: $(TARGET)

$(TARGET): $(OBJFILES)
	$(CC) $(CFILES) -o $(TARGET) $(CFLAGS)

clean:
	rm -f $(OBJFILES) $(TARGET) *~