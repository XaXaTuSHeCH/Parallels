CXX = g++
CXXFLAGS = -O2 -Wall -std=c++20

TYPE ?= double

all: DOUBLE

DOUBLE:
	$(CXX) $(CXXFLAGS) -DTYPE=double task1.cpp -o task1
	./task1

FLOAT:
	$(CXX) $(CXXFLAGS) -DTYPE=float task1.cpp -o task1
	./task1

clean:
	rm -f task1