CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -fPIC -O3
LDFLAGS = -L. -lrex -Wl,-rpath=.

REX_SRC = Rex.cc
REX_HDR = Rex.h
REX_OBJ = Rex.o
REX_TARGET = librex.so

TEA_SRC = teaRex.cc
TEA_HDR = teaRex.h
TEA_OBJ = teaRex.o
TEA_TARGET = libtearex.so

all: $(REX_TARGET) $(TEA_TARGET) 


# Build shared library
$(REX_TARGET): $(REX_SRC)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

$(TEA_TARGET): $(TEA_SRC) $(REX_TARGET)
	$(CXX) $(CXXFLAGS) -shared -o $@ $(TEA_SRC) $(LDFLAGS)

clean:
	rm -f $(REX_TARGET) $(TEA_TARGET) $(REX_OBJ) $(TEA_OBJ)