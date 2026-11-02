# Tools and flags
CXX      ?= g++
CPPFLAGS += -Iinclude
CXXFLAGS ?= -O3 -std=c++17 -fPIC -pthread
LDFLAGS  ?= -shared

# Install dirs
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCDIR ?= $(PREFIX)/include

# Sources / objects
SRCS := src/Rex.cc src/teaRex.cc
OBJS := $(SRCS:.cc=.o)
DEPS := $(OBJS:.o=.d)

# Default target
all: librex.so libtearex.so

# Shared libs
librex.so: src/Rex.o
	$(CXX) $(LDFLAGS) $^ -o $@

libtearex.so: src/teaRex.o
	$(CXX) $(LDFLAGS) $^ -o $@

# Compile with automatic header dependency generation
# -MMD: write .d files, -MP: add phony targets to avoid issues on rm
src/%.o: src/%.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

# Install headers + libs
install: all
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(INCDIR)
	install -m 0644 librex.so libtearex.so $(DESTDIR)$(LIBDIR)/
	install -m 0644 include/Rex.h include/teaRex.h $(DESTDIR)$(INCDIR)/

clean:
	rm -f $(OBJS) $(DEPS) librex.so libtearex.so

.PHONY: all install clean

# Include generated dependency files so header changes trigger rebuilds
-include $(DEPS)
