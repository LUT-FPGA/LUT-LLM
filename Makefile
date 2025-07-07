# Makefile to compile generate_lut.cpp

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -O2

# Target executable
TARGET = generate_lut

# Source files
SRCS = generate_lut.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Rule to build the target
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean