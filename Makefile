CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
CPPFLAGS ?= -Iinclude
TARGET := bin/microgpt
SRC := src/main.cpp

.PHONY: all clean test

all: $(TARGET)

bin:
	mkdir -p bin

$(TARGET): bin $(SRC) include/microgpt.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) -o $(TARGET)

test: $(TARGET)
	./$(TARGET) test

clean:
	rm -f $(TARGET)

