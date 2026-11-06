# Makefile for OjasX Native Split-Manifold LUT Execution Engine
# Supports macOS (Apple Silicon ARM NEON) and Linux (x86_64 AVX2)

CXX ?= clang++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -Wno-unused-parameter -Iinclude -fPIC

# OS Detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT := dylib
    LDFLAGS_SHARED := -dynamiclib -install_name @rpath/libojasx_lut.dylib
else
    LIB_EXT := so
    LDFLAGS_SHARED := -shared -Wl,-soname,libojasx_lut.so
endif

SRC_DIR := src
INC_DIR := include
TEST_DIR := tests
BIN_DIR := bin

CORE_SRCS := $(SRC_DIR)/ojasx_lut_cpu.cpp \
             $(SRC_DIR)/ojasx_lut_kv_cache.cpp \
             $(SRC_DIR)/ojasx_lut_c_api.cpp \
             $(SRC_DIR)/ojasx_ggml_adapter.cpp

CORE_OBJS := $(CORE_SRCS:.cpp=.o)

LIB_TARGET := libojasx_lut.$(LIB_EXT)
TEST_TARGET := test_native_lut
RUNNER_TARGET := ojasx_run

.PHONY: all clean test benchmark info

all: $(LIB_TARGET) $(TEST_TARGET) $(RUNNER_TARGET)

$(LIB_TARGET): $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS_SHARED) $^ -o $@
	@echo "Built shared library: $@"

$(TEST_TARGET): $(CORE_OBJS) $(TEST_DIR)/test_native_lut.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Built test binary: $@"

$(RUNNER_TARGET): $(CORE_OBJS) $(SRC_DIR)/ojasx_run.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Built runner binary: $@"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(TEST_TARGET)
	@echo "Running Native C++ Test Suite..."
	./$(TEST_TARGET)

benchmark: $(RUNNER_TARGET)
	@echo "Running Native C++ Benchmark Runner..."
	./$(RUNNER_TARGET) --benchmark

info: $(RUNNER_TARGET)
	./$(RUNNER_TARGET) --info

clean:
	rm -f $(SRC_DIR)/*.o $(LIB_TARGET) $(TEST_TARGET) $(RUNNER_TARGET)
	@echo "Cleaned all native build artifacts."
