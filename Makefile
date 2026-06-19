CXX ?= g++
CC ?= cc
AR ?= ar
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -pedantic
CPPFLAGS ?= -Iinclude
LDLIBS ?=
BACKEND ?= cpu
SANITIZE ?= none
STRICT ?= 0
TARGETS := bin/microgpt bin/mgpt bin/lab bin/chat bin/embed_api_example bin/embed_c_example bin/libmicrogpt.a
HDRS := $(shell find include -name '*.hpp')
PUBLIC_HDRS := $(shell find include -name '*.h')
UNAME_S := $(shell uname -s)
RUNTIME_SRC := src/metal_runtime_stub.cpp
RUNTIME_LANG :=
RUNTIME_OBJ := bin/backend_runtime_$(BACKEND)_sanitize-$(SANITIZE)_strict-$(STRICT).o
API_C_OBJ := bin/api_c_$(BACKEND)_sanitize-$(SANITIZE)_strict-$(STRICT).o
EMBED_C_OBJ := bin/embed_c_$(BACKEND)_sanitize-$(SANITIZE)_strict-$(STRICT).o

ifeq ($(STRICT),1)
CXXFLAGS += -Werror
CFLAGS += -Werror
endif

ifeq ($(SANITIZE),address)
CXXFLAGS += -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
CFLAGS += -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
LDLIBS += -fsanitize=address,undefined
endif

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := -framework Metal -framework Foundation -framework QuartzCore
endif

ifeq ($(BACKEND),metal)
ifneq ($(UNAME_S),Darwin)
$(error BACKEND=metal requires macOS/Darwin; use BACKEND=cpu on this platform)
endif
CPPFLAGS += -DMICROGPT_ENABLE_METAL=1
LDLIBS += $(METAL_LDLIBS)
RUNTIME_SRC := src/metal_runtime.mm
RUNTIME_LANG := -x objective-c++
endif

.PHONY: all clean test cli-test sanitize-build sanitize-test strict-test deps accel-deps print-metal-flags

all: $(TARGETS)

bin:
	mkdir -p bin

$(RUNTIME_OBJ): bin $(RUNTIME_SRC)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(RUNTIME_LANG) -c $(RUNTIME_SRC) -o $(RUNTIME_OBJ)

$(API_C_OBJ): bin src/api_c.cpp $(HDRS) $(PUBLIC_HDRS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c src/api_c.cpp -o $(API_C_OBJ)

bin/libmicrogpt.a: bin $(RUNTIME_OBJ) $(API_C_OBJ)
	$(AR) rcs bin/libmicrogpt.a $(RUNTIME_OBJ) $(API_C_OBJ)

bin/microgpt: bin src/main.cpp $(HDRS) $(RUNTIME_OBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/main.cpp $(RUNTIME_OBJ) -o bin/microgpt $(LDLIBS)

bin/mgpt: bin src/mgpt.cpp $(HDRS) $(RUNTIME_OBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/mgpt.cpp $(RUNTIME_OBJ) -o bin/mgpt $(LDLIBS)

bin/lab: bin src/lab.cpp $(HDRS) $(RUNTIME_OBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/lab.cpp $(RUNTIME_OBJ) -o bin/lab $(LDLIBS)

bin/chat: bin src/chat.cpp $(HDRS) $(RUNTIME_OBJ)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/chat.cpp $(RUNTIME_OBJ) -o bin/chat $(LDLIBS)

bin/embed_api_example: bin examples/embed_api.cpp $(HDRS) bin/libmicrogpt.a
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) examples/embed_api.cpp bin/libmicrogpt.a -o bin/embed_api_example $(LDLIBS)

$(EMBED_C_OBJ): bin examples/embed_c.c $(PUBLIC_HDRS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c examples/embed_c.c -o $(EMBED_C_OBJ)

bin/embed_c_example: bin $(EMBED_C_OBJ) bin/libmicrogpt.a
	$(CXX) $(CXXFLAGS) $(EMBED_C_OBJ) bin/libmicrogpt.a -o bin/embed_c_example $(LDLIBS)

test: bin/lab
	./bin/lab

cli-test: bin/mgpt
	./scripts/cli_smoke_test.sh

sanitize-build:
	$(MAKE) clean
	$(MAKE) all SANITIZE=address

sanitize-test:
	$(MAKE) clean
	$(MAKE) all test SANITIZE=address

strict-test:
	$(MAKE) clean
	$(MAKE) all test STRICT=1

deps: accel-deps

accel-deps:
	./scripts/check_acceleration_deps.sh

print-metal-flags:
	@echo "$(METAL_LDLIBS)"

clean:
	rm -f $(TARGETS) bin/backend_runtime_*.o
