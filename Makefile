# Asmnet Math Toolkit Makefile
# Builds the assembly math library and runs the C test harness.

CC      = gcc
NASM    = nasm
LD      = ld
CFLAGS  = -Wall -Wextra -O2 -no-pie
ASMFLAGS= -f elf64 -g

SRC_DIR   = math
TEST_DIR  = test
BUILD_DIR = build

ASM_SRCS  = $(wildcard $(SRC_DIR)/*.asm)
ASM_OBJS  = $(patsubst $(SRC_DIR)/%.asm,$(BUILD_DIR)/%.o,$(ASM_SRCS))

TEST_SRC  = $(TEST_DIR)/test_math.c
TEST_OBJ  = $(BUILD_DIR)/test_math.o
TEST_BIN  = $(BUILD_DIR)/test_math
LIB       = $(BUILD_DIR)/libasmnet.so

.PHONY: all clean test run lib

all: $(TEST_BIN)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.asm | $(BUILD_DIR)
	$(NASM) $(ASMFLAGS) -o $@ $<

$(TEST_OBJ): $(TEST_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(TEST_BIN): $(ASM_OBJS) $(TEST_OBJ)
	$(CC) $(CFLAGS) -no-pie -o $@ $(ASM_OBJS) $(TEST_OBJ) -lm

test: $(TEST_BIN)
	@./$(TEST_BIN)

run: test

lib: $(LIB)

$(LIB): $(ASM_OBJS) | $(BUILD_DIR)
	$(CC) -shared -nostartfiles -z noexecstack -o $@ $(ASM_OBJS)

clean:
	rm -rf $(BUILD_DIR)
