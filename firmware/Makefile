# Bare-metal RV32IM firmware for Spike.
#   make            build build/mnist.elf, build/isa_test.elf and the Spike extension
#   make run        MNIST inference (scalar and with the custom MAC instructions)
#   make isa-test   check the custom instructions against a C reference
# Generate build/model_data.h first: python firmware/gen_model_header.py (from the repo root)
CROSS     ?= riscv-none-elf-
CC         = $(CROSS)gcc
SPIKE     ?= spike
SPIKE_SRC ?= $(HOME)/.riscv-edge-ai-tools/src/spike
ARCH       = -march=rv32im_zicsr -mabi=ilp32
CFLAGS     = $(ARCH) -O2 -ffreestanding -fno-builtin -nostdlib -nostartfiles -Wall -Wextra -Ibuild -I. -I../hardware/include
LDFLAGS    = -T link.ld -Wl,--gc-sections -Wl,--no-warn-rwx-segments -lgcc
EXT        = build/libsimd_mac.so
SPIKE_RUN  = $(SPIKE) --isa=rv32im -m0x80000000:0x200000 --extlib=$(EXT) --extension=simd_mac

all: build/mnist.elf build/isa_test.elf $(EXT)

build/mnist.elf: crt0.S htif.c main.c htif.h build/model_data.h link.ld ../hardware/include/simd_mac_isa.h
	$(CC) $(CFLAGS) crt0.S htif.c main.c $(LDFLAGS) -o $@
	$(CROSS)size $@

build/isa_test.elf: crt0.S htif.c isa_test.c htif.h link.ld ../hardware/include/simd_mac_isa.h
	@mkdir -p build
	$(CC) $(CFLAGS) crt0.S htif.c isa_test.c $(LDFLAGS) -o $@

# The custom-0 instructions for Spike, built against the Spike source tree.
$(EXT): spike_ext/simd_mac_ext.cc
	@mkdir -p build
	g++ -std=c++17 -O2 -Wall -fPIC -shared -include cstdint \
	  -I$(SPIKE_SRC) -I$(SPIKE_SRC)/riscv -I$(SPIKE_SRC)/fesvr -I$(SPIKE_SRC)/softfloat -I$(SPIKE_SRC)/build \
	  $< -o $@

run: build/mnist.elf $(EXT)
	$(SPIKE_RUN) build/mnist.elf

isa-test: build/isa_test.elf $(EXT)
	$(SPIKE_RUN) build/isa_test.elf

clean:
	rm -f build/*.elf $(EXT)

.PHONY: all run isa-test clean
