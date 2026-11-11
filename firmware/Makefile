# Bare-metal RV32IM firmware for Spike.
#   make            build build/mnist.elf (needs riscv-none-elf-gcc or set CROSS=...)
#   make run        run it on Spike
# Generate build/model_data.h first: python firmware/gen_model_header.py (from the repo root)
CROSS  ?= riscv-none-elf-
CC      = $(CROSS)gcc
SPIKE  ?= spike
ARCH    = -march=rv32im_zicsr -mabi=ilp32
CFLAGS  = $(ARCH) -O2 -ffreestanding -fno-builtin -nostdlib -nostartfiles -Wall -Wextra -Ibuild -I.
LDFLAGS = -T link.ld -Wl,--gc-sections -Wl,--no-warn-rwx-segments -lgcc

build/mnist.elf: crt0.S htif.c main.c htif.h build/model_data.h link.ld
	$(CC) $(CFLAGS) crt0.S htif.c main.c $(LDFLAGS) -o $@
	$(CROSS)size $@

run: build/mnist.elf
	$(SPIKE) --isa=rv32im -m0x80000000:0x200000 build/mnist.elf

clean:
	rm -f build/mnist.elf

.PHONY: run clean
