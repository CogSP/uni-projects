# Memory Management Unit Simulator
Made for the exam "Sistemi Operativi" (Sapienza, Computer Engineering BSc, 2023) this is a simple MMU simulator in user space. 

![MMU Diagram](images/MMU.png)

I allocate a large virtual memory space (16 MB), and use a "swap file" having that size. I use a 1 MB buffer as physical memory.
 
For simplicity, the page table sits at the beginning of the memory and it supports the following flags:
 - valid
 - unswappable
 - read_bit (set by the mmu each time that page is read)
 - write_bit (set by the mmu each time a page is written)

## Functions I implemented:
- void MMU_writeByte(MMU* mmu, int pos, char c)
- char* MMU_readByte(MMU* mmu, int pos)

wrap the memory access, doing the right side effect on the mmu/page table. If an attempt to access at an invalid page, the function 
- void MMU_exception(MMU* mmu, int pos)

is called and it has to handle the page fault doing the appropriate operation on disk. I implement the **second chance algorithm** as page replacement algorithm.  I stressed the memory with different access patterns, and I assume all the virtual memory is allocated


# How this MMU works
The CPU want to write, let's say x = 2, but it only know the logical address of this variable. 
1. CPU provide the MMU with the logical address p|f
2. MMU converts the logical address in a physical one
	- it uses the "p" part of the address to get the page table index in which to find the index f of the RAM frame
	- it substitutes p with f, so the physical address is f|d 
3. The MMU accesses the x variable in RAM by getting the frame in position f at offset d and it writes 2

The process of reading is analogous

## What happens when the RAM is full
when there are no more free frames and a request for an invalid page is done, the page replacement algorithm is activated, so a frame is swapped in disk memory (a single 16MB file) to let the new frame be in RAM. Implemented with a clock algorithm, this algorithm chooses the victim frame by taking the one that has been "used the least".
