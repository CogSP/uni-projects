#include <stdio.h> 
#include <string.h> // for memset()
#include <stdlib.h> // for malloc()
#include "mmu.h"

#include <time.h> // for rand() in testing

// here I test the MMU
int main() {

    MMU* mmu = (MMU*) malloc(sizeof(MMU));

    // physical side    
    mmu -> ram = (RAM*) malloc(sizeof(RAM));

    // logical side, takes the first 8 frames of RAM
    mmu -> pagetable = (PageTable*) mmu -> ram;

    mmu -> freelist = (FreeList*) (((char*)(mmu -> ram)) + 8*FRAME_SIZE);

    // first we set all the RAM to 0 to clear
    for (int i = 0; i < N_FRAMES; i++) {
        memset(mmu -> ram -> frames[i].frameblock, 0, FRAME_SIZE);
    }
    

    printf("indirizzo ram %p\n", mmu -> ram);
    printf("indirizzo dopo %p\n",  mmu -> freelist);


    mmu -> freelist -> in_pos = 9;
    
    for (int i = 0; i < N_FRAMES; i++) {
    
        if (i < 9) {
            mmu -> freelist -> arraylist[i] = -1;
        }

        else if (i >= 9) {
            mmu -> freelist -> arraylist[i] = i+1;
        }

        if (i == N_FRAMES - 1) {
            mmu -> freelist -> arraylist[i] = -1;
        }
    }

 
    

    // then we populate the pagetable (that is in the first 16MB of the RAM)
    for (int i = 0; i < N_PAGES; i++) {


        // each pagetable entry is 8 byte (2 int). In a page we have 4KB so we can store 512 entry in each page.
        // we have 4096 page so 4096 entry
        // thus we need 4096 / 512 = 8 unswappable page for storing the page table 
        if (i < 8) {
            mmu -> pagetable -> pages[i].flags = 0;
            mmu -> pagetable -> pages[i].flags |= Unswappable;
            mmu -> pagetable -> pages[i].f = i;
        }

        // for the free list
        // just one page because the free list has 256 entry * 4 byte each entry = 1024 byte + 4 byte for 
        // the initial position  
        if (i == 8) {
            mmu -> pagetable -> pages[i].flags = 0;
            mmu -> pagetable -> pages[i].flags |= Unswappable;
            mmu -> pagetable -> pages[i].f = i;
        }

        //pages mapped in RAM start with Valid flag
        else if (i >= 9 && i < N_FRAMES) {
            mmu -> pagetable -> pages[i].flags = 0;
            mmu -> pagetable -> pages[i].f = -1;
        }


    }

    // TEST:

    char* ret;

    printf("sequential access:");
    for (int i = 0; i < PAG_SIZE*N_PAGES; i++) {

        int address = i;

        printf("let's write 'A' at address %d\n, ", address);
        
        MMU_writeByte(mmu, address, 'A');
        
        ret = MMU_readByte(mmu, address);
        if (ret == 0) {
            printf("readByte failed\n");
        }
        else {
            printf("then read it: %c\n", *ret);
        }
    }


    printf("scattered access:\n");

    int lower = 0;
    int upper = 16777215;

    srand(time(0));

    for (int i = 0; i < 10000; i++) {

        int address = (rand() % (upper - lower)) + lower;

        printf("let's write 'A' at address %d\n", address);
        
        MMU_writeByte(mmu, address, 'A');
    
        ret = MMU_readByte(mmu, address);
        if (ret == 0) {
            printf("readByte failed\n");
        }
        else {
            printf("then read it: %c\n", *ret);
        }
    }

    free(mmu -> ram);
    free(mmu);
       
   
}