#include <stdio.h> 
#include <stdlib.h>
#include <string.h> // for strcpy
#include "mmu.h"


void MMU_writeByte(MMU* mmu, int pos, char c) {

    // TODO: check if these conditions are enough
    if (pos < 0 || pos >= VMS) {
        //printf("out of memory border, exiting...");
        return;
    }

    // pos is valid, let's find the page
    int p = pos / PAG_SIZE; //page_index, / get us the integer rounded down
    int d = pos % PAG_SIZE; //page offset 

    //printf("address %d belongs to page %d at offset %d\n", pos, p, d);
    fflush(stdout);

    // we get the p-th entry of the page table    
    Page entry = mmu -> pagetable -> pages[p];
    //printf("flags of the page:\n \t Valid: %d Unswappable: %d Read: %d Write: %d Free: %d\n",
    //entry.flags & Valid, entry.flags & Unswappable, entry.flags & Read, entry.flags & Write, entry.flags & Free);


    while(1) {

        // to write, the page must be valid
        if (mmu -> pagetable -> pages[p].flags & Valid) {
            
            //printf("page %d is valid and mapped in frame %d\n", p, mmu -> pagetable -> pages[p].f);

            int f = mmu -> pagetable -> pages[p].f; // frame_index
            int d_frame = d; // frame offset


            mmu -> ram -> frames[f].frameblock[d_frame] = c;

            //printf("we wrote %c, success!\n", c);

            //set the byte WRITE
            mmu -> pagetable -> pages[p].flags |= Write;

            return;

        }

        if (mmu -> pagetable -> pages[p].flags & Unswappable) {
            //printf("this page is not valid: it's PageTable's (or freelist) home!\n");
            return;
        } 
        

        //printf("the page is not valid, handling fault...\n");


        // since it's a fault, if the handler went good, we
        // will restart from the istruction that caused 
        // the fault (writeByte)         
        MMU_exception(mmu, pos);


        //printf("retrying...\n");
        fflush(stdout);
    }


    //printf("you should not be here");
    return;
}


char* MMU_readByte(MMU* mmu, int pos) {
    
    if (pos < 0 || pos >= VMS) {
        //printf("out of memory border, exiting...");
        return 0;
    }

    int p = pos / PAG_SIZE; 
    int d = pos % PAG_SIZE; 

    //printf("address %d belongs to page %d at offset %d\n", pos, p, d);
    fflush(stdout);


    Page entry = mmu -> pagetable -> pages[p];

    //printf("flags of the page:\n \t Valid: %d Unswappable: %d Read: %d Write: %d\n",
    //entry.flags & Valid, entry.flags & Unswappable, entry.flags & Read, entry.flags & Write);


    while (1) {

        // to read, the page must be valid
        if (mmu -> pagetable -> pages[p].flags & Valid) {
            
            //printf("page %d is valid and mapped in frame %d\n", p, mmu -> pagetable -> pages[p].f);

            int f = entry.f; // frame_index
            int d_frame = d; //frame offset

            Frame frame = mmu -> ram -> frames[f];
            char var = frame.frameblock[d_frame];

            //printf("we read %c, success!\n", var);

            //TODO: set the byte READ
            mmu -> pagetable -> pages[p].flags |= Read;

            return &mmu -> ram -> frames[f].frameblock[d_frame];

        }

        if (mmu -> pagetable -> pages[p].flags & Unswappable) {
            //printf("this page is not valid: it's PageTable's (or FreeList) home!");
            return NULL;
        } 
        
        //printf("the page is not valid, handling page fault...\n");


        // since it's a fault, if the handler went good, we
        // will restart from the istruction that caused 
        // the fault (readByte)         
        MMU_exception(mmu, pos);


        //printf("retrying...\n");
        fflush(stdout);
    }


}


void MMU_exception(MMU* mmu, int pos) {


    FILE* swap_file = fopen("swap_file.bin", "w+b"); 
    //b for writing in binary mode

    if (swap_file == NULL) {
        //printf("an error occurred in opening the swap file\n");
        return;
    }

    int p = pos / PAG_SIZE;

    // if -1 it means that the frame is not on disk because is the first time we take it
    int pos_frame = mmu -> pagetable -> pages[pos / PAG_SIZE].f;


    // frame_in_disc is modified if the frame is on disk, otherwise is a block of zeros
    char* frame_in_disc = (char*) malloc(sizeof(char)*FRAME_SIZE);
    memset(frame_in_disc, 0, sizeof(char)*FRAME_SIZE);

    if (pos_frame == -1) {
        //printf("the page %d was never mapped to anything, so we just have to find the victim\n", p);
    }

    else {
        //printf("the page %d has the frame in disk at block %d\n", p, pos_frame);
        fseek(swap_file, pos_frame*FRAME_SIZE, SEEK_SET);
        fread(frame_in_disc, sizeof(char), FRAME_SIZE, swap_file);  
    }
    
    //the frame index we will map the page to
    int ret = -1;

    // if there are no free frame, we must swap a victim frame
    // with a frame from the swap space
    // I use second chance algorithm for the page replacement
    // precisely, this is a enhanced algorithm,
    // because we use the validity bit but also
    // a modify bit (2 actually, Read and Write)
    
    
    if (mmu -> freelist -> in_pos != -1) {
        printf("let's search in the free list\n");
        int f = mmu -> freelist -> in_pos;
        mmu -> pagetable -> pages[p].f = f;
        mmu -> pagetable -> pages[p].flags |= Valid | Read | Write;

        int aux = mmu -> freelist -> arraylist[mmu -> freelist -> in_pos];
        mmu -> freelist -> arraylist[mmu -> freelist -> in_pos] = -1;
        mmu -> freelist -> in_pos = aux;

        ret = fclose(swap_file);
        if (ret != 0) {
            //printf("error in closing the swap file\n");
        }

        //printf("now page %d has %d as frame\n", p, f);

        return;
    }
    
    
    printf("let's find a victim, clock algorithm running...\n");

    int k = 0;

    while (1) {
        k++;
        int i = 0;
        for (i; i < N_PAGES; i++) {

         
            // if we found Valid = 0 we ignore the frame
            // otherwise we check Read and Write,
            // if they are 0 then we take the frame
            // if Read is set to 1, we set 
            // it to 0 and we keep seeking, giving 
            // the page a second chance
            // if Read = 0 we swap the frame

            if ((mmu -> pagetable -> pages[i].flags & Valid)) {    
                if (!(mmu -> pagetable -> pages[i].flags & Unswappable)) {

                    //printf("possible victim found: page %d with flags Valid: %d Unswappable: %d Read: %d Write: %d\n", i, 
                    //mmu -> pagetable -> pages[i].flags & Valid, mmu -> pagetable -> pages[i].flags & Unswappable,
                    //mmu -> pagetable -> pages[i].flags & Read, mmu -> pagetable -> pages[i].flags & Write);
                    fflush(stdout);

                    if (!(mmu -> pagetable -> pages[i].flags & Read) && !(mmu -> pagetable -> pages[i].flags & Write)) {
                    // both Read and Write = 0
                    // we take the frame
                    // we do not have to swap out since Write = 0 means che dick version of the frame is updated

                        char* victim = mmu -> ram -> frames[mmu -> pagetable -> pages[i].f].frameblock;

                        printf("write and read (0,0), we take the frame %d as victim!\n", mmu -> pagetable -> pages[i].f);
                        fflush(stdout);
                        exit(0);

                        //mmu -> pagetable -> pages[i].flags &= !Valid;
                        mmu -> pagetable -> pages[i].flags = 0;

                        // actually not, write = 0 so we do not swap
                        //printf("page set to invalid, it's going to be put in disk, precisely at address %d\n", mmu -> pagetable -> pages[i].f*FRAME_SIZE);
                        fflush(stdout);


                        mmu -> pagetable -> pages[p].flags |= Valid | Write | Read;

            
                        if (mmu -> pagetable -> pages[p].f != -1) {
                            strcpy(mmu -> ram -> frames[mmu -> pagetable -> pages[p].f].frameblock, frame_in_disc);
                        }

                        mmu -> pagetable -> pages[p].f = mmu -> pagetable -> pages[i].f;

                        ret = fclose(swap_file);
                        if (ret != 0) {
                            //printf("error in closing the swap file\n");
                        }

                        return;
                    }

                 
                    if ((mmu -> pagetable -> pages[i].flags & Read) && (mmu -> pagetable -> pages[i].flags & Write)) {
                        // this frame is very used
                        // we give it a second chance by setting read to 0
                        mmu -> pagetable -> pages[i].flags &= !Read;
                        mmu -> pagetable -> pages[i].flags |= Valid;
                        mmu -> pagetable -> pages[i].flags |= Write;
                         
                        //printf("now read is %d, more over the page has Valid %d, Unswappable %d, Write %d", mmu -> pagetable -> pages[i].flags & Read, mmu -> pagetable -> pages[i].flags & Valid, mmu -> pagetable -> pages[i].flags & Unswappable, mmu -> pagetable -> pages[i].flags & Write);
                        fflush(stdout);
                    }

                    // just one of the two is 1
                    else {
                        
                        // read = 1 and write = 0
                        // giving a second chance to the frame
                        if ((mmu -> pagetable -> pages[i].flags & Read) && !(mmu -> pagetable -> pages[i].flags & Write)) {
                            mmu -> pagetable -> pages[i].flags &= !Read;
                            mmu -> pagetable -> pages[i].flags |= Valid;
                            mmu -> pagetable -> pages[i].flags |= Write;
                            //printf("now read is %d, more over the page has Valid %d, Unswappable %d, Write %d", mmu -> pagetable -> pages[i].flags & Read, mmu -> pagetable -> pages[i].flags & Valid, mmu -> pagetable -> pages[i].flags & Unswappable, mmu -> pagetable -> pages[i].flags & Write);
                            fflush(stdout);
                        }

                        // read = 0 and write = 1, in this case we have to swap
                        else if ((mmu -> pagetable -> pages[i].flags & Write) && !(mmu -> pagetable -> pages[i].flags & Read)) {
                            

                            char* victim = mmu -> ram -> frames[mmu -> pagetable -> pages[i].f].frameblock;

                            printf("write and read (0,1), we take the frame %d as victim!\n", mmu -> pagetable -> pages[i].f);
                            fflush(stdout);

                            //mmu -> pagetable -> pages[i].flags |= !Valid;
                            mmu -> pagetable -> pages[i].flags = 0;

                            //printf("page set to invalid, it's going to be put in disk, precisely at address %d\n", mmu -> pagetable -> pages[i].f*FRAME_SIZE);
                            fflush(stdout);


                            fseek(swap_file, mmu -> pagetable -> pages[i].f*FRAME_SIZE, SEEK_SET);

                            ret = fwrite(victim, sizeof(char), sizeof(victim) - 1, swap_file);
                            if (ret <= 0) {
                                //printf("error while writing the victim frame in the swap file");
                            }

                            mmu -> pagetable -> pages[p].flags |= Valid | Write | Read;


                            // if .f == -1 frame_in_disc is just an array of 0, it's useless to copy it.
                            if (mmu -> pagetable -> pages[p].f != -1) {
                                strcpy(mmu -> ram -> frames[mmu -> pagetable -> pages[p].f].frameblock, frame_in_disc);
                            }

                            mmu -> pagetable -> pages[p].f = mmu -> pagetable -> pages[i].f;

                            ret = fclose(swap_file);
                            if (ret != 0) {
                                //printf("error in closing the swap file\n");
                            }

                            return;

                        }
                    }


                    
                }

            }

            else {
                ////printf("this page: %d is invalid or unswappable, precisely its flag are Valid %d Unswappable %d Write %d Read %d:\n", i, mmu -> pagetable -> pages[i].flags & Valid, mmu -> pagetable -> pages[i].flags & Unswappable, mmu -> pagetable -> pages[i].flags & Read, mmu -> pagetable -> pages[i].flags & Write);
            }

        }

        // if (k == 2) {
        //     exit(0);
        // }

        //printf("we did not found a frame in this iteration of the RAM, let's try another iteration\n");
        
    }

    
  

}


