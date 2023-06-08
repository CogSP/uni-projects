
// TODO: check if 4 KB pages is right
#define VMS (1<<24)  //16 MB of Virtual Memory Space

// I tried using 4 * 1024 but it warns me about an overflow
// (not for VMS of any other constant, don't know why)
#define PAG_SIZE (1 << 12)  // 4 KB pages, is it right?
#define N_PAGES (VMS / PAG_SIZE) 

#define PMS (1<<20)       //1 MB of Physical Memory Space 
#define FRAME_SIZE PAG_SIZE 
#define N_FRAMES (PMS / PAG_SIZE)

// Physical Memory
typedef struct Frame {
    char frameblock[FRAME_SIZE];
} Frame;

typedef struct RAM {
    Frame frames[N_FRAMES];
} RAM;


// flags for each page
// with this arrangement we have:

// 1      1       1      1      1 
// Valid  Unswap  Read   Write  Free

// so I can use bitwise AND and OR 
// to check and change the flags
typedef enum {
  Valid = 1,
  Unswappable = 2, 
  Read = 4,
  Write = 8,
  Free = 16 //TODO: maybe it's not necessary
} Flags;

// Virtual Memory
typedef struct Page {
    int f; // the corresponding frame index
    Flags flags: 5;
} Page;


typedef struct PageTable {
    Page pages[N_PAGES]; 
} PageTable;

typedef struct FreeList {
    int arraylist[N_FRAMES];
    int in_pos; 
} FreeList;

typedef struct MMU {
    RAM* ram;
    PageTable* pagetable;
    FreeList* freelist;
} MMU;


void MMU_writeByte(MMU* mmu, int pos, char c);
char* MMU_readByte(MMU* mmu, int pos);
void MMU_exception(MMU* mmu, int pos);
