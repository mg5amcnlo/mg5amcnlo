/* 
----------------------------------------------------------------
   This header collects the mcfio initial information
----------------------------------------------------------------
*/

extern struct stdhd1 {
char date[255];		/* MCFIO_CREATIONDATE: creation date */
char title[255];	/* MCFIO_TITLE: title */
char comment[255];	/* MCFIO_COMMENT: comment */
} stdhd1_;

extern struct stdhd2 {
int dlen;         /* actual lenght of date */
int tlen;         /* actual lenght of title */
int clen;         /* actual lenght of comment */
int numblocks;    /* MCFIO_NUMBLOCKS: number of blocks per event */
int blkids[50];   /* MCFIO_BLOCKIDS: list of block types */
} stdhd2_;
