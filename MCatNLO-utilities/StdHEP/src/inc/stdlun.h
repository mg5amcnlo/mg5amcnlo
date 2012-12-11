/*            StdHep I/O unit and stream information            */

#define LUN_ARRAY 16	/* I/O array size */
extern struct heplun {
    int lnhwrt;		/* event output unit number */
    int lnhrd;		/* event input unit number */
    int lnhout;		/* line printer output unit number */
    int lnhdcy;		/* decay file unit number */
    int lnhpdf;		/* PDF file unit number */
    int lnhdmp;		/* ascii dump file unit number */
    int lnhrdm[LUN_ARRAY];	/* unit number array for multiple I/O files */
} heplun_;

extern struct stdstr {
    int ixdrstr[LUN_ARRAY];	/* array of xdr stream addresses */
} stdstr_;
