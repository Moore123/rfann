#define MAX_QUERY_LEN           2048
#define TRUE 1
#define FALSE 0

typedef struct NearNStruct {
      int index;
      unsigned char set;
      double dist;
} NNStruct;

typedef struct { 
  char stkname[9];
  double *clone;
  NNStruct *list;
} NNList;

int ndebug(char *str);
