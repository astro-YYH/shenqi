#ifndef _EVALUATOR_H_
#define _EVALUATOR_H_

enum TreeWalkReduceMode {
    TREEWALK_PRIMARY,
    TREEWALK_GHOSTS,
};
void TreeWalk_allocate_memory(void);
struct _TreeWalk;

typedef struct {
    MyIDType ID;
    int NodeList[NODELISTLENGTH];
    double Pos[3];
} TreeWalkQueryBase;

typedef struct {
    MyIDType ID;
} TreeWalkResultBase;

typedef struct {
    struct _TreeWalk * tw;

    int mode; /* 0 for Primary, 1 for Secondary */

    int *exportflag;
    int *exportnodecount;
    int *exportindex;
    int64_t Ninteractions;
    int64_t Nnodesinlist;
    int * ngblist;
} LocalTreeWalk;

typedef int (*TreeWalkVisitFunction) (const int target,
        TreeWalkQueryBase * input, TreeWalkResultBase * output, LocalTreeWalk * lv);

typedef int (*TreeWalkIsActiveFunction) (const int i);

typedef void (*TreeWalkFillQueryFunction)(const int j, TreeWalkQueryBase * data_in);
typedef void (*TreeWalkReduceResultFunction)(const int j, TreeWalkResultBase * data_result, const enum TreeWalkReduceMode mode);

typedef struct _TreeWalk {
    /* name of the evaluator (used in printing messages) */
    char * ev_label;

    TreeWalkVisitFunction ev_visit;
    TreeWalkIsActiveFunction ev_isactive;
    TreeWalkFillQueryFunction ev_copy;
    TreeWalkReduceResultFunction ev_reduce;

    int * ngblist;

    char * dataget;
    char * dataresult;

    int UseNodeList;
    int UseAllParticles; /* if 1 use all particles
                             if 0 use active particles */
    size_t query_type_elsize;
    size_t result_type_elsize;

    /* performance metrics */
    double timewait1;
    double timewait2;
    double timecomp1;
    double timecomp2;
    double timecommsumm1;
    double timecommsumm2;
    int64_t Ninteractions;
    int64_t Nnodesinlist;
    int64_t Nexport_sum;
    int64_t Niterations;

    /* internal flags*/

    int Nexport;
    int Nimport;
    int BufferFullFlag;
    int BunchSize;

    struct ev_task * PrimaryTasks;
    int * PQueue;
    int PQueueSize;

    /* per worker thread*/
    int *currentIndex;
    int *currentEnd;
} TreeWalk;

void treewalk_run(TreeWalk * tw);
int * treewalk_get_queue(TreeWalk * tw, int * len);

/*returns -1 if the buffer is full */
int ev_export_particle(LocalTreeWalk * lv, int target, int no);

#define TREEWALK_REDUCE(A, B) (A) = (mode==TREEWALK_PRIMARY)?(B):((A) + (B))
#endif
