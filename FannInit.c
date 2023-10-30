#include "nncommon.h"
#include "fann.h"

#define RegCallFun(a, b) {#a, (DL_FUNC)&a, b}

static R_CallMethodDef CallEntries[] = {

	RegCallFun(R_fann_CommitOpts, 1),

	RegCallFun(R_fann_sparse_create,1),
	RegCallFun(R_fann_cascade_create,1),
	RegCallFun(R_fann_expand_create,1),
	RegCallFun(R_fann_create, 1),
	
	RegCallFun(R_fann_train, 3),
	RegCallFun(R_fannTrainOnData, 2),

	RegCallFun(R_fann_test, 3),
	RegCallFun(R_fannTestOnData, 2),
	
	RegCallFun(R_fann_save, 2),	
	RegCallFun(R_fann_read, 2),
	
	RegCallFun(R_fann_data, 2),
	RegCallFun(R_fann_version_info, 0),

	RegCallFun(R_fann_opts,1),
	RegCallFun(R_fann_SetOpt,2),
	RegCallFun(R_fann_t, 2),

	{NULL, NULL, 0}
};


void
R_init_NN(DllInfo *dll)
{
    R_useDynamicSymbols(dll, TRUE);
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
}

