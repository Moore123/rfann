#include <stdio.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <string.h>
#include <pthread.h>
#include "fann.h"
#include "nncommon.h"

static void R_finalizeFannHandle(SEXP);
static void R_finalizeFannData(SEXP);

SEXP makeFannRObject(Fann *obj, int addFinalizer) {
	SEXP ans, klass, ref;
	int p=0;

	if(!obj) {
	PROBLEM "NULL Fann handle being returned\n"
	ERROR;
	}
	PROTECT(klass = MAKE_CLASS("FannHandle"));p++;
	PROTECT(ans = NEW(klass));p++;
	PROTECT(ref = R_MakeExternalPtr((void *) obj, Rf_install("FannHandle"), R_NilValue));p++;

	if(addFinalizer)
		 R_RegisterCFinalizer(ref, R_finalizeFannHandle);
	ans = SET_SLOT(ans, Rf_install("ref"), ref);

	UNPROTECT(p);

	return(ans);
} 



SEXP makeFannDataRObject(struct fann_train_data *obj, int addFinalizer) {
	SEXP ans, klass, ref;
	int p=0;

	if(!obj) {
	PROBLEM "NULL Fann data being returned"
	ERROR;
	}
	PROTECT(klass = MAKE_CLASS("FannData"));p++;
	PROTECT(ans = NEW(klass));p++;
	PROTECT(ref = R_MakeExternalPtr((void *) obj, Rf_install("FannData"), R_NilValue));p++;

	if(addFinalizer)
		 R_RegisterCFinalizer(ref, R_finalizeFannData);
	ans = SET_SLOT(ans, Rf_install("ref"), ref);

	UNPROTECT(p);

	return(ans);
} 

SEXP makeFannCodeRObject(FANNCODE val) {
	SEXP ans;
	ans = allocVector(INTSXP, 1);
	INTEGER(ans)[0] = val;
	return(ans);
}

Fann *getFannRObject(SEXP obj) {
	Fann *ann;
	SEXP ref;

	if(TYPEOF(obj) != EXTPTRSXP) {
   		ref = GET_SLOT(obj, Rf_install("ref"));
	} else 
        	ref = obj;

	ann = (Fann *) R_ExternalPtrAddr(ref);
	if(!ann) {
	ndebug("ann R_ExternalPtrAddr NULL\n");
	PROBLEM "Stale Fann handle being passed to libfann"
	ERROR;
	}

	if(R_ExternalPtrTag(ref) != Rf_install("FannHandle")) {
	PROBLEM "External pointer with wrong tag passed to libfann. Was %s",
                        CHAR(PRINTNAME(R_ExternalPtrTag(ref)))
	ERROR;
	}

	return(ann);
}

struct fann_train_data *getFannDataRObject(SEXP obj) {
	struct fann_train_data *annData;
	SEXP ref;
	if(TYPEOF(obj) != EXTPTRSXP)
   		ref = GET_SLOT(obj, Rf_install("ref"));
	else
       ref = obj;

	annData = (struct fann_train_data *) R_ExternalPtrAddr(ref);
	if(!annData) {
	PROBLEM "Stale Fann data being passed to libfann"
	ERROR;
	}

	if(R_ExternalPtrTag(ref) != Rf_install("FannData")) {
	PROBLEM "External pointer with wrong tag passed to libfann. Was %s",
                        CHAR(PRINTNAME(R_ExternalPtrTag(ref)))
	ERROR;
	}

	return(annData);
}

static void R_finalizeFannHandle(SEXP h)
{
   Fann *ann = getFannRObject(h);

   if(ann) {
   	fann_destroy(ann->ann);
	free(ann);
   }
}

static void R_finalizeFannData(SEXP h)
{
   struct fann_train_data *annData = getFannDataRObject(h);

   if(annData) {
   	fann_destroy_train(annData);
   }
}

SEXP R_fann_version_info(){
	return(mkString(FANN_CONF_VERSION));
}

SEXP R_fann_expand_create(SEXP list){

	SEXP retAnn;

	Fann *ann;

	ann = (Fann *) calloc(1,sizeof(Fann));
	R_fann_SetDefault(ann);
	ann->annType = FANN_EXPAND;

	retAnn = makeFannRObject(ann,TRUE);
	R_fann_SetOpt(retAnn,list);

	return(retAnn);
	
}

SEXP R_fann_cascade_create(SEXP list) {

	Fann *ann;
	SEXP retAnn;

	ann = (Fann *) calloc(1,sizeof(Fann));
	R_fann_SetDefault(ann);
	ann->annType = FANN_CASCADE;
	retAnn = makeFannRObject(ann,TRUE);
	R_fann_SetOpt(retAnn,list);
	
	return(retAnn);
}


SEXP R_fann_sparse_create(SEXP list) {

	SEXP retAnn;
	Fann *ann;

	(ann = (Fann *) calloc(1,sizeof(Fann)));
	R_fann_SetDefault(ann);
	ann->annType = FANN_SPARSE;

	retAnn = makeFannRObject(ann,TRUE);
	R_fann_SetOpt(retAnn,list);
	
    	return(retAnn);
}



SEXP R_fann_create(SEXP list) {

	Fann *ann;
	SEXP retAnn;

	(ann = (Fann *) calloc(1,sizeof(Fann)));
	R_fann_SetDefault(ann);
	ann->annType = FANN_SIMPLE;
	retAnn = makeFannRObject(ann,TRUE);
	R_fann_SetOpt(retAnn,list);
	
    	return(retAnn);
}

void doFannCreate(Fann *ann){

	switch(ann->annType) {

		case FANN_SIMPLE:
			ann->ann = fann_create_standard(ann->num_layers, 
				ann->num_input, ann->num_neurons_hidden, ann->num_output);
			break;
	
		case FANN_SPARSE:
			ann->ann = fann_create_sparse_array(ann->connection_rate ,ann->num_layers, 
				ann->layers);
			break;
			
		case FANN_CASCADE:
			ann->ann = fann_create_shortcut_array(ann->num_layers, ann->layers);
			break;

		case FANN_EXPAND:			
			ann->ann = fann_create_standard_array(ann->num_layers, ann->layers);			
			break;
			
		default :
			ndebug("Error parameter when create fann");
			

		}

}

SEXP R_fann_data(SEXP INPUT,SEXP OUTPUT) {
	struct fann_train_data *annData;
	
	annData = (struct fann_train_data *) read_data_from_R(INPUT,OUTPUT);

	if(annData != NULL) {
		return(makeFannDataRObject(annData,TRUE));
	} else  {
		ndebug("fann_data_from_R get NULL\n");
		return(R_NilValue);
	}
}


SEXP R_fannTrainOnData(SEXP obj,SEXP DATA)
{
	Fann *ann;
	struct fann_train_data *data;
	char buf[256];
	int i,j;

	ann = getFannRObject(obj);
	if(ann) {
		data = getFannDataRObject(DATA);
	
		if(data) {
#ifdef DEBUG
		for(i=0;i<data->num_data;i++) {
		  for(j=0;j<data->num_input;j++) {
			sprintf(buf,"[%d,%d] = %f\t",i,j,data->input[i][j]);
			ndebug(buf); }
		
		  for(j=0;j<data->num_output;j++) {
			sprintf(buf,"Y [%d,%d] = %f\n",i,j,data->output[i][j]);
			ndebug(buf); }

		}
#endif		
		if(ann->annType == FANN_CASCADE) 
		fann_cascadetrain_on_data(ann->ann, data, 
			ann->max_neurons, ann->epochs_between_reports, ann->desired_error);
	
		 else  
		  fann_train_on_data(ann->ann, data, 
		  	ann->max_epochs, ann->epochs_between_reports, ann->desired_error);

		}else {
		PROBLEM "getFannDataRObject ann_data failed"
		ERROR;
		return(R_NilValue);
		}
	}else {
	PROBLEM "getFannRObject ann failed"
	ERROR;
	return(R_NilValue);
	}

	return(mkString("Train OK"));
	
}

SEXP R_fann_train(SEXP obj,SEXP INPUT,SEXP OUTPUT)
{
	Fann *ann;
	struct fann_train_data *annData;
	int max_epochs,epochs_between_reports,desired_error;
	double *plist;
	SEXP retval;

	annData = (struct fann_train_data *) read_data_from_R(INPUT,OUTPUT);
	
	ann = getFannRObject(obj);
	if(ann) {
		switch (TYPEOF(OUTPUT)) {
			case VECSXP:
				ann->input_type = RList;
			break;
			case INTSXP:
				ann->input_type = Rint;
			break;
			case REALSXP:
				ann->input_type = Rreal;
			break;
			default:
			break;
		}
	}	

	if(annData != NULL) {

	    if(ann->init_weight == INIT)
		fann_init_weights(ann->ann,annData);
	    else
		fann_randomize_weights(ann->ann, ann->randomize_weight_low, ann->randomize_weight_low);
		
	   retval=R_fannTrainOnData(obj,makeFannDataRObject(annData,TRUE));
	   return(retval);
	} else 	
	   return(mkString("create annData from R Failed"));
	
}

SEXP R_fannTestOnData(SEXP obj,SEXP INPUT)
{
	fann_type *test_out;
	Fann *ann;
	struct fann_train_data *annData;
	int i;
		
	annData =  getFannDataRObject(INPUT);
		
	if(annData==NULL){
		PROBLEM "getFannDataRObject ann_data failed"
		ERROR;
		return(R_NilValue);
	}
	
	ann = getFannRObject(obj);

	if(ann) {
	 fann_reset_MSE(ann->ann);
	 test_out = fann_test(ann->ann, annData->input[i], annData->output[i]);
	 //fann_test_data(ann, annData);
	}else {
	PROBLEM "getFannRObject ann failed"
	ERROR;
	return(R_NilValue);
	}

	//fann_test_data(ann, test_data));
}

SEXP R_fann_test(SEXP obj,SEXP INPUT,SEXP OUTPUT) {

	fann_type **test_out,*output;
	Fann *ann;
	struct fann_train_data *annData;
	SEXP result;
	char buf[256];
	int i;
		
	annData = (struct fann_train_data *) read_data_from_R(INPUT,OUTPUT);
	if(annData==NULL){
		PROBLEM "getFannDataRObject ann_data failed"
		ERROR;
		return(R_NilValue);
	}
	
	ann = getFannRObject(obj);

	if(ann) {

	test_out = (fann_type **) calloc(annData->num_data,sizeof(fann_type *));
	for(i = 0; i != annData->num_data; i++){
	    test_out[i] = (fann_type *) calloc(annData->num_output,sizeof(fann_type));
	   }

	for(i = 0; i != annData->num_data; i++){
		fann_reset_MSE(ann->ann);
		output = fann_test(ann->ann, annData->input[i], annData->output[i]);
		memcpy(test_out[i],output,annData->num_output*sizeof(fann_type));
	   }

	result=dump_data_to_R(ann,annData,test_out);

	for(i = 0; i != annData->num_data; i++){
		free(test_out[i]);
	   }
	free(test_out);

	}else {
	PROBLEM "getFannRObject ann failed"
	ERROR;
	result = R_NilValue;
	}

	return(result);

}

SEXP R_fann_read(SEXP FNAME)
{
	Fann *ann;
	struct stat fst;

	ann = (Fann *) calloc(1,sizeof(Fann));

	if(stat(CHAR(STRING_ELT(FNAME,0)),&fst)==0) {	

        ann->ann = fann_create_from_file(CHAR(STRING_ELT(FNAME,0)));
	  if(ann->ann!=NULL)
	  	 R_fann_GetOpts(ann);

	} else {
	   ndebug("NULL Fann handle being returned\n");
	   ann = NULL;
	}
    return(makeFannRObject(ann,TRUE));	
}

SEXP R_fann_save(SEXP obj,SEXP FNAME)
{
	Fann *ann;
	ann = getFannRObject(obj);
	if(ann) {
	  fann_save(ann->ann, CHAR(STRING_ELT(FNAME,0)));
	  return(mkString("Saved"));
	} else {
	  return(R_NilValue);
	}
}



SEXP R_fann_t(SEXP X,SEXP Y){

  SEXP  resultSEXP;

  SEXP xdims;

  int *d;
  
  int i,n;

  char *buf,*s;

  buf = (char *)calloc(1024,sizeof(char));

 PROTECT(resultSEXP = allocVector(STRSXP,2));
/*
SET_STRING_ELT(resultSEXP,0,(mkChar("Hello1")));
SET_STRING_ELT(resultSEXP,1,(mkChar("the world")));

if(TYPEOF(X) == VECSXP)			
	R_ShowMessage("This is a VECSXP X"); 
else if(TYPEOF(X) == LISTSXP)			
	R_ShowMessage("This is a LIST X");
else if(TYPEOF(X) == STRSXP)			
	R_ShowMessage("This is a STR X");
else if(TYPEOF(X) == REALSXP)			
	R_ShowMessage("This is a REAL X");
else if(TYPEOF(X) == INTSXP)			
	R_ShowMessage("This is a INT X");

    s = CHAR(STRING_ELT(R_data_class(X, TRUE), 0));
	R_ShowMessage(s);


  if(isFrame(X)) {
      R_ShowMessage("isFrame");
	}
  if(isList(X)) {
      R_ShowMessage("isLIST");
	}
  if(isMatrix(X)) {
      R_ShowMessage("isMATRIX");
	}
  if(isReal(X)) {
      R_ShowMessage("isREAL");
	}

  if(TYPEOF(X) == LISTSXP )
      R_ShowMessage("isLISTSXP");

  sprintf(buf+strlen(buf),"X dims ncol is %d\n",ncols(X));

  sprintf(buf+strlen(buf),"X dims nrow is %d\n",nrows(X));

  xdims = getAttrib(X, R_DimSymbol);
	
if(xdims == R_NilValue) 
  sprintf(buf+strlen(buf),"X getAttrib == R_NilValue length %d\n",LENGTH(X));

else {
  sprintf(buf+strlen(buf),"X dims[0] is %d\n",INTEGER(xdims)[0]);
  sprintf(buf+strlen(buf),"X dims[1] is %d\n",INTEGER(xdims)[1]);

   }


  R_ShowMessage(buf);

*/

  sprintf(buf,"in fann_t : typeof %d",TYPEOF(X));
  R_ShowMessage(buf);

/* if(isFrame(X)) {

  xdims = getAttrib(X, R_DimSymbol);

       sprintf(buf+strlen(buf),"X dim LENGTH %d\n",LENGTH(X));
//       sprintf(buf+strlen(buf),"X dim col %d\n",INTEGER(xdims)[0]);
//       sprintf(buf+strlen(buf),"X dim row = %d\n",INTEGER(xdims)[1]);
	R_ShowMessage(buf);
}
//  sprintf(buf,"x %f",sexpValue(X,(int)sexpValue(Y,0,0),(int)sexpValue(Y,0,1)));
//  resultSEXP = getAttrib(CAR(X), R_DimSymbol); 
*/
    SET_STRING_ELT(resultSEXP,0,(mkChar(buf)));

    UNPROTECT(1);
    free(buf); 
    return(resultSEXP);
}
