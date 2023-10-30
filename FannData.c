#include <stdio.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <pthread.h>
#include "fann.h"
#include "nncommon.h"

int sexpXdims(SEXP DATA,XY CorR){

// CorR is column or Row

	int retval = 0;

	if(DATA == R_NilValue) return(retval);

	if(CorR == XCol) retval = ncols(DATA);
     	else if(CorR == YRow ) retval = nrows(DATA);	

	return(retval);

}

float sexpValue(SEXP DATA,int x,int y){

	int ncol, nrow;

	float retval;
	char buf[1024];

	ncol = sexpXdims(DATA,XCol);
	nrow = sexpXdims(DATA,YRow);
//	sprintf(buf,"ncol %d nrow %d >>>  x=%d y=%d\n",ncol,nrow,x,y);
//	ndebug(buf);

	if(isFrame(DATA))  {
	  PROBLEM "DataFrame type not correct nrows()"
	  ERROR; }
	
	switch (TYPEOF(DATA)) {
	case VECSXP:
	 retval = asReal(VECTOR_ELT(DATA,nrow*y + x));
	break;

	case INTSXP:
	case REALSXP:
	retval = REAL(DATA)[nrow*y + x];
	break;

	default:
	sprintf(buf,"typeof %d",TYPEOF(DATA));
	R_ShowMessage(buf);

	  break;
	}

	//sprintf(buf,"[%d ,%d] = %f\n",x,y,retval);
	//ndebug(buf);
	
	return(retval);

}

SEXP  dump_data_to_R (Fann *ann,struct fann_train_data *data,fann_type **out){

	SEXP dim;
	SEXP result;
	double *xptr;
	char buf[512];
	int i,j,p = 0;

	switch(ann->input_type) {

		case Rmatrix:
		case Rreal:
			PROTECT(result = NEW_NUMERIC(data->num_data * data->num_output));p++;
			xptr = NUMERIC_POINTER(result);
			break;
			
		case RList:
			PROTECT(result = allocVector(VECSXP,data->num_data * data->num_output)); p++;
			break;

		default:
			break;

	}

	for(i=0;i<data->num_data;i++){
		for(j=0;j<data->num_output;j++) {
			if(ann->input_type == RList)
               		   SET_VECTOR_ELT(result , data->num_data*j+i,ScalarReal(out[i][j]));
			else 
		          xptr[j*data->num_data+i] =(double)out[i][j];
		  }		
	}

	PROTECT(dim = allocVector(INTSXP, 2)); p++;
	INTEGER(dim)[0] = data->num_data; INTEGER(dim)[1] = data->num_output;
	setAttrib(result, R_DimSymbol, dim);

	UNPROTECT(p);
	return(result);

 }

 
 struct fann_train_data * read_data_from_R (SEXP X,SEXP Y){

	 unsigned int i, j;
	 int n;
 
	 fann_type *data_input, *data_output;
 
	 char *buf;
	 
	 struct fann_train_data *data =
	 		(struct fann_train_data *) malloc (sizeof (struct fann_train_data));
 
	 if (data == NULL) {
		  PROBLEM "data alloc error\n"
		  ERROR;
		  return(NULL);
	 }
 
	 fann_init_error_data ((struct fann_error *) data);

		 data->num_data = sexpXdims(X,YRow);
		 data->num_input = sexpXdims(X,XCol);
		 data->num_output = sexpXdims(Y,XCol);

	 buf = (char *) calloc(1024,sizeof(char));
 
	 data->input = (fann_type **) calloc (data->num_data, sizeof (fann_type *));
	 if (data->input == NULL) {
			R_ShowMessage("data->input FANN_E_CANT_ALLOCATE_MEM");
			fann_destroy_train (data);
			return NULL;
		  }
 
	 data->output = (fann_type **) calloc (data->num_data, sizeof (fann_type *));
	 if (data->output == NULL) {
			R_ShowMessage("data->output FANN_E_CANT_ALLOCATE_MEM");
			fann_destroy_train (data);
			return NULL;
		  }
 
	 data_input = (fann_type *) calloc (data->num_input * data->num_data, sizeof (fann_type));
	 if (data_input == NULL) {
			R_ShowMessage("data_input FANN_E_CANT_ALLOCATE_MEM");
			fann_destroy_train (data);
			return NULL;
		  }
	 if(Y!=R_NilValue) {
		 data_output = (fann_type *) calloc (data->num_output * data->num_data, sizeof (fann_type));
		 if (data_output == NULL) {
				R_ShowMessage("data_output FANN_E_CANT_ALLOCATE_MEM");
				fann_destroy_train (data);
				return NULL;
			  }
		}
 
	 for (i = 0; i < data->num_data; i++) {
		 data->input[i] = data_input;
		 data_input += data->num_input;
 
	 for (j = 0; j < data->num_input; j++) {
		 data->input[i][j] =  sexpValue(X,i,j);
	 }	 
		 
	 if(Y!=R_NilValue) {
 
		 data->output[i] = data_output;
		 data_output += data->num_output;
			 //ndebug("outY\n"); 
			 for (j = 0; j < data->num_output; j++)  {
				 data->output[i][j] =  sexpValue(Y,i,j);;
			 }
		 }
   }
	 free(buf);
	 return (data);
  }

