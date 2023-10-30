#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <errno.h>
#include <dirent.h>

#include <R.h>
#include <Rinternals.h>

#include <R_ext/Arith.h>
#include <R_ext/Error.h>
#include <Rdefines.h>
#include "R_ext/Rdynload.h"

#include <math.h>
#include "neardef.h"

SEXP ts2mx(SEXP Data,SEXP kdim) {

	SEXP result, dim;

	NNList *nndist;	
	int i,j, k, p,m,n, Rmax, Rcol , RunMax;
	double *xk;
	p = Rmax = Rcol = 0;

	int *Datadims = INTEGER(getAttrib(Data, R_DimSymbol));

	Rmax = Datadims[0];
	Rcol = Datadims[1];

	PROTECT(kdim = AS_NUMERIC(kdim));p++;
	xk = NUMERIC_POINTER(kdim);
	k = xk[0];

	if( Rmax < k) { 
		PROTECT(result = allocVector(VECSXP,1));p++;
		SET_VECTOR_ELT(result , 0 , ScalarString(NA_STRING));
		UNPROTECT(p);
		return(result); 
	}  
	RunMax = Rmax - k +1;

	PROTECT(result = allocVector(VECSXP,RunMax*((Rcol-1)*k+1)));p++;
	PROTECT(dim = allocVector(INTSXP, 2)); p++;
	INTEGER(dim)[0] = RunMax; INTEGER(dim)[1] = (Rcol-1)*k+1;
	setAttrib(result, R_DimSymbol, dim);

	nndist = (NNList *) calloc(sizeof(NNList), RunMax * k);

	for ( i = 0 ; i < RunMax ; i++ ) {
		memcpy(nndist[i].stkname,VECTOR_ELT(Data,i+k-1),8);
		SET_VECTOR_ELT(result , i , ScalarString(mkChar(nndist[i].stkname)));
	
	for( j=1,m=1,n=i+k-1; j<(Rcol-1)*k+1; j++, m++) { 
	    if(m >= Rcol) {  m=1; n--; }
		REAL(result)[j*RunMax+i]=REAL(Data)[m*Rmax+n]; 
	   }

       }

	free(nndist);

	UNPROTECT(p);

	return(result);
}


