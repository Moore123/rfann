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

int ndebug(char *str) {
	int retval=0;
	FILE *fp;
	if((fp=fopen("./tmpfile","a+"))!=NULL) {
	fprintf(fp,str);
	fclose(fp);
	} else retval = -1;

	return(retval);
}

double CalcDistancek(double *dest, double *src, int dim,int k) {

	float retv = 0.0;
	int i;

	for( i = 1; i<dim ; i++)
		retv += pow((src[i]-dest[i]),k);

	return pow(retv,1/k);

}

double CalcDistance(double *dest, double *src, int dim) {

	float retv = 0.0;
	int i;

	for( i = 1; i<dim ; i++)
		retv += (src[i]-dest[i])*(src[i]-dest[i]);

	return sqrt(retv);

}

int FindNearK(SEXP result, NNList *collect,int dptr, int k) {

	int  retval = 0;

	int i, j , curr,sRmax ,sRcol;

	int *Datadims = INTEGER(getAttrib(result, R_DimSymbol));
  	sRmax = Datadims[0];
  	sRcol = Datadims[1];
	
	for ( i = 0 ; i < sRmax ; i++ ) {

	   if( i != dptr ) {

		   collect[i].list[0].dist
			   = CalcDistance(collect[i].clone,collect[dptr].clone,sRcol);

		   for ( j=1 ; j<k ; j++ ) {

		    if(collect[dptr].list[j].set==FALSE) {
			   collect[dptr].list[j].dist = collect[i].list[0].dist;
			   collect[dptr].list[j].index = i;
			   collect[dptr].list[j].set = TRUE;
			   break;

			   } else if( collect[i].list[0].dist < collect[dptr].list[j].dist ) {
		
			   if( j<k ) {
				curr = k-1;
				while(curr!=0) {
				  if(curr >= j) break;
			          collect[i].list[curr].index =	collect[i].list[curr-1].index;
			          collect[i].list[curr].dist = collect[i].list[curr-1].dist;
			          collect[i].list[curr].set = collect[i].list[curr-1].set ;
				  curr--; }
			         }
			   
			   collect[dptr].list[j].dist = collect[i].list[0].dist;
			   collect[dptr].list[j].index = i;
			   collect[dptr].list[j].set = TRUE;
			   break;

			 }
		   }

	   }  

	}


	return(retval);

}

int wheren(int idx) {

	int j=0,i;

	i = idx;
	while(i>0) j+=i--;
	return(j);
	
}

static int distcmp(NNStruct *dest,NNStruct *src) {
	return(dest->dist > src->dist);
}

SEXP Xnearest(SEXP Data,SEXP knear) {

	SEXP result, dim;
	NNList *nndist;

	NNStruct **distmatrix;

	double *xk;

	int i,j, p, k,Rmax, Rcol,NaStart ;

	p = k = Rmax = Rcol = NaStart = 0;

	int *Datadims = INTEGER(getAttrib(Data, R_DimSymbol));

	Rmax = Datadims[0];
	Rcol = Datadims[1];

	PROTECT(knear = AS_NUMERIC(knear));p++;
	xk = NUMERIC_POINTER(knear);
	k = xk[0] +1 ;

	PROTECT(result = allocVector(STRSXP,Rmax*k));p++;
	PROTECT(dim = allocVector(INTSXP, 2)); p++;
	INTEGER(dim)[0] = Rmax; INTEGER(dim)[1] = k;
	setAttrib(result, R_DimSymbol, dim);

	nndist = (NNList *) calloc(sizeof(NNList), Rmax * k);

	distmatrix = (NNStruct **) calloc(sizeof(NNStruct), Rmax);
	for ( i = 0 ; i < Rmax ; i++ ) distmatrix[i] = (NNStruct *) calloc(sizeof(NNStruct), Rmax);

	for ( i = 0 ; i < Rmax ; i++ ) {
		memcpy(nndist[i].stkname,CHAR(asChar(VECTOR_ELT(Data,i))),8);
		SET_STRING_ELT(result , i , mkChar(nndist[i].stkname));

		nndist[i].clone = (double *) calloc(sizeof(double) , Rcol);

		for (j = 1; j < Rcol; j++ ) {
			if(ISNA(asReal(VECTOR_ELT(Data,Rmax*j + i)))) {
				NaStart = i+1; break;
			} else
			nndist[i].clone[j]  =  asReal(VECTOR_ELT(Data,Rmax*j + i));
		}
	}
	
	for ( i = NaStart ; i < Rmax ; i++ ) {
	   for ( j = NaStart ; j < Rmax ; j++ ) {
		  distmatrix[i][j].index = j;
		  distmatrix[i][j].dist  
		    = CalcDistance(nndist[i].clone,nndist[j].clone,Rcol);
	   }

	   qsort(distmatrix[i],Rmax,sizeof(NNStruct),(void *)distcmp);

	   for( j=1 ; j<k; j++) {
	       SET_STRING_ELT(result , j*Rmax+i , mkChar(nndist[distmatrix[i][j].index].stkname));
	     }
       }

	for ( i = 0 ; i < Rmax ; i++ ) free(nndist[i].clone);
	free(nndist);
	for ( i = 0 ; i < Rmax ; i++ ) free(distmatrix[i]);
	free(distmatrix);

	UNPROTECT(p);

	return(result);
}


SEXP Onearest(SEXP Data,SEXP Target, SEXP knear) {

	SEXP result;
	NNList *nndist, *ntarget;

	NNStruct *distmatrix;

	double *xk;

	int i,j, p, k,Rmax, Rcol,NaStart;

	char buff[9];

	p = k = Rmax = Rcol = NaStart = 0;

	int *Datadims = INTEGER(getAttrib(Data, R_DimSymbol));

	Rmax = Datadims[0];
	Rcol = Datadims[1];

	PROTECT(knear = AS_NUMERIC(knear));p++;
	xk = NUMERIC_POINTER(knear);
	k = xk[0] +1 ;

	PROTECT(result = allocVector(STRSXP,k));p++;

	nndist = (NNList *) calloc(sizeof(NNList), Rmax * k);
	ntarget = (NNList *) calloc(sizeof(NNList), 1);
	ntarget->clone = (double *) calloc(sizeof(double) , Rcol);

	distmatrix = (NNStruct *) calloc(sizeof(NNStruct), Rmax);

	for ( i = 0 ; i < Rmax ; i++ ) {
		memcpy(nndist[i].stkname,CHAR(asChar(VECTOR_ELT(Data,i))),8);

		nndist[i].clone = (double *) calloc(sizeof(double) , Rcol);

		for (j = 1; j < Rcol; j++ ) {
		if(ISNA(asReal(VECTOR_ELT(Data,Rmax*j + i)))) {
	                 NaStart = i; break;
	          } else
			nndist[i].clone[j]  =  asReal(VECTOR_ELT(Data,Rmax*j + i));
		}
	}

	memset(buff,0x0,sizeof(char)*9);
	memcpy(buff,CHAR(asChar(VECTOR_ELT(Target,0))),8);
	SET_STRING_ELT(result , 0 , mkChar(buff));

	for (j = 1; j < Rcol; j++ ) {
		ntarget->clone[j]  =  asReal(VECTOR_ELT(Target,j));
	}
	

	for ( j = NaStart ; j < Rmax ; j++ ) {
	  distmatrix[j].index = j;
	  distmatrix[j].dist  
	    = CalcDistance(ntarget->clone,nndist[j].clone,Rcol);
	 }

	qsort(distmatrix,Rmax,sizeof(NNStruct),(void *)distcmp);

	for( j=0 ; j<k; j++) {
	      SET_STRING_ELT(result , j , mkChar(nndist[distmatrix[j].index].stkname));
	  }

	for ( i = 0 ; i < Rmax ; i++ ) free(nndist[i].clone);
	free(nndist);

	free(distmatrix);

	UNPROTECT(p);

	return(result);
}


SEXP Enearest(SEXP Data,SEXP knear) {

	SEXP result;
	NNList *nndist, *ntarget;

	NNStruct *distmatrix;

	double *xk;

	int i,j, p, k,Rmax, Rmax2,Rcol,NaStart;

	char buff[9];

	p = k = Rmax = Rmax2 = Rcol = NaStart = 0;

	int *Datadims = INTEGER(getAttrib(Data, R_DimSymbol));

	Rmax = Datadims[0];
	Rcol = Datadims[1];

	Rmax2 = Rmax-1;

	PROTECT(knear = AS_NUMERIC(knear));p++;
	xk = NUMERIC_POINTER(knear);
	k = xk[0] +1 ;

	PROTECT(result = allocVector(STRSXP,k));p++;

	nndist = (NNList *) calloc(sizeof(NNList), Rmax2 * k);
	ntarget = (NNList *) calloc(sizeof(NNList), 1);
	ntarget->clone = (double *) calloc(sizeof(double) , Rcol);

	distmatrix = (NNStruct *) calloc(sizeof(NNStruct), Rmax2);

	for ( i = 0 ; i < Rmax2 ; i++ ) {
		memcpy(nndist[i].stkname,CHAR(asChar(VECTOR_ELT(Data,i))),8);

		nndist[i].clone = (double *) calloc(sizeof(double) , Rcol);

		for (j = 1; j < Rcol; j++ ) {

		if(ISNA(asReal(VECTOR_ELT(Data,Rmax*j + i)))) {
			NaStart = i; break;
		} else
		    nndist[i].clone[j]  =  asReal(VECTOR_ELT(Data,Rmax*j + i));
		}
	}

	memset(buff,0x0,sizeof(char)*9);
	memcpy(buff,CHAR(asChar(VECTOR_ELT(Data,Rmax2))),8);
	SET_STRING_ELT(result , 0 , mkChar(buff));

	for (j = 1; j < Rcol; j++ ) {
	      ntarget->clone[j]  =  asReal(VECTOR_ELT(Data,Rmax*j+Rmax2));
	}
	

	for ( j = NaStart ; j < Rmax2 ; j++ ) {
	  distmatrix[j].index = j;
	  distmatrix[j].dist  
	    = CalcDistance(ntarget->clone,nndist[j].clone,Rcol);
	 }

	qsort(distmatrix,Rmax2,sizeof(NNStruct),(void *)distcmp);

	for( j=0 ; j<k; j++) {
	      SET_STRING_ELT(result , j , mkChar(nndist[distmatrix[j].index].stkname));
	  }

	for ( i = 0 ; i < Rmax2 ; i++ ) free(nndist[i].clone);
	free(nndist);

	free(distmatrix);

	UNPROTECT(p);

	return(result);
}


