#include <stdio.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <pthread.h>
#include "fann.h"
#include "nncommon.h"

static char const *const FannTypeName[] = {
 "Simple",
 "Cascade",
 "Expand",
 "Sparse",
 "!!WrongSetting!!"
};

static char const *const FannInitWeight[] = {
"Init",
"Random",
"None"
};

#define FannTypeLen 4
#define FannInitLen 3

FannOpt fannopt[] = {

/* Save network parameters */
{ "------------------------------------------------------------",OptInt,FANN_SPARE_LINE},
{ "network type [ LAYER | SHORTCUT ] Can't be set",OptInt,FANN_SPARE_LINE},
{ "network_type=%s",OptInt,NETWORK_TYPE},
{ "",OptInt,FANN_SPARE_LINE},

{ "------------------------------------------------------------",OptInt,FANN_SPARE_LINE},
{ "ann type [ Simple | Cascade | Expand | Sparse ] For Create",OptInt,FANN_SPARE_LINE},
{ "ann_type=%s",OptInt,ANN_TYPE},
{ "",OptInt,FANN_SPARE_LINE},

{ "-------------------- Basic parameters --------------------",OptInt,FANN_SPARE_LINE},
{ "init_weight=%s",OptInt,INIT_WEIGHT},
{ "num_layers=%u",OptInt,NUM_LAYER},
{ "layer%u=%u",OptInt,annLAYERS},

{ "num_input=%d",OptInt,NUM_INPUT},
{ "num_output=%d",OptInt,NUM_OUTPUT},
{ "num_neurons_hidden=%d",OptInt,NUM_NEURONS_HIDDEN},
{ "desired_error=%f",OptFloat,DESIRED_ERROR},
{ "-------------------- Expand parameters --------------------",OptInt,FANN_SPARE_LINE},
{ "randomize_weight_high=%f",OptInt,RANDOMIZE_WEIGHT_HIGH},
{ "randomize_weight_low=%f",OptInt,RANDOMIZE_WEIGHT_LOW},
{ "learning_rate=%f",OptFloat,LEARNING_RATE},
{ "connection_rate=%f",OptFloat,CONNECTION_RATE},

{ "bit_fail_limit=%f",OptInt,BIT_FAIL_LIMIT},
{ "max_epochs=%u",OptInt,MAX_EPOCHS},
{ "epochs_between_reports=%u",OptInt,EPOCHS_BETWEEN_REPORT},
{ "learning_momentum=%f",OptFloat,LEARNING_MOMENTUM},
{ "------------------------------------------------------------",OptInt,FANN_SPARE_LINE},
{ "train algorithm: [ INCREMENTAL | BATCH | RPROP | QUICKPROP ]",OptInt,FANN_SPARE_LINE},
{ "training_algorithm=%s",OptInt,TRAINING_ALOGRITHM},
{ "",OptInt,FANN_SPARE_LINE},

{ "------------------------------------------------------------",OptInt,FANN_SPARE_LINE},
{ "error function: [ LINEAR | TANH ]",OptInt,FANN_SPARE_LINE},
{ "train_error_function=%s",OptInt,TRAIN_ERROR_FUNCTION},
{ "",OptInt,FANN_SPARE_LINE},

{ "------------------------------------------------------------",OptInt,FANN_SPARE_LINE},
{ "stop function: [ MSE | BIT ]",OptInt,FANN_SPARE_LINE},
{ "train_stop_function=%s",OptInt,TRAIN_STOP_FUNCTION},
{ "",OptInt,FANN_SPARE_LINE},

{ "-------------------- rprop parameters ---------------------",OptInt,FANN_SPARE_LINE},
{ "quickprop_decay=%f",OptFloat,QUICKPROP_DECAY},
{ "quickprop_mu=%f",OptFloat,QUICKPROP_MU},

{ "rprop_increase_factor=%f",OptFloat,RPROP_INCREASE_FACTOR},
{ "rprop_decrease_factor=%f",OptFloat,RPROP_DECREAST_FACTOR},
{ "rprop_delta_min=%f",OptFloat,RPROP_DELTA_MIN},
{ "rprop_delta_max=%f",OptFloat,RPROP_DELTA_MAX},
{ "rprop_delta_zero=%f",OptFloat,RPROP_DELTA_ZERO},
{ "",OptInt,FANN_SPARE_LINE},

//{ "activation_steepness=%f",OptInt,ACTIVATION_STEEPNESS},
//{ "activation_steepness_layer=%u",OptInt,ACTIVATION_STEEPNESS_LAYER},
{ "activation_steepness_hidden=%f",OptInt,ACTIVATION_STEEPNESS_HIDDEN},
{ "activation_steepness_output=%f",OptInt,ACTIVATION_STEEPNESS_OUTPUT},

{ "------------------------------------------------------------",OptInt,FANN_SPARE_LINE},
{ "Options below  [ LINEAR | THRESHOLD | THRESHOLD_SYMMETRIC ]",OptInt,FANN_SPARE_LINE}, 
{ "cont: [ SIGMOID| SIGMOID_STEPWISE| SIGMOID_SYMMETRIC | SIGMOID_SYMMETRIC_STEPWISE ]",OptInt,FANN_SPARE_LINE},
{ "cont: [ GAUSSIAN| GAUSSIAN_SYMMETRIC | GAUSSIAN_STEPWISE ]",OptInt,FANN_SPARE_LINE},
{ "cont: [ ELLIOT | ELLIOT_SYMMETRIC",OptInt,FANN_SPARE_LINE},	
{ "cont: [ LINEAR_PIECE | LINEAR_PIECE_SYMMETRIC ]",OptInt,FANN_SPARE_LINE},
{ "cont: [ SIN_SYMMETRIC| COS_SYMMETRIC | SIN | COS ]",OptInt,FANN_SPARE_LINE},
{ "------------------------------------------------------------",OptInt,FANN_SPARE_LINE},
//{ "activation_function=%s",OptInt,ACTIVATION_FUNCTION},
//{ "activation_function_layer=%s",OptInt,ACTIVATION_FUNCTION_LAYER},
{ "activation_function_hidden=%s",OptInt,ACTIVATION_FUNCTION_HIDDEN},
{ "activation_function_output=%s",OptInt,ACTIVATION_FUNCTION_OUTPUT},

{ "-------------------- cascade parameters -------------------",OptInt,CASCADE_SPARE_LINE},
//{ "cascade_activation_functions=%u",OptInt,CASCADE_ACTIVATION_FUNCTIONS},
//{ "cascade_activation_steepnesses=%u",OptInt,CASCADE_ACTIVATION_STEEPNESSES},
{ "max_neurons=%u",OptInt,MAX_NEURONS},
{ "cascade_output_change_fraction=%f",OptFloat,CASCADE_OUTPUT_CHANGE_FRACTION},
{ "cascade_output_stagnation_epochs=%u",OptInt,CASCADE_OUTPUT_STAGNATION_EPOCHS},
{ "cascade_candidate_change_fraction=%f",OptFloat,CASCADE_CANDIDATE_CHANGE_FRACTION},
{ "cascade_candidate_stagnation_epochs=%u",OptInt,CASCADE_CANDIDATE_STAGNATION_EPOCHS},
{ "cascade_max_out_epochs=%u",OptInt,CASCADE_MAX_OUT_EPOCHS},
{ "cascade_max_cand_epochs=%u",OptInt,CASCADE_MAX_CAND_EPOCHS},
{ "cascade_weight_multiplier=%f",OptInt,CASCADE_WEIGHT_MULTIPLIER},
{ "cascade_candidate_limit=%f",OptInt,CASCADE_CANDIDATE_LIMIT},
//{ "cascade_num_candidate_groups",OptInt,CASCADE_NUM_CANDIDATE_GROUPS},

{ NULL,0,0 }
};

#define CASCADE_PARAM_COUNT 9

#define pfxTRAIN 11
#define TrainNames 4
#define pfxTRAINError 15

#define actNames 18
#define pfxACT 5
#define LayerLen 5


SEXP R_fann_SetOpt(SEXP obj,SEXP list) {
	
	Fann *ann;

	int i,j,n = 0;
	int found = FALSE;
	int LayerN;
	char buf[256];

	SEXP pname,Lvalue,values;

	ann = getFannRObject(obj);

	if(ann) {

	pname = getAttrib(list, R_NamesSymbol);	

	while(1) { 
	
	if ( n >= LENGTH(list) ) break;

	values = VECTOR_ELT(list,n);

	i=0;
	found = FALSE;

 	while((fannopt[i].optname != NULL ) && (found != TRUE)){	

	if( fannopt[i].opt == annLAYERS ) {
	   if(strncmp(fannopt[i].optname,CHAR(STRING_ELT(pname,n)),LayerLen) == 0 )  {
		j=0;
		if( ann->num_layers < 2 ) {
			PROBLEM "Please prefer num_layers > 2 first"
			ERROR; }

		if( ann->layers == NULL) 
			ann->layers = (int *) calloc( ann->num_layers, sizeof(int));

		LayerN = 0;
		  
		sscanf(CHAR(STRING_ELT(pname,n)),"layer%d",&LayerN);
	
		if( ann->num_layers < LayerN ) { 
		   PROBLEM "Prefer layer[n] with n little than .or. equal to num_layers"
		   ERROR; 
		} else {
		  if(LayerN > 0) {
		      ann->layers[LayerN-1] = (int)REAL(VECTOR_ELT(list,n))[0];
		  	}
		  else { i++; continue; }
		}

	} else {
		i++; continue;
	       }

	} 
	else if(strncmp(fannopt[i].optname,CHAR(STRING_ELT(pname,n)),
					strlen(CHAR(STRING_ELT(pname,n))))!= 0 ) 
		{ i++; continue; }

	found = TRUE;

	switch(fannopt[i].opt) {

	case ANN_TYPE:
		j=0;
		  while(strcmp(FannTypeName[j],CHAR(STRING_ELT(values,0)))!=0){
		  	if(j >= FannTypeLen) break;
		  	j++;
		    }
		ann->annType = j;
		break;

	case INIT_WEIGHT:
		j=0;
		  while(strcmp(FannInitWeight[j],CHAR(STRING_ELT(values,0)))!=0){
		  	if(j >= FannInitLen) break;
		  	j++;
		    }
		ann->init_weight = j;
		break;
		
	case NUM_LAYER:
		ann->num_layers = (int)REAL(VECTOR_ELT(list,n))[0];
		if( ann->num_layers <2 ) {
			R_ShowMessage("Are you kidding me?");
			PROBLEM "Please prefer num_layers > 2 first"
			ERROR;
			}

		if(( ann->num_layers > 3 ) && ( ann->annType == FANN_SIMPLE)) 
			ann->annType = FANN_EXPAND;

		if(ann->layers != NULL)  {
			free(ann->layers);
			ann->layers = NULL;			
			}

		if( ann->layers == NULL) 
			ann->layers = (int *) calloc( ann->num_layers, sizeof(int));
		
		break;

	case NUM_INPUT:
		ann->num_input = (int)REAL(VECTOR_ELT(list,n))[0];
		break;	

	case NUM_OUTPUT:
		ann->num_output = (int)REAL(VECTOR_ELT(list,n))[0];
		break;	

	case NUM_NEURONS_HIDDEN:
		ann->num_neurons_hidden = (int)REAL(VECTOR_ELT(list,n))[0];
		break;	

	case DESIRED_ERROR:
		ann->desired_error = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case LEARNING_RATE:
		ann->learning_rate = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case CONNECTION_RATE:
		ann->connection_rate = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case NETWORK_TYPE:
		break;
		
	case LEARNING_MOMENTUM:		
		ann->learning_momentum = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case BIT_FAIL_LIMIT:
		ann->bit_fail_limit = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case TRAINING_ALOGRITHM:
		j=0;
		  while(strcmp(FANN_TRAIN_NAMES[j]+pfxTRAIN,CHAR(STRING_ELT(values,0)))!=0){
		  	if(j >= TrainNames ) break;
		  	j++;
		    }
		ann->training_algorithm = j;

		break;
		
	case TRAIN_ERROR_FUNCTION:
		j=0;
		  while(strcmp(FANN_ERRORFUNC_NAMES[j]+pfxTRAINError,CHAR(STRING_ELT(values,0)))!=0){
		  	if(j >= 1 ) break;
		  	j++;
		    }
		ann->train_error_function = j;
		break;
		
	case TRAIN_STOP_FUNCTION:
		j=0;
		  while(strcmp(FANN_STOPFUNC_NAMES[j]+pfxTRAINError-1,CHAR(STRING_ELT(values,0)))!=0){
		  	if(j >= 1 ) break;
		  	j++;
		    }
		ann->train_stop_function = j;
		break;
		
	case QUICKPROP_DECAY:
		ann->quickprop_decay = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case QUICKPROP_MU:
		ann->quickprop_mu= (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case RPROP_INCREASE_FACTOR:
		ann->rprop_increase_factor = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case RPROP_DECREAST_FACTOR:
		ann->rprop_decrease_factor = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case RPROP_DELTA_MIN:
		ann->rprop_delta_min = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case RPROP_DELTA_MAX:
		ann->rprop_delta_max = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case RPROP_DELTA_ZERO:
		ann->rprop_delta_zero = (float)REAL(VECTOR_ELT(list,n))[0];
		break;

	case MAX_NEURONS:
		ann->max_neurons = (int)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case CASCADE_OUTPUT_CHANGE_FRACTION:
		ann->cascade_output_change_fraction = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case CASCADE_OUTPUT_STAGNATION_EPOCHS:
		ann->cascade_output_stagnation_epochs = (int)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case CASCADE_CANDIDATE_CHANGE_FRACTION:
		ann->cascade_candidate_change_fraction = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case CASCADE_CANDIDATE_STAGNATION_EPOCHS:
		ann->cascade_candidate_stagnation_epochs = (int)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case CASCADE_MAX_OUT_EPOCHS:
		ann->cascade_max_out_epochs = (int)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case CASCADE_MAX_CAND_EPOCHS:
		ann->cascade_max_cand_epochs = (int)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case CASCADE_NUM_CANDIDATE_GROUPS:
		ann->cascade_num_candidate_groups = 0;
		break;
		
	case MAX_EPOCHS:
		ann->max_epochs = (int)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case EPOCHS_BETWEEN_REPORT:
		ann->epochs_between_reports = (int)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case RANDOMIZE_WEIGHT_HIGH:
		ann->randomize_weight_high = (float)REAL(VECTOR_ELT(list,n))[0];
		break;
		
	case RANDOMIZE_WEIGHT_LOW:
		ann->randomize_weight_low = (float)REAL(VECTOR_ELT(list,n))[0];
		break;

	case ACTIVATION_STEEPNESS:
		ann->activation_steepness= (float)REAL(VECTOR_ELT(list,n))[0];
		break;
	case ACTIVATION_STEEPNESS_LAYER:
		ann->activation_steepness_layer= (int)REAL(VECTOR_ELT(list,n))[0];
		break;
	case ACTIVATION_STEEPNESS_HIDDEN:
		ann->activation_steepness_hidden= (float)REAL(VECTOR_ELT(list,n))[0];
		break;
	case ACTIVATION_STEEPNESS_OUTPUT:
		ann->activation_steepness_output = (float)REAL(VECTOR_ELT(list,n))[0];
		break;

	case ACTIVATION_FUNCTION:
		break;

	case ACTIVATION_FUNCTION_LAYER:
		break;

	case ACTIVATION_FUNCTION_HIDDEN:
		j=0;
		  while(strcmp(FANN_ACTIVATIONFUNC_NAMES[j]+pfxACT,CHAR(STRING_ELT(values,0)))!=0){
		  	if(j >= actNames) break;
		  	j++;
		    }
		ann->activation_function_hidden = j;
		break;

	case ACTIVATION_FUNCTION_OUTPUT:
		j=0;
		  while(strcmp(FANN_ACTIVATIONFUNC_NAMES[j]+pfxACT,CHAR(STRING_ELT(values,0)))!=0){
		  	if(j >= actNames) break;
		  	j++;
		    }
		ann->activation_function_output = j;
		break;
		
	default :
		break;
		} // end switch{}

	 if(found == TRUE) break;
	 else {
		ndebug("Not match ");
	  }
	 
	} // end while(fannopt[i].optname != null

	  n++;	
  	} // end while(n< LENGTH();
	
	} // end if(ann)

	return(mkChar("OK"));	
}


int R_fann_GetOpts(Fann *ann) {
	
	int retval=0;
	
	if(ann) {

		ann->network_type = fann_get_network_type(ann->ann);

		ann->num_layers = fann_get_num_layers(ann->ann);

		ann->layers = (int *) calloc(ann->num_layers,sizeof(int));
		fann_get_layer_array(ann->ann,ann->layers);

		ann->num_input= fann_get_num_input(ann->ann);
		ann->num_output = fann_get_num_output(ann->ann);
	
		ann->epochs_between_reports = ann->epochs_between_reports;
	
		ann->bit_fail_limit = fann_get_bit_fail_limit(ann->ann);
	
		ann->learning_rate = fann_get_learning_rate(ann->ann);
	
		ann->connection_rate = fann_get_connection_rate(ann->ann);
	
		ann->network_type = fann_get_network_type(ann->ann);
	
		ann->learning_momentum = fann_get_learning_momentum(ann->ann);
	
		ann->training_algorithm = fann_get_training_algorithm(ann->ann);
	
		ann->train_error_function = fann_get_train_error_function(ann->ann);
	
		ann->train_stop_function = fann_get_train_stop_function(ann->ann);
	
		ann->quickprop_decay = fann_get_quickprop_decay(ann->ann);
	
		ann->quickprop_mu = fann_get_quickprop_mu(ann->ann);
	
		ann->rprop_increase_factor = fann_get_rprop_increase_factor(ann->ann);
	
		ann->rprop_decrease_factor = fann_get_rprop_decrease_factor(ann->ann);
	
		ann->rprop_delta_min = fann_get_rprop_delta_min(ann->ann);
	
		ann->rprop_delta_max = fann_get_rprop_delta_max(ann->ann);
	
		ann->rprop_delta_zero = fann_get_rprop_delta_zero(ann->ann);

		if(ann->annType == FANN_CASCADE) {
			ann->cascade_output_change_fraction = fann_get_cascade_output_change_fraction(ann->ann);

			ann->cascade_candidate_limit = fann_get_cascade_candidate_limit(ann->ann);
			ann->cascade_weight_multiplier = fann_get_cascade_weight_multiplier(ann->ann);
		//	ann->cascade_activation_functions = fann_get_cascade_activation_functions(ann->ann);
		//	ann->cascade_activation_steepnesses = fann_get_cascade_activation_steepnesses(ann->ann);
			ann->cascade_output_stagnation_epochs = fann_get_cascade_output_stagnation_epochs(ann->ann);
			ann->cascade_candidate_change_fraction = fann_get_cascade_candidate_change_fraction(ann->ann);
			ann->cascade_candidate_stagnation_epochs =
						fann_get_cascade_candidate_stagnation_epochs(ann->ann);
			ann->cascade_max_out_epochs = fann_get_cascade_max_out_epochs(ann->ann);
			ann->cascade_max_cand_epochs = fann_get_cascade_max_cand_epochs(ann->ann);
			ann->cascade_num_candidate_groups = fann_get_cascade_num_candidate_groups(ann->ann);
			}
	}

	return(retval);
	
}


SEXP R_fann_opts(SEXP obj) {
	
	Fann *ann;
	char *buf;
	int i,n,cntOpt,p = 0;
	int alreadyN;
	ann = getFannRObject(obj);
	SEXP retSEXP;
	
	if(ann) {
	/* Save network parameters */
	i = cntOpt = n = 0;
	while(fannopt[i].optname!=NULL) {
		cntOpt ++;
		i++;
		}

	if(ann->annType != FANN_CASCADE ) cntOpt -= CASCADE_PARAM_COUNT;
	if(ann->num_layers>3) cntOpt += ann->num_layers;
	if(ann->annType != FANN_SIMPLE)  cntOpt--;

	PROTECT(retSEXP = allocVector(STRSXP,cntOpt));p++;
	buf = (char *) calloc(MID_STR_LEN,sizeof(char));

	i = n = 0;
	while(fannopt[i].optname!=NULL) {
	memset(buf,0x0,MID_STR_LEN*sizeof(char));

	alreadyN = FALSE;
		
	switch(fannopt[i].opt) {
		case ANN_TYPE:
			if(ann->annType == FANN_SIMPLE)	
				if(ann->num_layers > 3) 
					ann->annType = FANN_EXPAND;	
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname,
				FannTypeName[ann->annType]);
			break;
		case INIT_WEIGHT:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname,
				FannInitWeight[ann->init_weight]);
			break;
		case NUM_LAYER:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname,ann->num_layers);
			break;
		case annLAYERS:
			if(ann->annType != FANN_SIMPLE)	{
			   alreadyN = TRUE;
			    while(n < ann->num_layers) {
				snprintf(buf,MID_STR_LEN-2,fannopt[i].optname,n+1,ann->layers[n]);
			
				SET_STRING_ELT(retSEXP,i+n,(mkChar(buf)));
				n++;
			    }
			}
			break;
		case LASTER_LAYER:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->last_layer);
			break;
		case FIRST_LAYER:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->first_layer);
			break;
		case NUM_INPUT:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname,ann->num_input);
			break;	

		case NUM_OUTPUT:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname,ann->num_output);
			break;	

		case NUM_NEURONS_HIDDEN:
		        if(ann->annType == FANN_SIMPLE) 
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname,ann->num_neurons_hidden);
			break;	

		case DESIRED_ERROR:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname,ann->desired_error);
			break;

		case MAX_EPOCHS:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->max_epochs);
			break;			
		case EPOCHS_BETWEEN_REPORT:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->epochs_between_reports);
			break;
		case RANDOMIZE_WEIGHT_HIGH:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->randomize_weight_high);
			break;			
		case RANDOMIZE_WEIGHT_LOW:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->randomize_weight_low);
			break;

		case BIT_FAIL_LIMIT:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname,ann->bit_fail_limit);
			break;

		case LEARNING_RATE:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->learning_rate);
			break;
		case CONNECTION_RATE:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->connection_rate);
			break;
		case NETWORK_TYPE:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, FANN_NETTYPE_NAMES[ann->network_type]);
			break;
		case LEARNING_MOMENTUM:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->learning_momentum);
			break;
		case TRAINING_ALOGRITHM:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, FANN_TRAIN_NAMES[ann->training_algorithm]);
			break;
		case TRAIN_ERROR_FUNCTION:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, FANN_ERRORFUNC_NAMES[ann->train_error_function]);
			break;
		case TRAIN_STOP_FUNCTION:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, FANN_STOPFUNC_NAMES[ann->train_stop_function]);
			break;
		case QUICKPROP_DECAY:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->quickprop_decay);
			break;
		case QUICKPROP_MU:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->quickprop_mu);
			break;
		case RPROP_INCREASE_FACTOR:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->rprop_increase_factor);
			break;
		case RPROP_DECREAST_FACTOR:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->rprop_decrease_factor);
			break;
		case RPROP_DELTA_MIN:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->rprop_delta_min);
			break;
		case RPROP_DELTA_MAX:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->rprop_delta_max);
			break;
		case RPROP_DELTA_ZERO:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->rprop_delta_zero);
			break;

		case FANN_SPARE_LINE:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname);
			break;
			
		case CASCADE_SPARE_LINE:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname);
			break;

		case MAX_NEURONS:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->max_neurons);
			break;

		case CASCADE_OUTPUT_CHANGE_FRACTION:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_output_change_fraction);
			break;
		case CASCADE_CANDIDATE_LIMIT:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_candidate_limit);
			break;
		case CASCADE_WEIGHT_MULTIPLIER:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_weight_multiplier);
			break;
		case CASCADE_ACTIVATION_FUNCTIONS:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_activation_functions);
			break;
		case CASCADE_ACTIVATION_STEEPNESSES:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_activation_steepnesses);
			break;
		case CASCADE_OUTPUT_STAGNATION_EPOCHS:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_output_stagnation_epochs);
			break;
		case CASCADE_CANDIDATE_CHANGE_FRACTION:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_candidate_change_fraction);
			break;
		case CASCADE_CANDIDATE_STAGNATION_EPOCHS:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_candidate_stagnation_epochs);
			break;
		case CASCADE_MAX_OUT_EPOCHS:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_max_out_epochs);
			break;
		case CASCADE_MAX_CAND_EPOCHS:
			if(ann->annType == FANN_CASCADE)
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->cascade_max_cand_epochs);	
			break;
		case ACTIVATION_STEEPNESS:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->activation_steepness);
			break;

		case ACTIVATION_STEEPNESS_HIDDEN:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->activation_steepness_hidden);
			break;
		case ACTIVATION_STEEPNESS_OUTPUT:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, ann->activation_steepness_output);
			break;
		case ACTIVATION_FUNCTION:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, 
				FANN_ACTIVATIONFUNC_NAMES[ann->activation_function]);
			break;
		case ACTIVATION_FUNCTION_HIDDEN:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, 
				FANN_ACTIVATIONFUNC_NAMES[ann->activation_function_hidden]);
			break;
		case ACTIVATION_FUNCTION_OUTPUT:
			snprintf(buf,MID_STR_LEN-2,fannopt[i].optname, 
				FANN_ACTIVATIONFUNC_NAMES[ann->activation_function_output]);
			break;
		default:
			ndebug(fannopt[i].optname);
			break;
		}
		
		if((alreadyN == FALSE) && (strlen(buf)!=0))
			SET_STRING_ELT(retSEXP,i+n,(mkChar(buf)));
	     i++;
	  }
	}
	UNPROTECT(p);

	free(buf);

	return(retSEXP);
	
}



SEXP R_fann_CommitOpts(SEXP handle)
{
	Fann *ann;

	int style = FANN_TRAIN_BATCH;
	int i,n=10;
	
	ann = getFannRObject(handle);
	 if(!ann) {
		return(R_NilValue);
		}

	 if(ann->ann == NULL) {
		 doFannCreate(ann);
	 	}


	fann_set_learning_rate(ann->ann, ann->learning_rate);
	fann_set_quickprop_decay(ann->ann, ann->quickprop_decay);
	fann_set_quickprop_mu(ann->ann, ann->quickprop_mu);

	fann_set_bit_fail_limit(ann->ann, ann->bit_fail_limit);
	fann_set_activation_steepness_output(ann->ann, ann->activation_steepness_output);

	fann_set_training_algorithm(ann->ann, ann->training_algorithm);
	
	fann_set_activation_function_hidden(ann->ann, ann->activation_function_hidden);

	fann_set_activation_function_output(ann->ann, ann->activation_function_output);

	fann_set_train_error_function(ann->ann, ann->train_error_function);
	fann_set_activation_steepness_hidden(ann->ann, ann->activation_steepness_hidden);
	fann_set_activation_steepness_output(ann->ann, ann->activation_steepness_output);

	fann_randomize_weights(ann->ann, ann->randomize_weight_low, ann->randomize_weight_low);
	

	if(ann->annType == FANN_CASCADE) {
		/*
		fann_set_cascade_weight_multiplier(ann, 0.4);
		fann_set_cascade_candidate_limit(ann, 1000.0);
		*/		
		fann_set_cascade_weight_multiplier(ann->ann, ann->cascade_weight_multiplier);
		fann_set_cascade_max_out_epochs(ann->ann, ann->cascade_max_out_epochs);
		
		fann_set_cascade_output_change_fraction(ann->ann, ann->cascade_output_change_fraction);
		fann_set_cascade_candidate_change_fraction(ann->ann, ann->cascade_candidate_change_fraction);
	}
	
	/*
	steepnesses = (fann_type *)calloc(1,  sizeof(fann_type));
	steepnesses[0] = (fann_type)1;
	fann_set_cascade_activation_steepnesses(ann, steepnesses, 1);
	*/	
	
	fann_set_train_stop_function(ann->ann, ann->train_stop_function);
	if(ann->showparam) 
	fann_print_parameters(ann->ann);

	return(R_NilValue);
	
	}

	
	
void R_fann_SetDefault(Fann *ann) {

	ann->annType = FANN_SIMPLE;	
	ann->num_layers = 2;
	ann->last_layer = 4;
	ann->first_layer = 1;
	ann->learning_rate = 0.7f;
	ann->connection_rate = 0.2f;
	ann->network_type = FANN_NETTYPE_LAYER;
	ann->bit_fail_limit = 0.01f;
	ann->desired_error = 0.002f;

	ann->learning_momentum =0.7f;
	
	ann->training_algorithm = FANN_TRAIN_RPROP;
	ann->train_error_function = FANN_ERRORFUNC_TANH;
	ann->train_stop_function = FANN_STOPFUNC_MSE;
	
	ann->quickprop_decay = -0.0001f;
	ann->quickprop_mu= 1.75f;	
	ann->rprop_increase_factor = 1.2f;
	ann->rprop_decrease_factor = 0.5f;
	ann->rprop_delta_min = 0.0f;
	ann->rprop_delta_max = 50.0f;
	ann->rprop_delta_zero = 0;

	ann->cascade_output_change_fraction =0.0f;
	ann->cascade_output_stagnation_epochs = 0;
	ann->cascade_candidate_change_fraction = 0;
	ann->cascade_candidate_stagnation_epochs = 0;
	ann->cascade_max_out_epochs = 150;
	ann->cascade_max_cand_epochs = 0;	
	ann->cascade_num_candidate_groups = 0;

	ann->train_stop_function = FANN_STOPFUNC_BIT;

	ann->max_epochs = 10000;
	ann->epochs_between_reports = 1000;
	ann->randomize_weight_high = 0.35f;
	ann->randomize_weight_low = 0.35f;
	ann->num_input = 2;
	ann->num_output = 1;
	ann->num_neurons_hidden = 3;

	ann->activation_function = FANN_LINEAR;

	ann->activation_function_hidden = FANN_SIGMOID_SYMMETRIC;
	ann->activation_function_output= FANN_GAUSSIAN_SYMMETRIC;

	ann->activation_steepness = FANN_LINEAR;
	ann->activation_steepness_hidden = 2;

	ann->activation_steepness_output = 4;

	ann->layers = NULL;

	return; 
}
