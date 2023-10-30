#include <R.h>
#include <Rinternals.h>
#include <Rinterface.h>

#include <R_ext/Arith.h>
#include <R_ext/Error.h>
#include <Rdefines.h>
#include "R_ext/Rdynload.h"
#include "fann.h"

#define MID_STR_LEN 256
#define MAX_STR_LEN 1024

typedef enum XYcolrow {
	XCol = 0,
	YRow 
} XY;

typedef enum {
	FANN_SIMPLE=0,
	FANN_CASCADE,
	FANN_EXPAND,
	FANN_SPARSE,
	FANN_TypeERROR

} FannType;


typedef enum {
	RList = 0,
	Rmatrix,
	Rreal,
	Rint
} RInputType;

typedef enum {
	INIT = 0,
	RAND
}  InitWeight;
	
typedef enum {
   OptInt = 0,
   OptFloat,
   OptString,
   OptChar
} OptType;

typedef enum {
	ANN_TYPE = 0,
	SHOW_PARAM,
	INIT_WEIGHT,
	NUM_LAYER,
	annLAYERS,
	LASTER_LAYER,
	FIRST_LAYER,
	LEARNING_RATE,
	CONNECTION_RATE,
	MAX_EPOCHS,
	EPOCHS_BETWEEN_REPORT,
	RANDOMIZE_WEIGHT_HIGH,
	RANDOMIZE_WEIGHT_LOW,
	NUM_INPUT,
	NUM_OUTPUT,
	NUM_NEURONS_HIDDEN,
	DESIRED_ERROR,
	BIT_FAIL_LIMIT,
	NETWORK_TYPE,
	LEARNING_MOMENTUM,
	TRAINING_ALOGRITHM,
	TRAIN_ERROR_FUNCTION,
	TRAIN_STOP_FUNCTION,
	QUICKPROP_DECAY,
	QUICKPROP_MU,
	RPROP_INCREASE_FACTOR,
	RPROP_DECREAST_FACTOR,
	RPROP_DELTA_MIN,
	RPROP_DELTA_MAX,
	RPROP_DELTA_ZERO,
	
	ACTIVATION_FUNCTION,
	ACTIVATION_FUNCTION_LAYER,
	ACTIVATION_FUNCTION_HIDDEN,
	ACTIVATION_FUNCTION_OUTPUT,
	ACTIVATION_STEEPNESS,
	ACTIVATION_STEEPNESS_LAYER,
	ACTIVATION_STEEPNESS_HIDDEN,
	ACTIVATION_STEEPNESS_OUTPUT,
	MAX_NEURONS,
	CASCADE_SPARE_LINE,
	CASCADE_OUTPUT_CHANGE_FRACTION,
	CASCADE_OUTPUT_STAGNATION_EPOCHS,
	CASCADE_CANDIDATE_CHANGE_FRACTION,
	CASCADE_CANDIDATE_STAGNATION_EPOCHS,
	CASCADE_NUM_CANDIDATE_GROUPS,
	CASCADE_CANDIDATE_LIMIT,
	CASCADE_MAX_OUT_EPOCHS,
	CASCADE_MAX_CAND_EPOCHS,
	CASCADE_WEIGHT_MULTIPLIER,
	CASCADE_ACTIVATION_STEEPNESSES,
	CASCADE_ACTIVATION_FUNCTIONS,
	FANN_SPARE_LINE
	
} FANNCODE;

typedef struct FANN {
	
	int num_layers, last_layer , first_layer;
	int showparam;
	InitWeight init_weight;
	float learning_rate, connection_rate;
	enum fann_nettype_enum network_type;
	
	float learning_momentum;
	enum fann_train_enum training_algorithm;
	enum fann_train_enum train_error_function;
	enum fann_stopfunc_enum train_stop_function;

	float quickprop_decay, quickprop_mu;
	float rprop_increase_factor, rprop_decrease_factor, rprop_delta_min, rprop_delta_max, rprop_delta_zero;

	float cascade_output_change_fraction, cascade_candidate_change_fraction;
	int cascade_output_stagnation_epochs, cascade_candidate_stagnation_epochs,
		 cascade_max_out_epochs, cascade_max_cand_epochs, cascade_num_candidate_groups;
	float cascade_candidate_limit,cascade_weight_multiplier;
	int cascade_activation_functions, cascade_activation_steepnesses;
	
	int activation_steepness_layer; 
	float activation_steepness,activation_steepness_hidden, activation_steepness_output;
	
	int activation_function,activation_function_layer,
						activation_function_hidden, activation_function_output;
//	int activation_function_layer;

	int num_input,num_output;
	int num_neurons_hidden,max_neurons;
	int max_epochs,epochs_between_reports;
	
	float bit_fail_limit;
	int steepness_hidden,steepness_out;
	float desired_error;

	float randomize_weight_high,randomize_weight_low;

	unsigned int *layers;
	
	FannType annType;

	struct fann *ann;

	//========================================

	char *logfiles;
	RInputType input_type;
	int desired_output;
	
} Fann;

typedef struct FANNOPT {
	char *optname;
	int type;
	FANNCODE opt;
//	(void *) value;
} FannOpt;

extern int ndebug(char *); 

extern void R_fann_SetDefault(Fann *);

extern Fann *getFannRObject(SEXP);

extern float sexpValue(SEXP DATA,int,int);
extern struct fann_train_data * read_data_from_R (SEXP,SEXP);
SEXP  dump_data_to_R (Fann *,struct fann_train_data *,fann_type **);

extern int sexpXdims(SEXP,XY);

extern void doFannCreate(Fann *);


SEXP makeFannRObject(Fann *, int );
SEXP makeFannDataRObject(struct fann_train_data *, int );

SEXP R_fann_cascade_create(SEXP);
SEXP R_fann_sparse_create(SEXP);
SEXP R_fann_expand_create(SEXP);

SEXP R_fann_data(SEXP ,SEXP);
SEXP R_fann_train(SEXP ,SEXP ,SEXP);
SEXP R_fannTrainOnData(SEXP ,SEXP);
SEXP R_fann_cascade_train(SEXP ,SEXP);
SEXP R_fann_test(SEXP ,SEXP, SEXP);
SEXP R_fannTestOnData(SEXP, SEXP);
SEXP R_fann_CommitOpts(SEXP);


SEXP R_fann_t(SEXP,SEXP);

SEXP R_fann_opts(SEXP);
SEXP R_fann_SetOpt(SEXP,SEXP);
extern int R_fann_GetOpts(Fann *) ;

SEXP R_fann_read(SEXP);
SEXP R_fann_save(SEXP,SEXP);
SEXP R_fann_create(SEXP);
SEXP R_fann_version_info();

// SEXP R_fann_setopt(SEXP , SEXP , SEXP , SEXP , SEXP );


