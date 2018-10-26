#pragma once

#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <deque>
#include <queue>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <random>


/*
 * Built-in functions
 *
 */

#define positive(x) (x>0.0? x : 0.0)
#define negative(x) (x<0.0? x : 0.0)
#define clip(x, a, b) (x<a? a : (x>b? b :x))
#define modulo(a, b) long(a) % long(b)
#define Equality(a, b) a == b
#define Eq(a, b) a == b
#define And(a, b) a && b
#define Or(a, b) a || b
#define Not(a) !a
#define Ne(a, b) a != b
#define ite(a, b, c) (a?b:c)
inline double power(double x, unsigned int a){
    double res=x;
    for(int i=0; i< a-1; i++){
        res *= x;
    }
    return res;
};


/*
 * Custom constants
 *
 */


/*
 * Custom functions
 *
 */


/*
 * Structures for the populations
 *
 */
#include "pop0.hpp"
#include "pop1.hpp"

/*
 * Structures for the projections
 *
 */
#include "proj0.hpp"



/*
 * Internal data
 *
*/
extern double dt;
extern long int t;
extern std::mt19937  rng;


/*
 * Declaration of the populations
 *
 */
extern PopStruct0 pop0;
extern PopStruct1 pop1;


/*
 * Declaration of the projections
 *
 */
extern ProjStruct0 proj0;


/*
 * Recorders
 *
 */
#include "Recorder.h"

extern std::vector<Monitor*> recorders;
void addRecorder(Monitor* recorder);
void removeRecorder(Monitor* recorder);

/*
 * Simulation methods
 *
*/

void initialize(double _dt, long int seed) ;

void run(int nbSteps);

int run_until(int steps, std::vector<int> populations, bool or_and);

void step();


/*
 * Time export
 *
*/
long int getTime() ;
void setTime(long int t_) ;

double getDt() ;
void setDt(double dt_);

/*
 * Number of threads
 *
*/
void setNumberThreads(int threads);

/*
 * Seed for the RNG
 *
*/
void setSeed(long int seed);

