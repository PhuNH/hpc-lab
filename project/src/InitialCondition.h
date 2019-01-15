#ifndef INITIALCONDITION_H_
#define INITIALCONDITION_H_

#include "typedefs.h"
#include "Grid.h"

void initialCondition(  GlobalConstants const& globals,
						LocalConstants const& locals,
                        Grid<Material>& materialGrid,
                        Grid<DegreesOfFreedom>& degreesOfFreedomGrid  );

void L2error_squared( double time,
              GlobalConstants const& globals,
			  LocalConstants const& locals,
              Grid<Material>& materialGrid,
              Grid<DegreesOfFreedom>& degreesOfFreedomGrid,
              double l2error[NUMBER_OF_QUANTITIES]  );
			  
void square_root_array(double * array, int length);

void initSourcetermPhi(double xi, double eta, SourceTerm& sourceterm);

#endif // INITIALCONDITION_H_
