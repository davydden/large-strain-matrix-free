/* ---------------------------------------------------------------------
 * Copyright (C) 2010 - 2015 by the deal.II authors and
 *                              Jean-Paul Pelteret and Andrew McBride
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 */

/*
 * Authors: Jean-Paul Pelteret, University of Erlangen-Nuremberg,
 *          Andrew McBride, University of Cape Town, 2015, 2017
 */

// own headers
#include <mf_elasticity.h>

// @sect3{Main function}
// Lastly we provide the main driver function which appears
// no different to the other tutorials.
int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace Cook_Membrane;

  try
    {
      deallog.depth_console(0);
      const std::string parameter_filename = argc > 1 ?
                                             argv[1] :
                                             "parameters.prm";
      Parameters::AllParameters parameters(parameter_filename);
      {
        std::cout << "Assembly method: Residual and linearisation are computed manually." << std::endl;

        // Allow multi-threading
        Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                            dealii::numbers::invalid_unsigned_int);

        typedef double NumberType;
        if (parameters.dim == 3)
          {
            Solid<3,1,2,NumberType> solid_3d(parameters);
            solid_3d.run();
          }
        else if (parameters.dim == 2)
          {
            Solid<2,1,2,NumberType> solid_2d(parameters);
            solid_2d.run();
          }
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
                << std::endl << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
