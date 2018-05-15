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

#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/list/at.hpp>
#include <boost/preprocessor/list/for_each_product.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>

#define GET_D(L)   BOOST_PP_TUPLE_ELEM(3,0,BOOST_PP_TUPLE_ELEM(1,0,L))
#define GET_Q(L)   BOOST_PP_TUPLE_ELEM(3,1,BOOST_PP_TUPLE_ELEM(1,0,L))

#define MF_DQ  BOOST_PP_TUPLE_TO_LIST(4,(\
                                          (1,2),\
                                          (2,3),\
                                          (3,4),\
                                          (4,5)\
                                         ))

#define DOIF2(R, L) \
  else if ( (degree == GET_D(L)) && (n_q_points == GET_Q(L)) ) \
    { \
      Solid<2,GET_D(L),GET_Q(L),double> solid_2d(parameters); \
      solid_2d.run(); \
    } \


#define DOIF3(R, L) \
  else if ( (degree == GET_D(L)) && (n_q_points == GET_Q(L)) ) \
    { \
      Solid<3,GET_D(L),GET_Q(L),double> solid_3d(parameters); \
      solid_3d.run(); \
    } \


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
        const unsigned int degree = parameters.poly_degree;
        const unsigned int n_q_points = parameters.quad_order;
        if (parameters.dim == 2)
          {
            if (degree==0)
              {
                AssertThrow(false, ExcInternalError());
              }
            BOOST_PP_LIST_FOR_EACH_PRODUCT(DOIF2, 1, (MF_DQ))
            else
              {
                AssertThrow(false,
                            ExcMessage("Matrix-free calculations with degree="+
                                        std::to_string(degree)+
                                        " and n_q_points_1d="+
                                        std::to_string(n_q_points)+
                                        " are not supported."));
              }
          }
        else if (parameters.dim == 3)
          {
            if (degree==0)
              {
                AssertThrow(false, ExcInternalError());
              }
            BOOST_PP_LIST_FOR_EACH_PRODUCT(DOIF3, 1, (MF_DQ))
            else
              {
                AssertThrow(false,
                            ExcMessage("Matrix-free calculations with degree="+
                                        std::to_string(degree)+
                                        " and n_q_points_1d="+
                                        std::to_string(n_q_points)+
                                        " are not supported."));
              }
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
