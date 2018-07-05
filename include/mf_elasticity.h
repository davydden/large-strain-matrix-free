#pragma once

/**
 * level of output to deallog, can be:
 * 0 - none
 * 1 - norms of vectors
 * 2 - CG iterations
 * 3 - GMG iterations
 */
static const unsigned int debug_level = 0;

// We start by including all the necessary deal.II header files and some C++
// related ones. They have been discussed in detail in previous tutorial
// programs, so you need only refer to past tutorials for details.
#include <deal.II/base/config.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_point_data.h>
#include <deal.II/base/std_cxx11/shared_ptr.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_selector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// These must be included below the AD headers so that
// their math functions are available for use in the
// definition of tensors and kinematic quantities
#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <material.h>
#include <mf_ad_nh_operator.h>
#include <mf_nh_operator.h>

#include <fstream>
#include <iostream>

using namespace dealii;

template <typename Number>
void
copy_trilinos(LinearAlgebra::distributed::Vector<Number> &dst,
              const TrilinosWrappers::MPI::Vector &       trilinos_vec)
{
  // copy-paste operator= from LA::d::Vectro
  IndexSet combined_set = dst.get_partitioner()->locally_owned_range();
  combined_set.add_indices(dst.get_partitioner()->ghost_indices());
  LinearAlgebra::ReadWriteVector<Number> rw_vector(combined_set);
  rw_vector.import(trilinos_vec, VectorOperation::insert);
  dst.import(rw_vector, VectorOperation::insert);

  if (dst.has_ghost_elements() || trilinos_vec.has_ghost_elements())
    dst.update_ghost_values();
}

// We then stick everything that relates to this tutorial program into a
// namespace of its own, and import all the deal.II function and class names
// into it:
namespace Cook_Membrane
{
  using namespace dealii;

  // @sect3{Run-time parameters}
  //
  // There are several parameters that can be set in the code so we set up a
  // ParameterHandler object to read in the choices at run-time.
  namespace Parameters
  {
    // @sect4{Finite Element system}

    // Here we specify the polynomial order used to approximate the solution.
    // The quadrature order should be adjusted accordingly.
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };


    void
    FESystem::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        prm.declare_entry("Polynomial degree",
                          "2",
                          Patterns::Integer(0),
                          "Displacement system polynomial order");

        prm.declare_entry("Quadrature order",
                          "3",
                          Patterns::Integer(0),
                          "Gauss quadrature order");
      }
      prm.leave_subsection();
    }

    void
    FESystem::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Finite element system");
      {
        poly_degree = prm.get_integer("Polynomial degree");
        quad_order  = prm.get_integer("Quadrature order");
      }
      prm.leave_subsection();
    }

    // @sect4{Geometry}

    // Make adjustments to the problem geometry and its discretisation.
    struct Geometry
    {
      unsigned int elements_per_edge;
      double       scale;
      unsigned int dim;
      unsigned int n_global_refinement;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Geometry::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        prm.declare_entry("Elements per edge",
                          "32",
                          Patterns::Integer(0),
                          "Number of elements per long edge of the beam");

        prm.declare_entry("Global refinement",
                          "0",
                          Patterns::Integer(0),
                          "Number of global refinements");

        prm.declare_entry("Grid scale",
                          "1e-3",
                          Patterns::Double(0.0),
                          "Global grid scaling factor");

        prm.declare_entry("Dimension",
                          "2",
                          Patterns::Integer(2, 3),
                          "Dimension of the problem");
      }
      prm.leave_subsection();
    }

    void
    Geometry::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Geometry");
      {
        elements_per_edge   = prm.get_integer("Elements per edge");
        scale               = prm.get_double("Grid scale");
        dim                 = prm.get_integer("Dimension");
        n_global_refinement = prm.get_integer("Global refinement");
      }
      prm.leave_subsection();
    }

    // @sect4{Materials}

    // We also need the shear modulus $ \mu $ and Poisson ration $ \nu $ for the
    // neo-Hookean material.
    struct Materials
    {
      double       nu;
      double       mu;
      unsigned int material_formulation;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Materials::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        prm.declare_entry("Poisson's ratio",
                          "0.3",
                          Patterns::Double(-1.0, 0.5),
                          "Poisson's ratio");

        prm.declare_entry("Shear modulus",
                          "0.4225e6",
                          Patterns::Double(),
                          "Shear modulus");

        prm.declare_entry("Formulation",
                          "0",
                          Patterns::Integer(0, 1),
                          "Formulation of the energy function");
      }
      prm.leave_subsection();
    }

    void
    Materials::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Material properties");
      {
        nu                   = prm.get_double("Poisson's ratio");
        mu                   = prm.get_double("Shear modulus");
        material_formulation = prm.get_integer("Formulation");
      }
      prm.leave_subsection();
    }

    // @sect4{Linear solver}

    // Next, we choose both solver and preconditioner settings.  The use of an
    // effective preconditioner is critical to ensure convergence when a large
    // nonlinear motion occurs within a Newton increment.
    struct LinearSolver
    {
      std::string  type_lin;
      double       tol_lin;
      double       max_iterations_lin;
      std::string  preconditioner_type;
      double       preconditioner_relaxation;
      unsigned int cond_number_cg_iterations;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    LinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        prm.declare_entry("Solver type",
                          "CG",
                          Patterns::Selection("CG|Direct|MF_CG|MF_AD_CG"),
                          "Type of solver used to solve the linear system");

        prm.declare_entry("Residual",
                          "1e-6",
                          Patterns::Double(0.0),
                          "Linear solver residual (scaled by residual norm)");

        prm.declare_entry(
          "Max iteration multiplier",
          "1",
          Patterns::Double(0.0),
          "Linear solver iterations (multiples of the system matrix size)");

        prm.declare_entry("Preconditioner type",
                          "jacobi",
                          Patterns::Selection("jacobi|ssor|gmg|none"),
                          "Type of preconditioner");

        prm.declare_entry("Preconditioner relaxation",
                          "0.65",
                          Patterns::Double(0.0),
                          "Preconditioner relaxation value");

        prm.declare_entry(
          "Condition number CG iterations",
          "20",
          Patterns::Integer(1),
          "Number of CG iterations to estimate condition number");
      }
      prm.leave_subsection();
    }

    void
    LinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Linear solver");
      {
        type_lin                  = prm.get("Solver type");
        tol_lin                   = prm.get_double("Residual");
        max_iterations_lin        = prm.get_double("Max iteration multiplier");
        preconditioner_type       = prm.get("Preconditioner type");
        preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
        cond_number_cg_iterations =
          prm.get_integer("Condition number CG iterations");
      }
      prm.leave_subsection();
    }

    // @sect4{Nonlinear solver}

    // A Newton-Raphson scheme is used to solve the nonlinear system of
    // governing equations.  We now define the tolerances and the maximum number
    // of iterations for the Newton-Raphson nonlinear solver.
    struct NonlinearSolver
    {
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    NonlinearSolver::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        prm.declare_entry("Max iterations Newton-Raphson",
                          "10",
                          Patterns::Integer(0),
                          "Number of Newton-Raphson iterations allowed");

        prm.declare_entry("Tolerance force",
                          "1.0e-9",
                          Patterns::Double(0.0),
                          "Force residual tolerance");

        prm.declare_entry("Tolerance displacement",
                          "1.0e-6",
                          Patterns::Double(0.0),
                          "Displacement error tolerance");
      }
      prm.leave_subsection();
    }

    void
    NonlinearSolver::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Nonlinear solver");
      {
        max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
        tol_f             = prm.get_double("Tolerance force");
        tol_u             = prm.get_double("Tolerance displacement");
      }
      prm.leave_subsection();
    }

    // @sect4{Time}

    // Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct Time
    {
      double delta_t;
      double end_time;

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    void
    Time::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        prm.declare_entry("End time", "1", Patterns::Double(), "End time");

        prm.declare_entry("Time step size",
                          "0.1",
                          Patterns::Double(),
                          "Time step size");
      }
      prm.leave_subsection();
    }

    void
    Time::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Time");
      {
        end_time = prm.get_double("End time");
        delta_t  = prm.get_double("Time step size");
      }
      prm.leave_subsection();
    }

    // @sect4{All parameters}

    // Finally we consolidate all of the above structures into a single
    // container that holds all of our run-time selections.
    struct AllParameters : public FESystem,
                           public Geometry,
                           public Materials,
                           public LinearSolver,
                           public NonlinearSolver,
                           public Time

    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(ParameterHandler &prm);

      void
      parse_parameters(ParameterHandler &prm);
    };

    AllParameters::AllParameters(const std::string &input_file)
    {
      ParameterHandler prm;
      declare_parameters(prm);
      prm.parse_input(input_file);
      parse_parameters(prm);
    }

    void
    AllParameters::declare_parameters(ParameterHandler &prm)
    {
      FESystem::declare_parameters(prm);
      Geometry::declare_parameters(prm);
      Materials::declare_parameters(prm);
      LinearSolver::declare_parameters(prm);
      NonlinearSolver::declare_parameters(prm);
      Time::declare_parameters(prm);
    }

    void
    AllParameters::parse_parameters(ParameterHandler &prm)
    {
      FESystem::parse_parameters(prm);
      Geometry::parse_parameters(prm);
      Materials::parse_parameters(prm);
      LinearSolver::parse_parameters(prm);
      NonlinearSolver::parse_parameters(prm);
      Time::parse_parameters(prm);
    }
  } // namespace Parameters

  // @sect3{Time class}

  // A simple class to store time data. Its functioning is transparent so no
  // discussion is necessary. For simplicity we assume a constant time step
  // size.
  class Time
  {
  public:
    Time(const double time_end, const double delta_t)
      : timestep(0)
      , time_current(0.0)
      , time_end(time_end)
      , delta_t(delta_t)
    {}

    virtual ~Time()
    {}

    double
    current() const
    {
      return time_current;
    }
    double
    end() const
    {
      return time_end;
    }
    double
    get_delta_t() const
    {
      return delta_t;
    }
    unsigned int
    get_timestep() const
    {
      return timestep;
    }
    void
    increment()
    {
      time_current += delta_t;
      ++timestep;
    }

  private:
    unsigned int timestep;
    double       time_current;
    const double time_end;
    const double delta_t;
  };

  // @sect3{Compressible neo-Hookean material within a one-field formulation}

  // @sect3{Quasi-static compressible finite-strain solid}

  // The Solid class is the central class in that it represents the problem at
  // hand. It follows the usual scheme in that all it really has is a
  // constructor, destructor and a <code>run()</code> function that dispatches
  // all the work to private functions of this class:
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  class Solid
  {
  public:
    using LevelNumberType = float;
    using LevelVectorType = LinearAlgebra::distributed::Vector<LevelNumberType>;
    using VectorType      = LinearAlgebra::distributed::Vector<double>;

    Solid(const Parameters::AllParameters &parameters);

    virtual ~Solid();

    void
    run();

  private:
    // We start the collection of member functions with one that builds the
    // grid:
    void
    make_grid();

    // Set up the finite element system to be solved:
    void
    system_setup();

    // Set up matrix-free
    void
    setup_matrix_free(const int &it_nr);

    // Function to assemble the system matrix and right hand side vecotr.
    void
    assemble_system();

    // Apply Dirichlet boundary conditions on the displacement field
    void
    make_constraints(const int &it_nr);

    // Solve for the displacement using a Newton-Raphson method. We break this
    // function into the nonlinear loop and the function that solves the
    // linearized Newton-Raphson step:
    void
    solve_nonlinear_timestep();

    /**
     * solve linear system and return a tuple consisting of
     * number of iterations, residual and condition number estiamte.
     */
    std::tuple<unsigned int, double, double>
    solve_linear_system(
      LinearAlgebra::distributed::Vector<double> &newton_update,
      TrilinosWrappers::MPI::Vector &             newton_update_trilinos) const;

    // Set total solution based on the current values of solution_n and
    // solution_delta:
    void
    set_total_solution();

    void
    output_results() const;

    /**
     * MPI communicator
     */
    MPI_Comm mpi_communicator;

    /**
     * Terminal output on root MPI process
     */
    ConditionalOStream pcout;

    // Finally, some member variables that describe the current state: A
    // collection of the parameters used to describe the problem setup...
    const Parameters::AllParameters &parameters;

    // ...the volume of the reference and current configurations...
    double vol_reference;
    double vol_current;

    // ...and description of the geometry on which the problem is solved:
    parallel::distributed::Triangulation<dim> triangulation;

    // Also, keep track of the current time and the time spent evaluating
    // certain functions
    Time                time;
    std::ofstream       timer_output_file;
    ConditionalOStream  timer_out;
    mutable TimerOutput timer;
    std::ofstream       deallogfile;

    // A description of the finite-element system including the displacement
    // polynomial degree, the degree-of-freedom handler, number of DoFs per
    // cell and the extractor objects used to retrieve information from the
    // solution vectors:
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler_ref;
    const unsigned int               dofs_per_cell;
    const FEValuesExtractors::Vector u_fe;

    IndexSet locally_owned_dofs, locally_relevant_dofs;

    // homogeneous material
    std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim, NumberType>>
      material;
    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim,
                                               VectorizedArray<NumberType>>>
      material_vec;
    std::shared_ptr<
      Material_Compressible_Neo_Hook_One_Field<dim, VectorizedArray<float>>>
      material_level;

    static const unsigned int n_components      = dim;
    static const unsigned int first_u_component = 0;

    enum
    {
      u_dof = 0
    };

    // Rules for Gauss-quadrature on both the cell and faces. The number of
    // quadrature points on both cells and faces is recorded.
    const QGauss<dim>     qf_cell;
    const QGauss<dim - 1> qf_face;
    const unsigned int    n_q_points;
    const unsigned int    n_q_points_f;

    // Objects that store the converged solution and right-hand side vectors,
    // as well as the tangent matrix. There is a AffineConstraints object used
    // to keep track of constraints.
    AffineConstraints<double>      constraints;
    TrilinosWrappers::SparseMatrix tangent_matrix;
    TrilinosWrappers::MPI::Vector  system_rhs_trilinos;
    TrilinosWrappers::MPI::Vector  newton_update_trilinos;

    LinearAlgebra::distributed::Vector<double> system_rhs;

    // solution at the previous time-step
    LinearAlgebra::distributed::Vector<double> solution_n;

    // current value of increment solution
    LinearAlgebra::distributed::Vector<double> solution_delta;

    // current total solution:  solution_tota = solution_n + solution_delta
    LinearAlgebra::distributed::Vector<double> solution_total;

    LinearAlgebra::distributed::Vector<double> newton_update;


    MGLevelObject<LevelVectorType> mg_solution_total;


    // Then define a number of variables to store norms and update norms and
    // normalisation factors.
    struct Errors
    {
      Errors()
        : norm(1.0)
        , u(1.0)
      {}

      void
      reset()
      {
        norm = 1.0;
        u    = 1.0;
      }
      void
      normalise(const Errors &rhs)
      {
        if (rhs.norm != 0.0)
          norm /= rhs.norm;
        if (rhs.u != 0.0)
          u /= rhs.u;
      }

      double norm, u;
    };

    mutable Errors error_residual, error_residual_0, error_residual_norm,
      error_update, error_update_0, error_update_norm;

    // Print information to screen in a pleasing way...
    void
    print_conv_header();

    void
    print_conv_footer();

    void
    print_vertical_tip_displacement();

    std::shared_ptr<
      MappingQEulerian<dim, LinearAlgebra::distributed::Vector<double>>>
                                             eulerian_mapping;
    std::shared_ptr<MatrixFree<dim, double>> mf_data_current;
    std::shared_ptr<MatrixFree<dim, double>> mf_data_reference;


    std::vector<std::shared_ptr<MappingQEulerian<dim, LevelVectorType>>>
                                                         mg_eulerian_mapping;
    std::vector<std::shared_ptr<MatrixFree<dim, float>>> mg_mf_data_current;
    std::vector<std::shared_ptr<MatrixFree<dim, float>>> mg_mf_data_reference;

    NeoHookOperator<dim, degree, n_q_points_1d, double>   mf_nh_operator;
    NeoHookOperatorAD<dim, degree, n_q_points_1d, double> mf_ad_nh_operator;

    typedef NeoHookOperator<dim, degree, n_q_points_1d, float> LevelMatrixType;

    MGLevelObject<LevelMatrixType> mg_mf_nh_operator;

    std::shared_ptr<MGTransferMatrixFree<dim, float>> mg_transfer;

    typedef PreconditionChebyshev<LevelMatrixType, LevelVectorType>
      SmootherChebyshev;

    // MGSmootherPrecondition<LevelMatrixType, SmootherChebyshev,
    // LevelVectorType> mg_smoother_chebyshev;
    mg::SmootherRelaxation<SmootherChebyshev, LevelVectorType>
      mg_smoother_chebyshev;

    MGCoarseGridApplySmoother<LevelVectorType> mg_coarse_chebyshev;

    std::shared_ptr<SolverControl>             coarse_solver_control;
    std::shared_ptr<SolverCG<LevelVectorType>> coarse_solver;

    MGCoarseGridIterativeSolver<LevelVectorType,
                                SolverCG<LevelVectorType>,
                                LevelMatrixType,
                                SmootherChebyshev>
      mg_coarse_iterative;

    mg::Matrix<LevelVectorType> mg_operator_wrapper;

    std::shared_ptr<Multigrid<LevelVectorType>> multigrid;

    std::shared_ptr<
      PreconditionMG<dim, LevelVectorType, MGTransferMatrixFree<dim, float>>>
      multigrid_preconditioner;

    MGConstrainedDoFs mg_constrained_dofs;
  };

  // @sect3{Implementation of the <code>Solid</code> class}

  // @sect4{Public interface}

  // We initialise the Solid class using data extracted from the parameter file.
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  Solid<dim, degree, n_q_points_1d, NumberType>::Solid(
    const Parameters::AllParameters &parameters)
    : mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , parameters(parameters)
    , vol_reference(0.0)
    , vol_current(0.0)
    , triangulation(
        mpi_communicator,
        typename Triangulation<dim>::MeshSmoothing(
          // guarantee that the mesh also does not change by more than
          // refinement level across vertices that might connect two cells:
          Triangulation<dim>::limit_level_difference_at_vertices),
        typename parallel::distributed::Triangulation<dim>::Settings(
          // needed for GMG:
          parallel::distributed::Triangulation<
            dim>::construct_multigrid_hierarchy))
    , time(parameters.end_time, parameters.delta_t)
    , timer_out(timer_output_file,
                Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    , timer(mpi_communicator,
            timer_out,
            TimerOutput::summary,
            TimerOutput::wall_times)
    ,
    // The Finite Element System is composed of dim continuous displacement
    // DOFs.
    fe(FE_Q<dim>(degree), dim)
    , // displacement
    dof_handler_ref(triangulation)
    , dofs_per_cell(fe.dofs_per_cell)
    , u_fe(first_u_component)
    , material(std::make_shared<
               Material_Compressible_Neo_Hook_One_Field<dim, NumberType>>(
        parameters.mu,
        parameters.nu,
        parameters.material_formulation))
    , material_vec(
        std::make_shared<Material_Compressible_Neo_Hook_One_Field<
          dim,
          VectorizedArray<NumberType>>>(parameters.mu,
                                        parameters.nu,
                                        parameters.material_formulation))
    , material_level(
        std::make_shared<
          Material_Compressible_Neo_Hook_One_Field<dim,
                                                   VectorizedArray<float>>>(
          parameters.mu,
          parameters.nu,
          parameters.material_formulation))
    , qf_cell(n_q_points_1d)
    , qf_face(n_q_points_1d)
    , n_q_points(qf_cell.size())
    , n_q_points_f(qf_face.size())
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        deallogfile.open("deallog.txt");
        deallog.attach(deallogfile);

        timer_output_file.open("timings.txt");
      }

    mf_nh_operator.set_material(material_vec);
    mf_ad_nh_operator.set_material(material_vec);
  }

  // The class destructor simply clears the data held by the DOFHandler
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  Solid<dim, degree, n_q_points_1d, NumberType>::~Solid()
  {
    mf_nh_operator.clear();
    mf_ad_nh_operator.clear();

    mf_data_current.reset();
    mf_data_reference.reset();
    eulerian_mapping.reset();

    dof_handler_ref.clear();

    multigrid_preconditioner.reset();
    multigrid.reset();
    mg_coarse_chebyshev.clear();
    mg_smoother_chebyshev.clear();
    mg_operator_wrapper.reset();
    mg_mf_nh_operator.clear_elements();
    mg_transfer.reset();
  }


  // In solving the quasi-static problem, the time becomes a loading parameter,
  // i.e. we increasing the loading linearly with time, making the two concepts
  // interchangeable. We choose to increment time linearly using a constant time
  // step size.
  //
  // We start the function with preprocessing, and then output the initial grid
  // before starting the simulation proper with the first time (and loading)
  // increment.
  //
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::run()
  {
    make_grid();
    system_setup();
    output_results();
    time.increment();

    // We then declare the incremental solution update $\varDelta
    // \mathbf{\Xi}:= \{\varDelta \mathbf{u}\}$ and start the loop over the
    // time domain.
    //
    // At the beginning, we reset the solution update for this time step...
    while (time.current() <= time.end())
      {
        solution_delta = 0.0;

        // ...solve the current time step and update total solution vector
        // $\mathbf{\Xi}_{\textrm{n}} = \mathbf{\Xi}_{\textrm{n-1}} +
        // \varDelta \mathbf{\Xi}$...
        solve_nonlinear_timestep();
        solution_n += solution_delta;

        // ...and plot the results before moving on happily to the next time
        // step:
        output_results();
        time.increment();
      }

    // Lastly, we print the vertical tip displacement of the Cook cantilever
    // after the full load is applied
    print_vertical_tip_displacement();
  }


  // @sect3{Private interface}

  // @sect4{Solid::make_grid}

  // On to the first of the private member functions. Here we create the
  // triangulation of the domain, for which we choose a scaled an
  // anisotripically discretised rectangle which is subsequently transformed
  // into the correct of the Cook cantilever. Each relevant boundary face is
  // then given a boundary ID number.
  //
  // We then determine the volume of the reference configuration and print it
  // for comparison.

  template <int dim>
  Point<dim>
  grid_y_transform(const Point<dim> &pt_in)
  {
    const double &x = pt_in[0];
    const double &y = pt_in[1];

    const double y_upper =
      44.0 + (16.0 / 48.0) * x; // Line defining upper edge of beam
    const double y_lower =
      0.0 + (44.0 / 48.0) * x;     // Line defining lower edge of beam
    const double theta = y / 44.0; // Fraction of height along left side of beam
    const double y_transform =
      (1 - theta) * y_lower + theta * y_upper; // Final transformation

    Point<dim> pt_out = pt_in;
    pt_out[1]         = y_transform;

    return pt_out;
  }

  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::make_grid()
  {
    // Divide the beam, but only along the x- and y-coordinate directions
    std::vector<unsigned int> repetitions(dim, parameters.elements_per_edge);
    // Only allow one element through the thickness
    // (modelling a plane strain condition)
    if (dim == 3)
      repetitions[dim - 1] = 1;

    const Point<dim> bottom_left =
      (dim == 3 ? Point<dim>(0.0, 0.0, -0.5) : Point<dim>(0.0, 0.0));
    const Point<dim> top_right =
      (dim == 3 ? Point<dim>(48.0, 44.0, 0.5) : Point<dim>(48.0, 44.0));

    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              bottom_left,
                                              top_right);

    // Since we wish to apply a Neumann BC to the right-hand surface, we
    // must find the cell faces in this part of the domain and mark them with
    // a distinct boundary ID number.  The faces we are looking for are on the
    // +x surface and will get boundary ID 11.
    // Dirichlet boundaries exist on the left-hand face of the beam (this fixed
    // boundary will get ID 1) and on the +Z and -Z faces (which correspond to
    // ID 2 and we will use to impose the plane strain condition)
    const double tol_boundary = 1e-6;
    // NOTE: we need to set IDs regardless of cell being locally owned or not
    // as in the global refinement cells will be repartitioned and faces of
    // their parents should have right IDs
    for (auto cell : triangulation.active_cell_iterators())
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell->face(face)->at_boundary() == true)
          {
            if (std::abs(cell->face(face)->center()[0] - 0.0) < tol_boundary)
              cell->face(face)->set_boundary_id(1); // -X faces
            else if (std::abs(cell->face(face)->center()[0] - 48.0) <
                     tol_boundary)
              cell->face(face)->set_boundary_id(11); // +X faces
            else if (std::abs(std::abs(cell->face(face)->center()[0]) - 0.5) <
                     tol_boundary)
              cell->face(face)->set_boundary_id(2); // +Z and -Z faces
          }

    // Transform the hyper-rectangle into the beam shape
    GridTools::transform(&grid_y_transform<dim>, triangulation);

    GridTools::scale(parameters.scale, triangulation);

    triangulation.refine_global(parameters.n_global_refinement);

    vol_reference = GridTools::volume(triangulation);
    vol_current   = vol_reference;
    pcout << "Grid:\n\t Reference volume: " << vol_reference << std::endl;
  }


  // @sect4{Solid::system_setup}

  // Next we describe how the FE system is setup.  We first determine the number
  // of components per block. Since the displacement is a vector component, the
  // first dim components belong to it.
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::system_setup()
  {
    timer.enter_subsection("Setup system");

    // The DOF handler is then initialised and we renumber the grid in an
    // efficient manner. We also record the number of DOFs per block.
    dof_handler_ref.distribute_dofs(fe);
    dof_handler_ref.distribute_mg_dofs();
    DoFRenumbering::Cuthill_McKee(dof_handler_ref);

    pcout << "Triangulation:"
          << "\n\t Number of active cells: "
          << triangulation.n_global_active_cells()
          << "\n\t Number of degrees of freedom: " << dof_handler_ref.n_dofs()
          << std::endl;

    locally_owned_dofs = dof_handler_ref.locally_owned_dofs();
    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler_ref,
                                            locally_relevant_dofs);

    tangent_matrix.clear();
    {
      // Setup the sparsity pattern and tangent matrix
      TrilinosWrappers::SparsityPattern sp(locally_owned_dofs,
                                           mpi_communicator);

      DoFTools::make_sparsity_pattern(dof_handler_ref,
                                      sp,
                                      constraints,
                                      /* keep_constrained_dofs */ false,
                                      Utilities::MPI::this_mpi_process(
                                        mpi_communicator));

      sp.compress();
      tangent_matrix.reinit(sp);
    }

    // We then set up storage vectors
    system_rhs.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    solution_n.reinit(locally_owned_dofs,
                      locally_relevant_dofs,
                      mpi_communicator);
    solution_delta.reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          mpi_communicator);
    solution_total.reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          mpi_communicator);
    newton_update.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);

    newton_update_trilinos.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs_trilinos.reinit(locally_owned_dofs, mpi_communicator);

    // switch to ghost mode:
    solution_n.update_ghost_values();
    solution_delta.update_ghost_values();
    solution_total.update_ghost_values();

    timer.leave_subsection();
  }



  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::setup_matrix_free(
    const int &it_nr)
  {
    timer.enter_subsection("Setup matrix-free");

    const unsigned int max_level = triangulation.n_global_levels() - 1;

    mg_coarse_iterative.clear();
    coarse_solver.reset();

    if (it_nr <= 1)
      {
        // GMG main classes
        multigrid_preconditioner.reset();
        multigrid.reset();
        // clear all pointers to level matrices in smoothers:
        mg_coarse_chebyshev.clear();
        mg_smoother_chebyshev.clear();
        // reset wrappers before resizing matrices
        mg_operator_wrapper.reset();
        // and clean up transfer which is also initialized with mg_matrices:
        mg_transfer.reset();
        // Now we can reset mg_matrices
        mg_mf_nh_operator.resize(0, max_level);
        mg_eulerian_mapping.clear();

        mg_solution_total.resize(0, max_level);
      }

    // The constraints in Newton-Raphson are different for it_nr=0 and 1,
    // and then they are the same so we only need to re-init the data
    // according to the updated displacement/mapping
    const QGauss<1> quad(n_q_points_1d);

    typename MatrixFree<dim, double>::AdditionalData data;
    data.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
    data.mapping_update_flags  = update_gradients | update_JxW_values;

    typename MatrixFree<dim, float>::AdditionalData mg_additional_data;
    mg_additional_data.tasks_parallel_scheme =
      MatrixFree<dim, float>::AdditionalData::none; // partition_color;
    mg_additional_data.mapping_update_flags =
      update_gradients | update_JxW_values;

    if (it_nr <= 1)
      {
        mg_transfer =
          std::make_shared<MGTransferMatrixFree<dim, LevelNumberType>>(
            mg_constrained_dofs);
        mg_transfer->build(dof_handler_ref);

        mg_mf_data_current.resize(triangulation.n_global_levels());
        mg_mf_data_reference.resize(triangulation.n_global_levels());
      }

    // transfer displacement to MG levels:
    LinearAlgebra::distributed::Vector<LevelNumberType> solution_total_transfer;
    solution_total_transfer.reinit(solution_total);
    solution_total_transfer = solution_total;
    mg_transfer->interpolate_to_mg(dof_handler_ref,
                                   mg_solution_total,
                                   solution_total_transfer);

    if (it_nr <= 1)
      {
        // solution_total is the point around which we linearize
        eulerian_mapping = std::make_shared<
          MappingQEulerian<dim, LinearAlgebra::distributed::Vector<double>>>(
          degree, dof_handler_ref, solution_total);

        mf_data_current   = std::make_shared<MatrixFree<dim, double>>();
        mf_data_reference = std::make_shared<MatrixFree<dim, double>>();

        mf_data_reference->reinit(dof_handler_ref, constraints, quad, data);
        mf_data_current->reinit(
          *eulerian_mapping, dof_handler_ref, constraints, quad, data);

        mf_nh_operator.initialize(mf_data_current,
                                  mf_data_reference,
                                  solution_total);
        mf_ad_nh_operator.initialize(mf_data_current,
                                     mf_data_reference,
                                     solution_total);

        mg_eulerian_mapping.resize(0);
        for (unsigned int level = 0; level <= max_level; ++level)
          {
            mg_additional_data.level_mg_handler = level;

            AffineConstraints<double> level_constraints;
            IndexSet                  relevant_dofs;
            DoFTools::extract_locally_relevant_level_dofs(dof_handler_ref,
                                                          level,
                                                          relevant_dofs);
            level_constraints.reinit(relevant_dofs);
            level_constraints.add_lines(
              mg_constrained_dofs.get_boundary_indices(level));
            level_constraints.close();

            // GMG MF operators do not support edge indices yet
            Assert(
              mg_constrained_dofs.get_refinement_edge_indices(level).is_empty(),
              ExcNotImplemented());

            mg_mf_data_current[level] =
              std::make_shared<MatrixFree<dim, float>>();
            mg_mf_data_reference[level] =
              std::make_shared<MatrixFree<dim, float>>();

            std::shared_ptr<MappingQEulerian<dim, LevelVectorType>>
              euler_level =
                std::make_shared<MappingQEulerian<dim, LevelVectorType>>(
                  degree, dof_handler_ref, mg_solution_total[level], level);

            mg_mf_data_reference[level]->reinit(dof_handler_ref,
                                                level_constraints,
                                                quad,
                                                mg_additional_data);
            mg_mf_data_current[level]->reinit(*euler_level,
                                              dof_handler_ref,
                                              level_constraints,
                                              quad,
                                              mg_additional_data);

            mg_eulerian_mapping.push_back(euler_level);

            mg_mf_nh_operator[level].initialize(
              mg_mf_data_current[level],
              mg_mf_data_reference[level],
              mg_solution_total[level]); // (mg_level_data, mg_constrained_dofs,
                                         // level);
          }
      }
    else
      {
        // here reinitialize MatrixFree with initialize_indices=false
        // as the mapping has to be recomputed but the topology of cells is the
        // same
        data.initialize_indices = false;
        mf_data_current->reinit(
          *eulerian_mapping, dof_handler_ref, constraints, quad, data);

        mg_additional_data.initialize_indices = false;
        for (unsigned int level = 0; level <= max_level; ++level)
          {
            AffineConstraints<double> level_constraints;
            IndexSet                  relevant_dofs;
            DoFTools::extract_locally_relevant_level_dofs(dof_handler_ref,
                                                          level,
                                                          relevant_dofs);
            level_constraints.reinit(relevant_dofs);
            level_constraints.add_lines(
              mg_constrained_dofs.get_boundary_indices(level));
            level_constraints.close();

            mg_additional_data.level_mg_handler = level;
            std::shared_ptr<MappingQEulerian<dim, LevelVectorType>>
              euler_level =
                std::make_shared<MappingQEulerian<dim, LevelVectorType>>(
                  degree, dof_handler_ref, mg_solution_total[level], level);
            mg_mf_data_current[level]->reinit(*euler_level,
                                              dof_handler_ref,
                                              level_constraints,
                                              quad,
                                              mg_additional_data);
          }
      }

    // adjust ghost range if needed
    {
      const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner =
        mf_data_current->get_vector_partitioner();

      Assert(partitioner->is_globally_compatible(
               *mf_data_reference->get_vector_partitioner().get()),
             ExcInternalError());

      adjust_ghost_range_if_necessary(partitioner, newton_update);
      adjust_ghost_range_if_necessary(partitioner, system_rhs);

      adjust_ghost_range_if_necessary(partitioner, solution_total);
      solution_total.update_ghost_values();
    }

    // FIXME: interpolate_to_mg will resize MG vector, make sure it has the
    // right partition for MF
    for (unsigned int level = 0; level <= max_level; ++level)
      {
        const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner =
          mg_mf_data_current[level]->get_vector_partitioner();

        Assert(partitioner->is_globally_compatible(
                 *mg_mf_data_reference[level]->get_vector_partitioner().get()),
               ExcInternalError());

        adjust_ghost_range_if_necessary(partitioner, mg_solution_total[level]);
        mg_solution_total[level].update_ghost_values();
      }

    // need to cache prior to diagonal computations:
    mf_nh_operator.cache();
    mf_nh_operator.compute_diagonal();
    if (debug_level > 0)
      {
        deallog << "GMG setup Newton iteration " << it_nr << std::endl;
        deallog
          << "Number of constrained DoFs "
          << Utilities::MPI::sum(mf_data_current->get_constrained_dofs().size(),
                                 mpi_communicator)
          << std::endl;
        deallog << "Diagonal:       "
                << mf_nh_operator.get_matrix_diagonal_inverse()
                     ->get_vector()
                     .l2_norm()
                << std::endl;
        deallog << "Solution total: " << solution_total.l2_norm() << std::endl;
      }
    if (parameters.type_lin == "MF_AD_CG")
      {
        mf_ad_nh_operator.cache();
        mf_ad_nh_operator.compute_diagonal();
      }

    for (unsigned int level = 0; level <= max_level; ++level)
      {
        mg_mf_nh_operator[level].set_material(material_level);
        mg_mf_nh_operator[level].cache();
        mg_mf_nh_operator[level].compute_diagonal();

        if (debug_level > 0)
          {
            deallog << "Diagonal on level       " << level << ": "
                    << mg_mf_nh_operator[level]
                         .get_matrix_diagonal_inverse()
                         ->get_vector()
                         .l2_norm()
                    << std::endl;
            deallog << "Solution total on level " << level << ": "
                    << mg_solution_total[level].l2_norm() << std::endl;
          }
      }

    // setup GMG preconditioner
    const bool cheb_coarse = true;
    {
      MGLevelObject<typename SmootherChebyshev::AdditionalData> smoother_data;
      smoother_data.resize(0, triangulation.n_global_levels() - 1);
      for (unsigned int level = 0; level < triangulation.n_global_levels();
           ++level)
        {
          if (cheb_coarse && level == 0)
            {
              smoother_data[level].smoothing_range =
                1e-3; // reduce residual by this relative tolerance
              smoother_data[level].degree =
                numbers::invalid_unsigned_int; // use as a solver
              smoother_data[level].eig_cg_n_iterations =
                mg_mf_nh_operator[level].m();
            }
          else
            {
              // [1.2 \lambda_{max}/range, 1.2 \lambda_{max}]
              smoother_data[level].smoothing_range = 2;
              // With degree zero, the Jacobi method with optimal damping
              // parameter is retrieved
              smoother_data[level].degree = 4;
              // number of CG iterataions to estimate the largest eigenvalue:
              smoother_data[level].eig_cg_n_iterations = 30;
            }
          smoother_data[level].preconditioner =
            mg_mf_nh_operator[level].get_matrix_diagonal_inverse();
        }
      mg_smoother_chebyshev.initialize(mg_mf_nh_operator, smoother_data);
      mg_coarse_chebyshev.initialize(mg_smoother_chebyshev);
    }

    coarse_solver_control =
      std::make_shared<ReductionControl>(std::max(mg_mf_nh_operator[0].m(),
                                                  (unsigned int)100),
                                         1e-10,
                                         1e-3,
                                         false,
                                         false);
    coarse_solver =
      std::make_shared<SolverCG<LevelVectorType>>(*coarse_solver_control);
    mg_coarse_iterative.initialize(*coarse_solver,
                                   mg_mf_nh_operator[0],
                                   mg_smoother_chebyshev[0]);

    // wrap our level and interface matrices in an object having the required
    // multiplication functions.
    mg_operator_wrapper.initialize(mg_mf_nh_operator);

    multigrid_preconditioner.reset();
    if (cheb_coarse)
      multigrid =
        std::make_shared<Multigrid<LevelVectorType>>(mg_operator_wrapper,
                                                     mg_coarse_chebyshev,
                                                     *mg_transfer,
                                                     mg_smoother_chebyshev,
                                                     mg_smoother_chebyshev,
                                                     /*min_level*/ 0);
    else
      multigrid =
        std::make_shared<Multigrid<LevelVectorType>>(mg_operator_wrapper,
                                                     mg_coarse_iterative,
                                                     *mg_transfer,
                                                     mg_smoother_chebyshev,
                                                     mg_smoother_chebyshev,
                                                     /*min_level*/ 0);

    // set_debug() is deprecated, keep this commented for now
    // if (debug_level > 2)
    //   multigrid->set_debug(5);


    multigrid->connect_coarse_solve(
      [&](const bool start, const unsigned int level) {
        if (start)
          timer.enter_subsection("Coarse solve level " +
                                 Utilities::int_to_string(level));
        else
          timer.leave_subsection();
      });

    multigrid->connect_restriction(
      [&](const bool start, const unsigned int level) {
        if (start)
          timer.enter_subsection("Coarse solve level " +
                                 Utilities::int_to_string(level));
        else
          timer.leave_subsection();
      });
    multigrid->connect_prolongation(
      [&](const bool start, const unsigned int level) {
        if (start)
          timer.enter_subsection("Prolongation level " +
                                 Utilities::int_to_string(level));
        else
          timer.leave_subsection();
      });
    multigrid->connect_pre_smoother_step(
      [&](const bool start, const unsigned int level) {
        if (start)
          timer.enter_subsection("Pre-smoothing level " +
                                 Utilities::int_to_string(level));
        else
          timer.leave_subsection();
      });

    multigrid->connect_post_smoother_step(
      [&](const bool start, const unsigned int level) {
        if (start)
          timer.enter_subsection("Post-smoothing level " +
                                 Utilities::int_to_string(level));
        else
          timer.leave_subsection();
      });

    // and a preconditioner object which uses GMG
    multigrid_preconditioner = std::make_shared<
      PreconditionMG<dim, LevelVectorType, MGTransferMatrixFree<dim, float>>>(
      dof_handler_ref, *multigrid, *mg_transfer);

    timer.leave_subsection();
  }

  // @sect4{Solid::solve_nonlinear_timestep}

  // The next function is the driver method for the Newton-Raphson scheme. At
  // its top we create a new vector to store the current Newton update step,
  // reset the error storage objects and print solver header.
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::solve_nonlinear_timestep()
  {
    pcout << std::endl
          << "Timestep " << time.get_timestep() << " @ " << time.current()
          << "s" << std::endl;

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();
    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();

    // We now perform a number of Newton iterations to iteratively solve the
    // nonlinear problem.  Since the problem is fully nonlinear and we are
    // using a full Newton method, the data stored in the tangent matrix and
    // right-hand side vector is not reusable and must be cleared at each
    // Newton step.  We then initially build the right-hand side vector to
    // check for convergence (and store this value in the first iteration).
    // The unconstrained DOFs of the rhs vector hold the out-of-balance
    // forces. The building is done before assembling the system matrix as the
    // latter is an expensive operation and we can potentially avoid an extra
    // assembly process by not assembling the tangent matrix when convergence
    // is attained.
    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR; ++newton_iteration)
      {
        pcout << " " << std::setw(2) << newton_iteration << " " << std::flush;

        // If we have decided that we want to continue with the iteration, we
        // assemble the tangent, make and impose the Dirichlet constraints,
        // and do the solve of the linearized system:
        make_constraints(newton_iteration);

        // update total solution prior to assembly
        set_total_solution();

        // setup matrix-free part:
        setup_matrix_free(newton_iteration);

        // now ready to go-on and assmble linearized problem around solution_n +
        // solution_delta for this iteration.
        assemble_system();

#ifdef DEBUG
        // check vmult of matrix-based and matrix-free for a random vector:
        {
          TrilinosWrappers::MPI::Vector src_trilinos(newton_update_trilinos),
            dst_mb(newton_update_trilinos);
          for (unsigned int i = 0; i < locally_owned_dofs.n_elements(); ++i)
            src_trilinos(locally_owned_dofs.nth_index_in_set(i)) =
              ((double)std::rand()) / RAND_MAX;

          constraints.set_zero(src_trilinos);

          LinearAlgebra::distributed::Vector src(newton_update),
            dst_mf(newton_update), diff(newton_update);
          copy_trilinos(src, src_trilinos);

          tangent_matrix.vmult(dst_mb, src_trilinos);
          mf_nh_operator.vmult(dst_mf, src);

          copy_trilinos(diff, dst_mb);
          diff.add(-1, dst_mf);

          // FIXME: looks like there are some severe round-off errors.
          const unsigned int ulp = std::pow(10, 9);

          for (unsigned int i = 0; i < diff.local_size(); ++i)
            Assert(std::abs(diff.local_element(i)) <=
                     std::numeric_limits<double>::epsilon() * ulp *
                       std::abs(dst_mf.local_element(i)),
                   ExcMessage("MF and MB are different on local element " +
                              std::to_string(i) + ": " +
                              std::to_string(dst_mf.local_element(i)) +
                              " diff " + std::to_string(diff.local_element(i)) +
                              " at Newton iteration " +
                              std::to_string(newton_iteration)));

          // now check Jacobi preconditioner
          TrilinosWrappers::PreconditionJacobi trilinos_jacobi;
          trilinos_jacobi.initialize(tangent_matrix, 0.8);
          trilinos_jacobi.vmult(dst_mb, src_trilinos);
          mf_nh_operator.precondition_Jacobi(dst_mf, src, 0.8);

          copy_trilinos(diff, dst_mb);
          diff.add(-1, dst_mf);
          for (unsigned int i = 0; i < diff.local_size(); ++i)
            Assert(std::abs(diff.local_element(i)) <=
                     10000. * std::numeric_limits<double>::epsilon() *
                       std::abs(dst_mf.local_element(i)),
                   ExcMessage(
                     "MF and MB Jacobi are different on local element " +
                     std::to_string(i) + ": " +
                     std::to_string(dst_mf.local_element(i)) + " diff " +
                     std::to_string(diff.local_element(i)) +
                     " at Newton iteration " +
                     std::to_string(newton_iteration)));
        }
#endif

        if (newton_iteration == 0)
          error_residual_0 = error_residual;

        // We can now determine the normalised residual error and check for
        // solution convergence:
        error_residual_norm = error_residual;
        error_residual_norm.normalise(error_residual_0);

        if (newton_iteration > 0 && error_update_norm.u <= parameters.tol_u &&
            error_residual_norm.u <= parameters.tol_f)
          {
            pcout << " CONVERGED! " << std::endl;
            print_conv_footer();

            break;
          }

        const std::tuple<unsigned int, double, double> lin_solver_output =
          solve_linear_system(newton_update, newton_update_trilinos);

        if (newton_iteration == 0)
          error_update_0 = error_update;

        // We can now determine the normalised Newton update error, and
        // perform the actual update of the solution increment for the current
        // time step, update all quadrature point information pertaining to
        // this new displacement and stress state and continue iterating:
        error_update_norm = error_update;
        error_update_norm.normalise(error_update_0);

        solution_delta += newton_update;

        pcout << " | " << std::fixed << std::setprecision(3) << std::setw(7)
              << std::scientific << std::get<0>(lin_solver_output) << "  "
              << std::get<1>(lin_solver_output) << "  "
              << std::get<2>(lin_solver_output) << "  "
              << error_residual_norm.norm << "  " << error_residual_norm.u
              << "  "
              << "  " << error_update_norm.norm << "  " << error_update_norm.u
              << "  " << std::endl;
      }

    // At the end, if it turns out that we have in fact done more iterations
    // than the parameter file allowed, we raise an exception that can be
    // caught in the main() function. The call <code>AssertThrow(condition,
    // exc_object)</code> is in essence equivalent to <code>if (!cond) throw
    // exc_object;</code> but the former form fills certain fields in the
    // exception object that identify the location (filename and line number)
    // where the exception was raised to make it simpler to identify where the
    // problem happened.
    AssertThrow(newton_iteration <= parameters.max_iterations_NR,
                ExcMessage("No convergence in nonlinear solver!"));
  }


  // @sect4{Solid::print_conv_header, Solid::print_conv_footer and
  // Solid::print_vertical_tip_displacement}

  // This program prints out data in a nice table that is updated
  // on a per-iteration basis. The next two functions set up the table
  // header and footer:
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::print_conv_header()
  {
    static const unsigned int l_width = 98;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << "_";
    pcout << std::endl;

    pcout << "    SOLVER STEP    "
          << " |  LIN_IT    LIN_RES   COND_NUM   RES_NORM   "
          << "   RES_U      NU_NORM  "
          << "     NU_U" << std::endl;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << "_";
    pcout << std::endl;
  }



  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::print_conv_footer()
  {
    static const unsigned int l_width = 98;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << "_";
    pcout << std::endl;

    pcout << "Relative errors:" << std::endl
          << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
          << "Force: \t\t" << error_residual.u / error_residual_0.u << std::endl
          << "v / V_0:\t" << vol_current << " / " << vol_reference << std::endl;
  }

  // At the end we also output the result that can be compared to that found in
  // the literature, namely the displacement at the upper right corner of the
  // beam.
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::
    print_vertical_tip_displacement()
  {
    static const unsigned int l_width = 87;

    for (unsigned int i = 0; i < l_width; ++i)
      pcout << "_";
    pcout << std::endl;

    Point<dim> soln_pt;
    soln_pt[0] = 48.0 * parameters.scale;
    soln_pt[1] = 60.0 * parameters.scale;
    if (dim == 3)
      soln_pt[2] = 0.5 * parameters.scale;
    double vertical_tip_displacement       = 0.0;
    double vertical_tip_displacement_check = 0.0;

    for (const auto &cell : dof_handler_ref.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
               ++v)
            if (cell->vertex(v).distance(soln_pt) < 1e-6)
              {
                // Extract y-component of solution at the given point
                // This point is coindicent with a vertex, so we can
                // extract it directly as we're using FE_Q finite elements
                // that have support at the vertices
                vertical_tip_displacement =
                  solution_n(cell->vertex_dof_index(v, u_dof + 1));

                // Sanity check using alternate method to extract the solution
                // at the given point. To do this, we must create an FEValues
                // instance to help us extract the solution value at the desired
                // point
                const MappingQ<dim> mapping(degree);
                const Point<dim>    qp_unit =
                  mapping.transform_real_to_unit_cell(cell, soln_pt);
                const Quadrature<dim> soln_qrule(qp_unit);
                AssertThrow(soln_qrule.size() == 1, ExcInternalError());
                FEValues<dim> fe_values_soln(fe, soln_qrule, update_values);
                fe_values_soln.reinit(cell);

                // Extract y-component of solution at given point
                std::vector<Tensor<1, dim>> soln_values(soln_qrule.size());
                fe_values_soln[u_fe].get_function_values(solution_n,
                                                         soln_values);
                vertical_tip_displacement_check = soln_values[0][u_dof + 1];

                break;
              }
        }
    vertical_tip_displacement =
      Utilities::MPI::max(vertical_tip_displacement, mpi_communicator);
    vertical_tip_displacement_check =
      Utilities::MPI::max(vertical_tip_displacement_check, mpi_communicator);
    AssertThrow(vertical_tip_displacement > 0.0,
                ExcMessage("Found no cell with point inside!"))

        pcout
      << "Vertical tip displacement: " << vertical_tip_displacement
      << "\t Check: " << vertical_tip_displacement_check << std::endl;
  }


  // @sect4{Solid::set_total_solution}

  // This function sets the total solution, which is valid at any Newton step.
  // This is required as, to reduce computational error, the total solution is
  // only updated at the end of the timestep.
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::set_total_solution()
  {
    solution_total.equ(1, solution_n);
    solution_total += solution_delta;
  }

  // Note that we must ensure that
  // the matrix is reset before any assembly operations can occur.
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::assemble_system()
  {
    TimerOutput::Scope t(timer, "Assemble linear system");
    pcout << " ASM " << std::flush;

    tangent_matrix      = 0.0;
    system_rhs_trilinos = 0.0;

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<2, dim, NumberType>> solution_grads_u_total(
      qf_cell.size());

    // values at quadrature points:
    std::vector<Tensor<2, dim, NumberType>>          grad_Nx(dofs_per_cell);
    std::vector<SymmetricTensor<2, dim, NumberType>> symm_grad_Nx(
      dofs_per_cell);

    FEValues<dim>     fe_values_ref(fe,
                                qf_cell,
                                update_gradients | update_JxW_values);
    FEFaceValues<dim> fe_face_values_ref(fe,
                                         qf_face,
                                         update_values | update_JxW_values);

    for (const auto &cell : dof_handler_ref.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values_ref.reinit(cell);
          cell_rhs    = 0.;
          cell_matrix = 0.;
          cell->get_dof_indices(local_dof_indices);

          // We first need to find the solution gradients at quadrature points
          // inside the current cell and then we update each local QP using the
          // displacement gradient:
          fe_values_ref[u_fe].get_function_gradients(solution_total,
                                                     solution_grads_u_total);

          // Now we build the local cell stiffness matrix. Since the global and
          // local system matrices are symmetric, we can exploit this property
          // by building only the lower half of the local matrix and copying the
          // values to the upper half.
          //
          // In doing so, we first extract some configuration dependent
          // variables from our QPH history objects for the current quadrature
          // point.
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            {
              const Tensor<2, dim, NumberType> &grad_u =
                solution_grads_u_total[q_point];
              const Tensor<2, dim, NumberType> F =
                Physics::Elasticity::Kinematics::F(grad_u);
              const SymmetricTensor<2, dim, NumberType> b =
                Physics::Elasticity::Kinematics::b(F);
              const NumberType                 det_F = determinant(F);
              const Tensor<2, dim, NumberType> F_bar =
                Physics::Elasticity::Kinematics::F_iso(F);
              const SymmetricTensor<2, dim, NumberType> b_bar =
                Physics::Elasticity::Kinematics::b(F_bar);
              const Tensor<2, dim, NumberType> F_inv = invert(F);
              Assert(det_F > NumberType(0.0), ExcInternalError());

              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  grad_Nx[k] = fe_values_ref[u_fe].gradient(k, q_point) * F_inv;
                  symm_grad_Nx[k] = symmetrize(grad_Nx[k]);
                }

              SymmetricTensor<2, dim, NumberType> tau;
              material->get_tau(tau, det_F, b_bar, b);
              const Tensor<2, dim, NumberType> tau_ns(tau);
              const double                     JxW = fe_values_ref.JxW(q_point);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  cell_rhs(i) -= (symm_grad_Nx[i] * tau) * JxW;

                  for (unsigned int j = 0; j <= i; ++j)
                    {
                      // This is the $\mathsf{\mathbf{k}}_{\mathbf{u}
                      // \mathbf{u}}$ contribution. It comprises a material
                      // contribution, and a geometrical stress contribution
                      // which is only added along the local matrix diagonals:
                      cell_matrix(i, j) +=
                        (symm_grad_Nx[i] *
                         material->act_Jc(
                           det_F,
                           b_bar,
                           b,
                           symm_grad_Nx[j])) // The material contribution:
                        * JxW;
                      // geometrical stress contribution
                      const Tensor<2, dim> geo = egeo_grad(grad_Nx[j], tau_ns);
                      cell_matrix(i, j) +=
                        double_contract<0, 0, 1, 1>(grad_Nx[i], geo) * JxW;
                    }
                }
            } // end loop over quadrature points

          // Finally, we need to copy the lower half of the local matrix into
          // the upper half:
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
              cell_matrix(i, j) = cell_matrix(j, i);

          // Next we assemble the Neumann contribution. We first check to see it
          // the cell face exists on a boundary on which a traction is applied
          // and add the contribution if this is the case.
          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               ++face)
            if (cell->face(face)->at_boundary() == true &&
                cell->face(face)->boundary_id() == 11)
              {
                fe_face_values_ref.reinit(cell, face);
                for (unsigned int f_q_point = 0; f_q_point < n_q_points_f;
                     ++f_q_point)
                  {
                    // We specify the traction in reference configuration.
                    // For this problem, a defined total vertical force is
                    // applied in the reference configuration. The direction of
                    // the applied traction is assumed not to evolve with the
                    // deformation of the domain.

                    // Note that the contributions to the right hand side vector
                    // we compute here only exist in the displacement components
                    // of the vector.
                    const double time_ramp = (time.current() / time.end());
                    const double magnitude =
                      (1.0 /
                       (16.0 * parameters.scale * 1.0 * parameters.scale)) *
                      time_ramp; // (Total force) / (RHS surface area)
                    Tensor<1, dim> dir;
                    dir[1]                        = 1.0;
                    const Tensor<1, dim> traction = magnitude * dir;

                    for (unsigned int i = 0; i < dofs_per_cell; ++i)
                      {
                        const unsigned int component_i =
                          fe.system_to_component_index(i).first;
                        const double Ni =
                          fe_face_values_ref.shape_value(i, f_q_point);
                        const double JxW = fe_face_values_ref.JxW(f_q_point);
                        cell_rhs(i) += (Ni * traction[component_i]) * JxW;
                      }
                  }
              }

          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 tangent_matrix,
                                                 system_rhs_trilinos);
        }

    tangent_matrix.compress(VectorOperation::add);
    system_rhs_trilinos.compress(VectorOperation::add);

    // Determine the true residual error for the problem.  That is, determine
    // the error in the residual for the unconstrained degrees of freedom. Note
    // that to do so, we need to ignore constrained DOFs by setting the residual
    // in these vector components to zero. That will not affect the solution of
    // linear system, though.
    constraints.set_zero(system_rhs_trilinos);
    error_residual.norm = system_rhs_trilinos.l2_norm();
    error_residual.u    = system_rhs_trilinos.l2_norm();

    copy_trilinos(system_rhs, system_rhs_trilinos);
  }


  // @sect4{Solid::make_constraints}
  // The constraints for this problem are simple to describe.
  // However, since we are dealing with an iterative Newton method,
  // it should be noted that any displacement constraints should only
  // be specified at the zeroth iteration and subsequently no
  // additional contributions are to be made since the constraints
  // are already exactly satisfied.
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::make_constraints(
    const int &it_nr)
  {
    pcout << " CST " << std::flush;

    // Since the constraints are different at different Newton iterations, we
    // need to clear the constraints matrix and completely rebuild
    // it. However, after the first iteration, the constraints remain the same
    // and we can simply skip the rebuilding step if we do not clear it.
    if (it_nr > 1)
      return;

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler_ref);

    const bool apply_dirichlet_bc = (it_nr == 0);

    // The boundary conditions for the indentation problem are as follows: On
    // the -x, -y and -z faces (ID's 0,2,4) we set up a symmetry condition to
    // allow only planar movement while the +x and +y faces (ID's 1,3) are
    // traction free. In this contrived problem, part of the +z face (ID 5) is
    // set to have no motion in the x- and y-component. Finally, as described
    // earlier, the other part of the +z face has an the applied pressure but
    // is also constrained in the x- and y-directions.
    //
    // In the following, we will have to tell the function interpolation
    // boundary values which components of the solution vector should be
    // constrained (i.e., whether it's the x-, y-, z-displacements or
    // combinations thereof). This is done using ComponentMask objects (see
    // @ref GlossComponentMask) which we can get from the finite element if we
    // provide it with an extractor object for the component we wish to
    // select. To this end we first set up such extractor objects and later
    // use it when generating the relevant component masks:

    // We also setup GMG constraints

    // Fixed left hand side of the beam
    {
      const int  boundary_id = 1;
      const auto mask        = fe.component_mask(u_fe);

      mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_ref,
                                                         {boundary_id},
                                                         mask);

      if (apply_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(
                                                   n_components),
                                                 constraints,
                                                 fe.component_mask(u_fe));
      else
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(
                                                   n_components),
                                                 constraints,
                                                 fe.component_mask(u_fe));
    }

    // Zero Z-displacement through thickness direction
    // This corresponds to a plane strain condition being imposed on the beam
    if (dim == 3)
      {
        const int                        boundary_id = 2;
        const FEValuesExtractors::Scalar z_displacement(2);
        const auto mask = fe.component_mask(z_displacement);

        mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_ref,
                                                           {boundary_id},
                                                           mask);

        if (apply_dirichlet_bc == true)
          VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                   boundary_id,
                                                   ZeroFunction<dim>(
                                                     n_components),
                                                   constraints,
                                                   mask);
        else
          VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                   boundary_id,
                                                   ZeroFunction<dim>(
                                                     n_components),
                                                   constraints,
                                                   mask);
      }

    constraints.close();
  }

  // @sect4{Solid::solve_linear_system}
  // As the system is composed of a single block, defining a solution scheme
  // for the linear problem is straight-forward.
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  std::tuple<unsigned int, double, double>
  Solid<dim, degree, n_q_points_1d, NumberType>::solve_linear_system(
    LinearAlgebra::distributed::Vector<double> &newton_update,
    TrilinosWrappers::MPI::Vector &             newton_update_trilinos) const
  {
    unsigned int lin_it      = 0;
    double       lin_res     = 0.0;
    double       cond_number = 1.0;

    // reset solution vector each iteration
    newton_update          = 0.;
    newton_update_trilinos = 0.;

    // We solve for the incremental displacement $d\mathbf{u}$.
    const double tol_sol = parameters.tol_lin * system_rhs.l2_norm();

    // estimate condition number of matrix-free operator from dummy CG
    {
      IterationNumberControl control_condition(
        parameters.cond_number_cg_iterations, tol_sol, false, false);
      SolverCG<LinearAlgebra::distributed::Vector<double>> solver_condition(
        control_condition);

      solver_condition.connect_condition_number_slot(
        [&](const double number) { cond_number = number; });

      solver_condition.solve(mf_nh_operator,
                             newton_update,
                             system_rhs,
                             PreconditionIdentity());

      // reset back to zero
      newton_update = 0.;
    }

    timer.enter_subsection("Linear solver");
    const int solver_its = tangent_matrix.m() * parameters.max_iterations_lin;

    SolverControl solver_control(solver_its,
                                 tol_sol,
                                 (debug_level > 1 ? true : false),
                                 (debug_level > 0 ? true : false));

    pcout << " SLV " << std::flush;
    if (parameters.type_lin == "CG" || parameters.type_lin == "MF_CG" ||
        parameters.type_lin == "MF_AD_CG")
      {
        SolverCG<LinearAlgebra::distributed::Vector<double>> solver_CG(
          solver_control);
        constraints.set_zero(newton_update);

        if (parameters.type_lin == "CG")
          {
            AssertThrow(parameters.preconditioner_type == "jacobi",
                        ExcNotImplemented());

            GrowingVectorMemory<TrilinosWrappers::MPI::Vector> GVM_trilinos;
            SolverCG<TrilinosWrappers::MPI::Vector> solver_CG_trilinos(
              solver_control, GVM_trilinos);

            TrilinosWrappers::PreconditionJacobi preconditioner;
            preconditioner.initialize(tangent_matrix,
                                      parameters.preconditioner_relaxation);

            solver_CG_trilinos.solve(tangent_matrix,
                                     newton_update_trilinos,
                                     system_rhs_trilinos,
                                     preconditioner);

            copy_trilinos(newton_update, newton_update_trilinos);
          }
        else if (parameters.type_lin == "MF_AD_CG")
          {
            AssertThrow(parameters.preconditioner_type == "jacobi",
                        ExcNotImplemented());
            PreconditionJacobi<
              NeoHookOperatorAD<dim, degree, n_q_points_1d, double>>
              preconditioner;
            preconditioner.initialize(mf_ad_nh_operator,
                                      parameters.preconditioner_relaxation);

            solver_CG.solve(mf_ad_nh_operator,
                            newton_update,
                            system_rhs,
                            preconditioner);
          }
        else
          {
            Assert(parameters.type_lin == "MF_CG", ExcInternalError());

            if (parameters.preconditioner_type == "jacobi")
              {
                PreconditionJacobi<
                  NeoHookOperator<dim, degree, n_q_points_1d, double>>
                  preconditioner;
                preconditioner.initialize(mf_nh_operator,
                                          parameters.preconditioner_relaxation);

                solver_CG.solve(mf_nh_operator,
                                newton_update,
                                system_rhs,
                                preconditioner);
              }
            else if (parameters.preconditioner_type == "none")
              {
                solver_CG.solve(mf_nh_operator,
                                newton_update,
                                system_rhs,
                                PreconditionIdentity());
              }
            else
              {
                Assert(parameters.preconditioner_type == "gmg",
                       ExcInternalError());
                solver_CG.solve(mf_nh_operator,
                                newton_update,
                                system_rhs,
                                *multigrid_preconditioner);
              }
          }

        lin_it  = solver_control.last_step();
        lin_res = solver_control.last_value();
      }
    else if (parameters.type_lin == "Direct")
      {
        TrilinosWrappers::SolverDirect A_direct(solver_control);
        A_direct.initialize(tangent_matrix);

        A_direct.solve(newton_update_trilinos, system_rhs_trilinos);

        copy_trilinos(newton_update, newton_update_trilinos);

        lin_it  = 1;
        lin_res = 0.0;
      }
    else
      Assert(false, ExcMessage("Linear solver type not implemented"));

    timer.leave_subsection();

    constraints.set_zero(newton_update);

    error_update.norm = newton_update.l2_norm();
    error_update.u    = newton_update.l2_norm();

    // Now that we have the displacement update, distribute the constraints
    // back to the Newton update:
    constraints.distribute(newton_update);

    return std::make_tuple(lin_it, lin_res, cond_number);
  }

  // @sect4{Solid::output_results}
  // Here we present how the results are written to file to be viewed
  // using ParaView or Visit. The method is similar to that shown in the
  // tutorials so will not be discussed in detail.
  template <int dim, int degree, int n_q_points_1d, typename NumberType>
  void
  Solid<dim, degree, n_q_points_1d, NumberType>::output_results() const
  {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof_handler_ref);
    data_out.add_data_vector(solution_n,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Per-cell data (MPI subdomains):
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    // Since we are dealing with a large deformation problem, it would be nice
    // to display the result on a displaced grid!  The MappingQEulerian class
    // linked with the DataOut class provides an interface through which this
    // can be achieved without physically moving the grid points in the
    // Triangulation object ourselves.  We first need to copy the solution to
    // a temporary vector and then create the Eulerian mapping. We also
    // specify the polynomial degree to the DataOut object in order to produce
    // a more refined output data set when higher order polynomials are used.
    MappingQEulerian<dim, LinearAlgebra::distributed::Vector<double>> q_mapping(
      degree, dof_handler_ref, solution_n);
    data_out.build_patches(q_mapping, degree);

    auto name_func = [&](const unsigned int proc) -> std::string {
      return "solution-" + std::to_string(time.get_timestep()) + "_" +
             std::to_string(proc) + ".vtu";
    };

    const std::string filename =
      name_func(triangulation.locally_owned_subdomain());
    std::ofstream output(filename.c_str());
    data_out.write_vtu(output);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        // first write out "pvtu" file which combines output from all MPI cores
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back(name_func(i));

        const std::string filename_pvtu =
          "solution-" + std::to_string(time.get_timestep()) + ".pvtu";
        std::ofstream pvtu_master(filename_pvtu.c_str());
        data_out.write_pvtu_record(pvtu_master, filenames);
      }

    // output MG mesh
    const std::string mg_mesh = "mg_mesh";
    GridOut           grid_out;
    grid_out.write_mesh_per_processor_as_vtu(triangulation,
                                             mg_mesh,
                                             true,
                                             /*artificial*/ false);
  }

} // namespace Cook_Membrane
