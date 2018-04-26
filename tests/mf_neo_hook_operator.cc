#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>

#include <iostream>


using namespace dealii;

template <int dim, int fe_degree>
void test ()
{
  typedef double number;

  parallel::distributed::Triangulation<dim> tria (MPI_COMM_WORLD);
  GridGenerator::hyper_cube (tria);
  tria.refine_global(1);
  typename Triangulation<dim>::active_cell_iterator
  cell = tria.begin_active (),
  endc = tria.end();
  cell = tria.begin_active ();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      if (cell->center().norm()<0.2)
        cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();
  if (dim < 3 && fe_degree < 2)
    tria.refine_global(2);
  else
    tria.refine_global(1);
  if (tria.begin(tria.n_levels()-1)->is_locally_owned())
    tria.begin(tria.n_levels()-1)->set_refine_flag();
  if (tria.last()->is_locally_owned())
    tria.last()->set_refine_flag();
  tria.execute_coarsening_and_refinement();
  cell = tria.begin_active ();
  for (unsigned int i=0; i<10-3*dim; ++i)
    {
      cell = tria.begin_active ();
      unsigned int counter = 0;
      for (; cell!=endc; ++cell, ++counter)
        if (cell->is_locally_owned())
          if (counter % (7-i) == 0)
            cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }

  FE_Q<dim> fe (fe_degree);
  DoFHandler<dim> dof (tria);
  dof.distribute_dofs(fe);

  IndexSet owned_set = dof.locally_owned_dofs();
  IndexSet relevant_set;
  DoFTools::extract_locally_relevant_dofs (dof, relevant_set);

  ConstraintMatrix constraints (relevant_set);
  DoFTools::make_hanging_node_constraints(dof, constraints);
  VectorTools::interpolate_boundary_values (dof, 0, Functions::ZeroFunction<dim>(),
                                            constraints);
  constraints.close();

  deallog << "Testing " << dof.get_fe().get_name() << std::endl;
  //std::cout << "Number of cells: " << tria.n_global_active_cells() << std::endl;
  //std::cout << "Number of degrees of freedom: " << dof.n_dofs() << std::endl;
  //std::cout << "Number of constraints: " << constraints.n_constraints() << std::endl;

  std::shared_ptr<MatrixFree<dim,number> > mf_data(new MatrixFree<dim,number> ());
  {
    const QGauss<1> quad (fe_degree+2);
    typename MatrixFree<dim,number>::AdditionalData data;
    data.tasks_parallel_scheme =
      MatrixFree<dim,number>::AdditionalData::none;
    data.tasks_block_size = 7;
    mf_data->reinit (dof, constraints, quad, data);
  }

  MatrixFreeOperators::MassOperator<dim,fe_degree, fe_degree+2, 1, LinearAlgebra::distributed::Vector<number> > mf;
  mf.initialize(mf_data);
  mf.compute_diagonal();
  LinearAlgebra::distributed::Vector<number> in, out, ref;
  mf_data->initialize_dof_vector (in);
  out.reinit (in);
  ref.reinit (in);

  for (unsigned int i=0; i<in.local_size(); ++i)
    {
      const unsigned int glob_index =
        owned_set.nth_index_in_set (i);
      if (constraints.is_constrained(glob_index))
        continue;
      in.local_element(i) = ((double)std::rand())/RAND_MAX;
    }

  mf.vmult (out, in);


  // assemble trilinos sparse matrix with
  // (v, u) for reference
  TrilinosWrappers::SparseMatrix sparse_matrix;
  {
    TrilinosWrappers::SparsityPattern csp (owned_set, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern (dof, csp, constraints, true,
                                     Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    csp.compress();
    sparse_matrix.reinit (csp);
  }
  {
    QGauss<dim>  quadrature_formula(fe_degree+2);

    FEValues<dim> fe_values (dof.get_fe(), quadrature_formula,
                             update_values    |  update_gradients |
                             update_JxW_values);

    const unsigned int   dofs_per_cell = dof.get_fe().dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
    cell = dof.begin_active(),
    endc = dof.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          fe_values.reinit (cell);

          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                  cell_matrix(i,j) += (fe_values.shape_value(i,q_point) *
                                       fe_values.shape_value(j,q_point)) *
                                      fe_values.JxW(q_point);
              }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global (cell_matrix,
                                                  local_dof_indices,
                                                  sparse_matrix);
        }
  }
  sparse_matrix.compress(VectorOperation::add);

  sparse_matrix.vmult (ref, in);
  out -= ref;
  const double diff_norm = out.linfty_norm();

  deallog << "Norm of difference: " << diff_norm << std::endl << std::endl;
}


template <int dim>
class Displacement : public Function<dim>
{
public:
  Displacement() :
    Function<dim>(dim)
  {}

  double value (const Point<dim> &p,
                const unsigned int component) const
  {
    Assert (dim>=2, ExcNotImplemented());
    // simple shear
    static const double gamma = 0.1;
    if (component==0)
      return p[1]*gamma;
    else
      return 0.;
  }
};


template <int dim, int fe_degree, int n_q_points_1d>
void test_elasticity ()
{
  typedef double number;
  parallel::distributed::Triangulation<dim> tria (MPI_COMM_WORLD);
  GridGenerator::hyper_cube (tria);

  FESystem<dim> fe(FE_Q<dim>(fe_degree),dim);
  DoFHandler<dim> dof (tria);
  dof.distribute_dofs(fe);

  IndexSet owned_set = dof.locally_owned_dofs();
  IndexSet relevant_set;
  DoFTools::extract_locally_relevant_dofs (dof, relevant_set);

  ConstraintMatrix constraints (relevant_set);
  DoFTools::make_hanging_node_constraints(dof, constraints);
  // VectorTools::interpolate_boundary_values (dof, 0, Functions::ZeroFunction<dim>(),
  //                                           constraints);
  constraints.close();

  std::cout << "Testing " << dof.get_fe().get_name() << std::endl;
  std::cout << "Number of cells: " << tria.n_global_active_cells() << std::endl;
  std::cout << "Number of degrees of freedom: " << dof.n_dofs() << std::endl;

  // setup some FE displacement
  LinearAlgebra::distributed::Vector<number> displacement;
  displacement.reinit(owned_set,
                      relevant_set,
                      MPI_COMM_WORLD);

  {
    Displacement<dim> displacement_function;
    VectorTools::interpolate(dof,
                             displacement_function,
                             displacement);
    displacement.compress(VectorOperation::insert);
    displacement.update_ghost_values();
  }

  // setup current configuration mapping
  auto mapping = std::make_shared<MappingQEulerian<dim,LinearAlgebra::distributed::Vector<number>>>(/*degree*/1,dof,displacement);
  //auto mapping = std::make_shared<MappingFEField<dim,dim,LinearAlgebra::distributed::Vector<number>>>(dof,displacement);

  // output for debug purposes
  {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim,
                                  DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof);
    data_out.add_data_vector(displacement,
                             solution_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // write output
    data_out.build_patches(*mapping);

    const std::string filename = "solution.vtk";
    std::ofstream output(filename.c_str());
    data_out.write_vtk(output);
  }

  std::shared_ptr<MatrixFree<dim,number> > mf_data_current  (new MatrixFree<dim,number> ());
  std::shared_ptr<MatrixFree<dim,number> > mf_data_reference(new MatrixFree<dim,number> ());
  {
    const QGauss<1> quad (n_q_points_1d);
    typename MatrixFree<dim,number>::AdditionalData data;
    data.tasks_parallel_scheme = MatrixFree<dim,number>::AdditionalData::none;
    data.tasks_block_size = 7;

    mf_data_reference->reinit (         dof, constraints, quad, data);
    mf_data_current->reinit   (*mapping,dof, constraints, quad, data);
  }

  // do one cell:
  FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_current  (*mf_data_current);
  FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_reference(*mf_data_reference);

  const unsigned int n_q_points = phi_current.n_q_points;
  Assert (phi_current.n_q_points == phi_reference.n_q_points, ExcInternalError());

  const unsigned int cell=0;
  {
    // initialize on this cell
    phi_current.reinit(cell);
    phi_reference.reinit(cell);

    // read-in solution vector and evaluate gradients
    phi_reference.read_dof_values(displacement);
    phi_current.  read_dof_values(displacement);
    phi_reference.evaluate (false,true,false);
    phi_current.  evaluate (false,true,false);
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        // reference configuration:
        const Tensor<2,dim,VectorizedArray<number>>         &grad_u = phi_reference.get_gradient(q);
        const Tensor<2,dim,VectorizedArray<number>>          F      = Physics::Elasticity::Kinematics::F(grad_u);
        const VectorizedArray<number>                        det_F  = determinant(F);
        const Tensor<2,dim,VectorizedArray<number>>          F_bar  = Physics::Elasticity::Kinematics::F_iso(F);
        const SymmetricTensor<2,dim,VectorizedArray<number>> b_bar  = Physics::Elasticity::Kinematics::b(F_bar);

        // current configuration
        const Tensor<2,dim,VectorizedArray<number>>          &grad_Nx_u      = phi_current.get_gradient(q);
        const SymmetricTensor<2,dim,VectorizedArray<number>> &symm_grad_Nx_u = phi_current.get_symmetric_gradient(q);
      }
  }

  // MatrixFreeOperators::MassOperator<dim,fe_degree, fe_degree+2, 1, LinearAlgebra::distributed::Vector<number> > mf;
  // mf.initialize(mf_data);
  // mf.compute_diagonal();
  // LinearAlgebra::distributed::Vector<number> in, out, ref;
  // mf_data->initialize_dof_vector (in);
  // out.reinit (in);
  // ref.reinit (in);

  // for (unsigned int i=0; i<in.local_size(); ++i)
  //   {
  //     const unsigned int glob_index =
  //       owned_set.nth_index_in_set (i);
  //     if (constraints.is_constrained(glob_index))
  //       continue;
  //     in.local_element(i) = ((double)std::rand())/RAND_MAX;
  //   }

  // mf.vmult (out, in);

}


int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

  unsigned int myid = Utilities::MPI::this_mpi_process (MPI_COMM_WORLD);
  deallog.push(Utilities::int_to_string(myid));

  if (myid == 0)
    {
      const std::string deallogname = "output";
      std::ofstream deallogfile;
      deallogfile.open(deallogname.c_str());
      deallog.attach(deallogfile);
      deallog.depth_console(0);
      deallog << std::setprecision(4);

      deallog.push("2d");
      test<2,1>();
      test<2,2>();
      test_elasticity<2,1,2>();
      deallog.pop();

      deallog.push("3d");
      test<3,1>();
      test<3,2>();
      deallog.pop();
    }
  else
    {
      test<2,1>();
      test<2,2>();
      test_elasticity<2,1,2>();
      test<3,1>();
      test<3,2>();
    }
}