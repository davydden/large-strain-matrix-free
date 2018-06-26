#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
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
#include <deal.II/physics/transformations.h>


#include <iostream>

#include <mf_elasticity.h>

// same as mf_neo_hook_operator but tests cell-based large strain elasticity operator
// with formulation 1

using namespace dealii;

template <int dim>
class Shear : public Function<dim>
{
public:
  Shear() :
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


/**
 * Similar to shear but make i non-linear (quadratic)
 */
template <int dim>
class Shear2 : public Function<dim>
{
public:
  Shear2() :
    Function<dim>(dim)
  {}

  double value (const Point<dim> &p,
                const unsigned int component) const
  {
    Assert (dim>=2, ExcNotImplemented());
    // simple shear
    static const double gamma = 0.1;
    if (component==0)
      return p[1]*p[1]*gamma;
    else
      return 0.;
  }
};



template <int dim>
class UniaxialTension : public Function<dim>
{
public:
  UniaxialTension() :
    Function<dim>(dim)
  {}

  double value (const Point<dim> &p,
                const unsigned int component) const
  {
    Assert (dim>=2, ExcNotImplemented());
    static const double dx = 0.1;
    if (component==0)
      return p[0]*dx;
    else
      return 0.;
  }
};


/**
 * Same as UniaxialTension but non-linear (quadratic w.r.t. Px)
 */
template <int dim>
class UniaxialTension2 : public Function<dim>
{
public:
  UniaxialTension2() :
    Function<dim>(dim)
  {}

  double value (const Point<dim> &p,
                const unsigned int component) const
  {
    Assert (dim>=2, ExcNotImplemented());
    static const double dx = 0.1;
    if (component==0)
      return p[0]*p[0]*dx;
    else
      return 0.;
  }
};

template <int dim> class Rotation;

template <>
class Rotation<2> : public Function<2>
{
public:
  Rotation() :
    Function<2>(2),
    matrix(Physics::Transformations::Rotations::rotation_matrix_2d(0.5))
  {}

  double value (const Point<2> &p,
                const unsigned int component) const
  {
    Tensor<1,2> t = p;
    Tensor<1,2> res = matrix * t;
    return res[component];
  }

  const Tensor<2,2> matrix;
};



template <>
class Rotation<3> : public Function<3>
{
public:
  Rotation() :
    Function<3>(3)
  {}

  double value (const Point<3> &p,
                const unsigned int component) const
  {
    const Tensor<2,3> matrix= Physics::Transformations::Rotations::rotation_matrix_3d(Point<3>(0,0,1.),0.6 * p[2]);
    Tensor<1,3> t = p;
    Tensor<1,3> res = matrix * t;
    return res[component];
  }
};




template <int dim, int fe_degree, int n_q_points_1d=fe_degree+1>
void test_elasticity (const Function<dim> &displacement_function)
{
  typedef double number;
  typedef VectorizedArray<number> NumberType;

  parallel::distributed::Triangulation<dim> tria (MPI_COMM_WORLD);
  GridGenerator::hyper_cube (tria);

  FESystem<dim> fe(FE_Q<dim>{fe_degree},dim);
  DoFHandler<dim> dof (tria);
  dof.distribute_dofs(fe);

  IndexSet owned_set = dof.locally_owned_dofs();
  IndexSet relevant_set;
  DoFTools::extract_locally_relevant_dofs (dof, relevant_set);

  AffineConstraints<double> constraints (relevant_set);
  DoFTools::make_hanging_node_constraints(dof, constraints);
  // VectorTools::interpolate_boundary_values (dof, 0, Functions::ZeroFunction<dim>(),
  //                                           constraints);
  constraints.close();

  std::cout << "Testing " << dof.get_fe().get_name() << std::endl;
  std::cout << "Number of cells: " << tria.n_global_active_cells() << std::endl;
  std::cout << "Number of degrees of freedom: " << dof.n_dofs() << std::endl;

  // setup some FE displacement
  LinearAlgebra::distributed::Vector<number> displacement, src, dst;
  displacement.reinit(owned_set,
                      relevant_set,
                      MPI_COMM_WORLD);

  {
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

  const QGauss<1> quad (n_q_points_1d);
  typename MatrixFree<dim,number>::AdditionalData data;
  data.tasks_parallel_scheme = MatrixFree<dim,number>::AdditionalData::none;
  data.tasks_block_size = 7;

  mf_data_reference->reinit (         dof, constraints, quad, data);
  mf_data_current->reinit   (*mapping,dof, constraints, quad, data);

  mf_data_current->initialize_dof_vector(dst);
  mf_data_current->initialize_dof_vector(src);

  for (unsigned int i=0; i<src.local_size(); ++i)
    src.local_element(i) = ((double)std::rand())/RAND_MAX;

  constraints.set_zero(src);

  const double nu = 0.3; // poisson
  const double mu = 0.4225e6; // shear
  const double lambda = (2.0*mu*nu)/(1.0-2.0*nu);
  const unsigned int material_formulation = 1;
  Material_Compressible_Neo_Hook_One_Field<dim,VectorizedArray<number>> material(mu,nu,material_formulation);
  Material_Compressible_Neo_Hook_One_Field<dim,number> material_standard(mu,nu,material_formulation );

  // before going into the cell loop, for Eulerian part one should reinitialize MatrixFree with
  // initialize_indices=false
  // as the mapping has to be recomputed but the topology of cells is the same
  data.initialize_indices = false;
  mf_data_current->reinit   (*mapping,dof, constraints, quad, data);

  // do one cell:
  FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_current  (*mf_data_current);
  FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_current_s(*mf_data_current);
  FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_reference(*mf_data_reference);

  const unsigned int n_q_points = phi_current.n_q_points;
  Assert (phi_current.n_q_points == phi_reference.n_q_points, ExcInternalError());

  const unsigned int cell=0;

  //
  // for debug purpose the matrix-based part
  //
  const auto dof_cell = dof.begin_active();
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const QGauss<dim> qf_cell(n_q_points_1d);
  std::vector<Tensor<2,dim,number>>  solution_grads_u_total(qf_cell.size());
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const FEValuesExtractors::Vector u_fe(0);
  FEValues<dim>  fe_values_ref(fe, qf_cell, update_gradients | update_JxW_values | update_quadrature_points);
  std::vector<Tensor<2, dim,number>>         grad_Nx(dofs_per_cell);
  std::vector<SymmetricTensor<2,dim,number>> symm_grad_Nx(dofs_per_cell);
  FullMatrix<double> cell_matrix(dofs_per_cell,dofs_per_cell);
  Vector<double> src_local(dofs_per_cell);
  Vector<double> dst_local(dofs_per_cell);
  Vector<double> dst_diff(dofs_per_cell);
  //
  //

  {
    fe_values_ref.reinit(dof_cell);
    fe_values_ref[u_fe].get_function_gradients(displacement, solution_grads_u_total);
    dof_cell->get_dof_indices(local_dof_indices);
    dof_cell->get_dof_values(src,
                             src_local.begin(),
                             src_local.end());

    // initialize on this cell
    phi_current.reinit(cell);
    phi_current_s.reinit(cell);
    phi_reference.reinit(cell);

    // read-in total displacement and src vector and evaluate gradients
    phi_reference.read_dof_values(displacement);
    phi_current.  read_dof_values(src);
    phi_current_s.read_dof_values(src);
    phi_reference.evaluate (false,true,false);
    phi_current.  evaluate (false,true,false);
    phi_current_s.evaluate (false,true,false);
    for (unsigned int q=0; q<n_q_points; ++q)
      {
        // formulation == 1 in mf_nh_operator.h
        // reference configuration:
        const Tensor<2,dim,NumberType>         &grad_u = phi_reference.get_gradient(q);
        const Tensor<2,dim,NumberType>          F      = Physics::Elasticity::Kinematics::F(grad_u);
        const SymmetricTensor<2,dim,NumberType> b      = Physics::Elasticity::Kinematics::b(F);

        const Tensor<2,dim,NumberType>          &grad_Nx_v      = phi_current.get_gradient(q);
        const SymmetricTensor<2,dim,NumberType> &symm_grad_Nx_v = phi_current.get_symmetric_gradient(q);

        // cached part
        const VectorizedArray<number>           det_F  = determinant(F);
        const VectorizedArray<number>           log_J  = std::log(det_F);


        SymmetricTensor<2,dim,NumberType> tau;
        {
          tau = mu*b;
          const NumberType tmp = mu - 2.0*lambda*log_J;
          for (unsigned int d = 0; d < dim; ++d)
            tau[d][d] -= tmp;
        }

        SymmetricTensor<2,dim,VectorizedArray<number>> jc_part;
        {
          jc_part = 2.0*(mu - 2.0*lambda*log_J)*symm_grad_Nx_v;
          const NumberType tmp = 2.0*lambda*trace(symm_grad_Nx_v);
          for (unsigned int i = 0; i < dim; ++i)
            jc_part[i][i] += tmp;
        }

        const VectorizedArray<number> & JxW_current = phi_current.JxW(q);
        VectorizedArray<number> JxW_scale = phi_reference.JxW(q);
        for (unsigned int i = 0; i < VectorizedArray<number>::n_array_elements; ++i)
          if (std::abs(JxW_current[i])>1e-10)
            JxW_scale[i] *= 1./JxW_current[i];

        phi_current_s.submit_symmetric_gradient(
          jc_part * JxW_scale,q);

        const Tensor<2,dim,VectorizedArray<number>> tau_ns (tau);
        const Tensor<2,dim,VectorizedArray<number>> geo = grad_Nx_v * tau_ns;
        phi_current.submit_gradient(
          geo * JxW_scale,q);

        //=================
        // DEBUG
        //=================
        // Grad u
        const Tensor<2,dim,number> &grad_u_standard = solution_grads_u_total[q];
        const Tensor<2,dim,number>  F_standard = Physics::Elasticity::Kinematics::F(grad_u_standard);
        const SymmetricTensor<2,dim,number>  b_standard = Physics::Elasticity::Kinematics::b(F_standard);
        const number                det_F_standard = determinant(F_standard);
        const Tensor<2,dim,number> F_bar_standard = Physics::Elasticity::Kinematics::F_iso(F_standard);
        const SymmetricTensor<2,dim,number> b_bar_standard = Physics::Elasticity::Kinematics::b(F_bar_standard);
        const Tensor<2,dim,number>  F_inv_standard = invert(F_standard);

        // v_k Grad Nk * F^{-1} = v_k grad Nk = grad v  , v - source vector
        Tensor<2,dim,number>  grad_Nx_v_standard;
        SymmetricTensor<2,dim,number> symm_grad_Nx_v_standard;
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            grad_Nx[k] = fe_values_ref[u_fe].gradient(k, q) * F_inv_standard;
            symm_grad_Nx[k] = symmetrize(grad_Nx[k]);

            grad_Nx_v_standard      += src(local_dof_indices[k]) * grad_Nx[k];
            symm_grad_Nx_v_standard += src(local_dof_indices[k]) * symm_grad_Nx[k];
          }

        SymmetricTensor<2,dim,number> tau_standard;
        material_standard.get_tau(tau_standard,det_F_standard,b_bar_standard,b_standard);
        const Tensor<2,dim,number> tau_ns_standard (tau_standard);
        const double JxW = fe_values_ref.JxW(q);

        const SymmetricTensor<2,dim,number> jc_part_standard = material_standard.act_Jc(det_F_standard,b_bar_standard,b_standard,symm_grad_Nx_v_standard);
        const Tensor<2, dim> geo_standard_v = egeo_grad(grad_Nx_v_standard,tau_ns_standard);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j <= i; ++j)
            {
              cell_matrix(i, j) += (symm_grad_Nx[i] * material_standard.act_Jc(det_F_standard,b_bar_standard,b_standard,symm_grad_Nx[j]))
                                   * JxW;
              const Tensor<2, dim> geo_standard = egeo_grad(grad_Nx[j],tau_ns_standard);
              cell_matrix(i, j) += double_contract<0,0,1,1>(grad_Nx[i],geo_standard) * JxW;
            }

        const auto & qp_coord = fe_values_ref.get_quadrature_points();
        std::cout << "=====================" << std::endl
                  << "quadrature point " << q << ": "
                  << qp_coord[q] << std::endl;

        std::cout << "JxW ref:" << std::endl
                  << JxW << std::endl
                  << phi_reference.JxW(q)[0] << std::endl
                  << "JxW current:" << std::endl
                  << phi_current.JxW(q)[0] << std::endl;

        std::cout << "Grad u:"<< std::endl;
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              std::cout << grad_u_standard[i][j] << " ";
            std::cout << std::endl;
          }
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              std::cout << grad_u[i][j][0] << " ";
            std::cout << std::endl;
          }

        std::cout << "v_k grad N_k = grad v:"<< std::endl;
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              std::cout << grad_Nx_v_standard[i][j] << " ";
            std::cout << std::endl;
          }
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              std::cout << grad_Nx_v[i][j][0] << " ";
            std::cout << std::endl;
          }

        std::cout << "Jc action:" << std::endl;
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              std::cout << jc_part_standard[i][j] << " ";
            std::cout << std::endl;
          }
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              std::cout << jc_part[i][j][0] << " ";
            std::cout << std::endl;
          }

        std::cout << "geo action:" << std::endl;
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              std::cout << geo_standard_v[i][j] << " ";
            std::cout << std::endl;
          }
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              std::cout << geo[i][j][0] << " ";
            std::cout << std::endl;
          }

        Tensor<2,dim,number> grad_u_diff, grad_Nx_v_diff, jc_diff, geo_diff;
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
            {
              grad_u_diff[i][j]    = grad_u_standard[i][j]    - grad_u[i][j][0];
              grad_Nx_v_diff[i][j] = grad_Nx_v_standard[i][j] - grad_Nx_v[i][j][0];
              jc_diff[i][j]        = jc_part_standard[i][j]   - jc_part[i][j][0];
              geo_diff[i][j]       = geo_standard_v[i][j]     - geo[i][j][0];
            }

        AssertThrow(grad_u_diff.norm() < 1e-12, ExcMessage("Grad u"));
        AssertThrow(grad_Nx_v_diff.norm() < 1e-12, ExcMessage("v_k grad N_k"));
        AssertThrow(jc_diff.norm() < 1e-12 * jc_part_standard.norm() , ExcMessage("Jc"));
        AssertThrow(geo_diff.norm() < 1e-12 * geo_standard_v.norm(), ExcMessage("geo"));

      } // end of the loop over quadrature

    // actually do the contraction
    phi_current.integrate (false,true);
    phi_current_s.integrate (false,true);

    phi_current.distribute_local_to_global(dst);
    phi_current_s.distribute_local_to_global(dst);

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
        cell_matrix(i, j) = cell_matrix(j, i);

    cell_matrix.vmult(dst_local,src_local);

    std::cout << std::endl << "vmult matrix-based:" << std::endl;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      std::cout << dst_local[i] << " ";
    std::cout << std::endl << "vmult matrix-free:" << std::endl;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      std::cout << dst[local_dof_indices[i]] << " ";

    std::cout << std::endl << "diff:" << std::endl;
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const double diff = dst_local[i] - dst[local_dof_indices[i]];
        std::cout << diff << " " ;
        AssertThrow(std::abs(diff) < 1e-10 * std::abs(dst_local[i]), ExcInternalError());
      }
    std::cout << std::endl;

  } // end of the loop over cells

  deallog << "Ok" << std::endl;

}


int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

  Assert (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)==1,
          ExcNotImplemented());

  const std::string deallogname = "output";
  std::ofstream deallogfile;
  deallogfile.open(deallogname.c_str());
  deallog.attach(deallogfile);
  deallog.depth_console(0);
  deallog << std::setprecision(4);

  deallog.push("2d");
  std::cout << "=====================" << std::endl << "Shear" << std::endl;
  test_elasticity<2,1>(Shear<2>());
  std::cout << "=====================" << std::endl << "Shear2" << std::endl;
  test_elasticity<2,2>(Shear2<2>());
  std::cout << "=====================" << std::endl << "UniaxialTension" << std::endl;
  test_elasticity<2,1>(UniaxialTension<2>());
  std::cout << "=====================" << std::endl << "UniaxialTension2" << std::endl;
  test_elasticity<2,2>(UniaxialTension2<2>());

  deallog.pop();
  deallog.push("3d");
  std::cout << "=====================" << std::endl << "Shear" << std::endl;
  test_elasticity<3,1>(Shear<3>());
  std::cout << "=====================" << std::endl << "Shear2" << std::endl;
  test_elasticity<3,2>(Shear2<3>());
  std::cout << "=====================" << std::endl << "UniaxialTension" << std::endl;
  test_elasticity<3,1>(UniaxialTension<3>());
  std::cout << "=====================" << std::endl << "UniaxialTension2" << std::endl;
  test_elasticity<3,2>(UniaxialTension2<3>());
  std::cout << "=====================" << std::endl << "Rotation" << std::endl;
  test_elasticity<3,1>(Rotation<3>());
  deallog.pop();

}
