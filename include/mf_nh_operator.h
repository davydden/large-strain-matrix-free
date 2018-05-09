#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <material.h>

using namespace dealii;

  /**
   * Large strain Neo-Hook tangent operator.
   *
   * Follow https://github.com/dealii/dealii/blob/master/tests/matrix_free/step-37.cc
   */
  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  class NeoHookOperator : public Subscriptor
  {
  public:
    NeoHookOperator ();

    void clear();

    void initialize(std::shared_ptr<const MatrixFree<dim,number>> data_current,
                    std::shared_ptr<const MatrixFree<dim,number>> data_reference,
                    Vector<number> &displacement);

    void set_material(std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim,VectorizedArray<number>>> material);

    unsigned int m () const;
    unsigned int n () const;

    void vmult (Vector<double> &dst,
                const Vector<double> &src) const;

    void Tvmult (Vector<double> &dst,
                 const Vector<double> &src) const;
    void vmult_add (Vector<double> &dst,
                    const Vector<double> &src) const;
    void Tvmult_add (Vector<double> &dst,
                     const Vector<double> &src) const;
    /*
    number el (const unsigned int row,
               const unsigned int col) const;
    void set_diagonal (const Vector<number> &diagonal);
    */

  private:
    /*
    void local_apply (const MatrixFree<dim,number>    &data,
                      Vector<double>                      &dst,
                      const Vector<double>                &src,
                      const std::pair<unsigned int,unsigned int> &cell_range) const;
    */

   /**
    * Perform operation on a cell. @p phi_current and @phi_current_s correspond to the deformed configuration
    * where @p phi_reference is for the current configuration.
    */
   void do_operation_on_cell(FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_current,
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_current_s,
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_reference,
                             const unsigned int cell) const;

    std::shared_ptr<const MatrixFree<dim,number>> data_current;
    std::shared_ptr<const MatrixFree<dim,number>> data_reference;

    Vector<number> *displacement;

    std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim,VectorizedArray<number>>> material;

    Vector<number>  diagonal_values;
    bool            diagonal_is_available;
  };



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::NeoHookOperator ()
    :
    Subscriptor()
  {}



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  unsigned int
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::m () const
  {
    return data_current.get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  unsigned int
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::n () const
  {
    return data_current.get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::clear ()
  {
    data_current.reset();
    data_reference.reset();
    diagonal_is_available = false;
    diagonal_values.reinit(0);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::initialize(
                    std::shared_ptr<const MatrixFree<dim,number>> data_current_,
                    std::shared_ptr<const MatrixFree<dim,number>> data_reference_,
                    Vector<number> &displacement_)
  {
    data_current = data_current_;
    data_reference = data_reference_;
    displacement = &displacement_;
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::set_material(std::shared_ptr<Material_Compressible_Neo_Hook_One_Field<dim,VectorizedArray<number>>> material_)
  {
    material = material_;
  }

  /*
  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  LaplaceOperator<dim,fe_degree,n_q_points_1d,number>::reinit (const DoFHandler<dim>  &dof_handler,
                                                 const ConstraintMatrix &constraints,
                                                 const unsigned int      level)
  {
    typename MatrixFree<dim,number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim,number>::AdditionalData::partition_color;
    additional_data.level_mg_handler = level;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points);
    data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
                 additional_data);
    evaluate_coefficient(Coefficient<dim>());
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::
  evaluate_coefficient (const Coefficient<dim> &coefficient_function)
  {
    const unsigned int n_cells = data.n_macro_cells();
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);
    coefficient.resize (n_cells * phi.n_q_points);
    for (unsigned int cell=0; cell<n_cells; ++cell)
      {
        phi.reinit (cell);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          coefficient[cell*phi.n_q_points+q] =
            coefficient_function.value(phi.quadrature_point(q));
      }
  }




  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::
  local_apply (const MatrixFree<dim,number>         &data,
               Vector<double>                       &dst,
               const Vector<double>                 &src,
               const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);
    AssertDimension (coefficient.size(),
                     data.n_macro_cells() * phi.n_q_points);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        phi.read_dof_values(src);
        phi.evaluate (false,true,false);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          phi.submit_gradient (coefficient[cell*phi.n_q_points+q] *
                               phi.get_gradient(q), q);
        phi.integrate (false,true);
        phi.distribute_local_to_global (dst);
      }
  }
  */



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::vmult (Vector<double>       &dst,
                                                const Vector<double> &src) const
  {
    dst = 0;
    vmult_add (dst, src);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::Tvmult (Vector<double>       &dst,
                                                 const Vector<double> &src) const
  {
    dst = 0;
    vmult_add (dst,src);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::Tvmult_add (Vector<double>       &dst,
                                                     const Vector<double> &src) const
  {
    vmult_add (dst,src);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::vmult_add (Vector<double>       &dst,
                                                    const Vector<double> &src) const
  {
    // FIXME: can't use cell_loop as we need both matrix-free data objects.
    // for now do it by hand.
    // BUT I might try cell_loop(), and simply use another MF object inside...
    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_current  (*data_current);
    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_current_s(*data_current);
    FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> phi_reference(*data_reference);

    const unsigned int n_cells = data_current->n_macro_cells();

    Assert (n_cells == data_reference->n_macro_cells(), ExcInternalError());
    Assert (phi_current.n_q_points == phi_reference.n_q_points, ExcInternalError());

    // MatrixFree::cell_loop() is more complicated than a simple update_ghost_values() / compress(),
    // it loops on different cells (inner without ghosts and outer) in different order
    // and do update_ghost_values() and compress_start()/compress_finish() in between.
    // https://www.dealii.org/developer/doxygen/deal.II/matrix__free_8h_source.html#l00109

    // 1. make sure ghosts are updated
    // src.update_ghost_values();

    // 2. loop over all locally owned cell blocks
    for (unsigned int cell=0; cell<n_cells; ++cell)
      {
        // initialize on this cell
        phi_current.reinit(cell);
        phi_current_s.reinit(cell);
        phi_reference.reinit(cell);

        // read-in total displacement and src vector and evaluate gradients
        phi_reference.read_dof_values_plain(*displacement);
        phi_current.  read_dof_values(src);
        phi_current_s.read_dof_values(src);

        do_operation_on_cell(phi_current,phi_current_s,phi_reference,cell);

        phi_current.distribute_local_to_global(dst);
        phi_current_s.distribute_local_to_global(dst);

      }  // end of the loop over cells

    // 3. communicate results with MPI
    // dst.compress(VectorOperation::add);

    // 4. constraints
    const std::vector<unsigned int> &
    constrained_dofs = data_current->get_constrained_dofs(); // FIXME: is it current or reference?
    for (unsigned int i=0; i<constrained_dofs.size(); ++i)
      dst(constrained_dofs[i]) += src(constrained_dofs[i]);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::do_operation_on_cell(
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_current,
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_current_s,
                             FEEvaluation<dim,fe_degree,n_q_points_1d,dim,number> &phi_reference,
                             const unsigned int /*cell*/) const
  {
    phi_reference.evaluate (false,true,false);
    phi_current.  evaluate (false,true,false);
    phi_current_s.evaluate (false,true,false);

    for (unsigned int q=0; q<phi_current.n_q_points; ++q)
      {
        // reference configuration:
        const Tensor<2,dim,VectorizedArray<number>>         &grad_u = phi_reference.get_gradient(q);
        const Tensor<2,dim,VectorizedArray<number>>          F      = Physics::Elasticity::Kinematics::F(grad_u);
        const VectorizedArray<number>                        det_F  = determinant(F);
        const Tensor<2,dim,VectorizedArray<number>>          F_bar  = Physics::Elasticity::Kinematics::F_iso(F);
        const SymmetricTensor<2,dim,VectorizedArray<number>> b_bar  = Physics::Elasticity::Kinematics::b(F_bar);

        // current configuration
        const Tensor<2,dim,VectorizedArray<number>>          &grad_Nx_v      = phi_current.get_gradient(q);
        const SymmetricTensor<2,dim,VectorizedArray<number>> &symm_grad_Nx_v = phi_current.get_symmetric_gradient(q);

        const SymmetricTensor<2,dim,VectorizedArray<number>> tau = material->get_tau(det_F,b_bar);
        const Tensor<2,dim,VectorizedArray<number>> tau_ns (tau);

        const SymmetricTensor<2,dim,VectorizedArray<number>> jc_part = material->act_Jc(det_F,b_bar,symm_grad_Nx_v);

        const VectorizedArray<number> & JxW_current = phi_current.JxW(q);
        VectorizedArray<number> JxW_scale = phi_reference.JxW(q);
        for (unsigned int i = 0; i < VectorizedArray<number>::n_array_elements; ++i)
          if (std::abs(JxW_current[i])>1e-10)
            JxW_scale[i] *= 1./JxW_current[i];

        // This is the $\mathsf{\mathbf{k}}_{\mathbf{u} \mathbf{u}}$
        // contribution. It comprises a material contribution, and a
        // geometrical stress contribution which is only added along
        // the local matrix diagonals:
        phi_current_s.submit_symmetric_gradient(
          jc_part * JxW_scale
          // Note: We need to integrate over the reference element, so the weights have to be adjusted
          ,q);

        // geometrical stress contribution
        const Tensor<2,dim,VectorizedArray<number>> geo = egeo_grad(grad_Nx_v,tau_ns);
        phi_current.submit_gradient(
          geo * JxW_scale
          // Note: We need to integrate over the reference element, so the weights have to be adjusted
          // phi_reference.JxW(q) / phi_current.JxW(q)
          ,q);

      } // end of the loop over quadrature points

    // actually do the contraction
    phi_current.integrate (false,true);
    phi_current_s.integrate (false,true);
  }


  /*
  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  number
  LaplaceOperator<dim,fe_degree,n_q_points_1d,number>::el (const unsigned int row,
                                             const unsigned int col) const
  {
    Assert (row == col, ExcNotImplemented());
    Assert (diagonal_is_available == true, ExcNotInitialized());
    return diagonal_values(row);
  }



  template <int dim, int fe_degree, int n_q_points_1d, typename number>
  void
  LaplaceOperator<dim,fe_degree,n_q_points_1d,number>::set_diagonal(const Vector<number> &diagonal)
  {
    AssertDimension (m(), diagonal.size());

    diagonal_values = diagonal;

    const std::vector<unsigned int> &
    constrained_dofs = data.get_constrained_dofs();
    for (unsigned int i=0; i<constrained_dofs.size(); ++i)
      diagonal_values(constrained_dofs[i]) = 1.0;

    diagonal_is_available = true;
  }
  */