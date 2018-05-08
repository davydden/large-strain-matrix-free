#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

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
                    std::shared_ptr<const MatrixFree<dim,number>> data_reference);

    void set_coefficients(const double mu, const double nu);

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

    std::shared_ptr<const MatrixFree<dim,number>> data_current;
    std::shared_ptr<const MatrixFree<dim,number>> data_reference;

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
    data_current.clear();
    data_reference.clear();
    diagonal_is_available = false;
    diagonal_values.reinit(0);
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
  NeoHookOperator<dim,fe_degree,n_q_points_1d,number>::vmult_add (Vector<double>       &/*dst*/,
                                                    const Vector<double> &/*src*/) const
  {
    /*
    data.cell_loop (&LaplaceOperator::local_apply, this, dst, src);

    const std::vector<unsigned int> &
    constrained_dofs = data.get_constrained_dofs();
    for (unsigned int i=0; i<constrained_dofs.size(); ++i)
      dst(constrained_dofs[i]) += src(constrained_dofs[i]);
    */
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