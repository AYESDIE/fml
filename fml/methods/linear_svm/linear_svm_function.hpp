//
// Created by ayesdie on 25/10/19.
//

#ifndef FML_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_HPP
#define FML_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_HPP

#include "fml/core.hpp"

namespace fml {
namespace svm {

/**
 * The hinge loss function for the linear SVM objective function.
 * This is used by various optimizers to train the linear
 * SVM model.
 */
template <typename DatasetType = xt::xtensor<double, 2>,
          typename LabelsType = xt::xtensor<size_t, 2>>
class LinearSVMFunction
{
public:
  /**
   * Construct the Linear SVM objective function with given parameters.
   *
   * @param dataset Input training data, each column associate with one sample
   * @param labels Labels associated with the feature data.
   * @param numClasses Number of classes for classification.
   * @param lambda L2-regularization constant.
   * @paran delta Margin of difference between correct class and other classes.
   * @param fitIntercept Intercept term flag.
   */
  LinearSVMFunction(const DatasetType& dataset,
                    const LabelsType& labels,
                    const size_t numClasses,
                    const double lambda = 0.0001,
                    const double delta = 1.0,
                    const bool fitIntercept = false);

  /**
   * Evaluate the hinge loss function for all the datapoints
   *
   * @param paramters The parameters of the SVM.
   * @return The value of the loss function for the entire dataset.
   */
  template <typename E>
  double Evaluate(const E& parameters);

  /**
   * Evaluate the gradient of the hinge loss function following the
   * LinearFunctionType requirements on the Gradient function.
   *
   * @tparam GradType Type of the gradient matrix.
   * @param parameters The parameters of the SVM.
   * @param gradient Linear matrix to output the gradient into.
   */
  template <typename E, typename G>
  void Gradient(const E& parameters,
                G& gradient);

private:
  //! The initial point, from which to start the optimization.
  xt::xtensor<double, 2> initialPoint;

  //! Label matrix for provided data
  DatasetType groundTruth;

  //! The datapoints for training.
  DatasetType dataset;

  //! Number of Classes.
  size_t numClasses;

  //! The regularization parameter for L2-regularization.
  double lambda;

  //! The margin between the correct class and all other classes.
  double delta;

  //! Intercept term flag.
  bool fitIntercept;
};

} // namespace svm
} // namespace fml


#endif //FML_METHODS_LINEAR_SVM_LINEAR_SVM_FUNCTION_HPP
