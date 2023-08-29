//
// R package abclass developed by Wenjie Wang <wang@wwenjie.org>
// Copyright (C) 2021-2023 Eli Lilly and Company
//
// This file is part of the R package abclass.
//
// The R package abclass is free software: You can redistribute it and/or
// modify it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or any later
// version (at your option). See the GNU General Public License at
// <https://www.gnu.org/licenses/> for details.
//
// The R package abclass is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//

#ifndef ABCLASS_ABRANK_H
#define ABCLASS_ABRANK_H

#include <vector>
#include <RcppArmadillo.h>
#include "AbclassNet.h"
#include "AbclassGroupLasso.h"
#include "AbclassGroupSCAD.h"
#include "AbclassGroupMCP.h"
#include "Control.h"

namespace abclass
{
    template <typename T_class, typename T_x>
    class Abrank
    {
    protected:
        T_x x_;
        arma::vec query_weight_;
        arma::vec lambda_weight_;

        // generate pairwise data
        arma::mat gen_pairs(const arma::mat& x,
                            const arma::vec& y,
                            std::vector<unsigned int>& ivec,
                            std::vector<unsigned int>& jvec) const
        {
            size_t ii {0};
            std::vector<double> w;
            size_t npairs { y.n_elem * (y.n_elem + 1) / 2 };
            w.reserve(npairs);
            ivec.reserve(npairs);
            jvec.reserve(npairs);
            for (size_t i {0}; i < y.n_elem; ++i) {
                for (size_t j {0}; j < y.n_elem; ++j) {
                    if (i == j) {
                        continue;
                    }
                    double tmp { y[i] - y[j] };
                    if (tmp > 0.0) {
                        w.push_back(tmp);
                        ivec.push_back(i);
                        jvec.push_back(j);
                        ++ii;
                    }
                }
            }
            arma::mat out_x { arma::zeros(ii, x.n_cols) };
            for (size_t i {0}; i < ii; ++i) {
                out_x.row(i) = x.row(ivec[i]) - x.row(jvec[i]);
            }
            return out_x;
        }


    public:

        // constructors
        Abrank(const std::vector<T_x>& xs,
               const std::vector<arma::vec>& ys,
               const Control& control = Control())
        {


        }



    };

}  // abclass


#endif /* ABCLASS_ABRANK_H */
