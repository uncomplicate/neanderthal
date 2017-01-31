//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package uncomplicate.neanderthal.protocols;

public interface BLAS {

    long iamax (Vector x);

    Block swap (Block x, Block y);

    Block copy (Block x, Block y);

    Object dot (Vector x, Vector y);

    Object nrm2 (Vector x);

    Object asum (Vector x);

    Vector rot (Vector x, Vector y, double c, double s);

    Vector rotg (Vector abcs);

    Vector rotm (Vector x, Vector y, Vector params);

    Vector rotmg (Vector d1d2xy, Vector param);

    Block scal (Object alpha, Block x);

    Block axpy (Object alpha, Block x, Block y);

    Vector mv (Object alpha, Matrix a, Vector x, Object beta, Vector y);

    Vector mv (Matrix a, Vector x);

    Matrix rank (Object alpha, Vector x, Vector y, Matrix a);

    Matrix mm (Object alpha, Matrix a, Matrix b, Object beta, Matrix c);

    Matrix mm (Object alpha, Matrix a, Matrix b, boolean right);

}
