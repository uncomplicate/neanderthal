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

    void rotg (Vector x);

    void rotmg (Vector p, Vector args);

    void rotm (Vector x, Vector y, Block p);

    void swap (Block x, Block y);

    void copy (Block x, Block y);

    Object dot (Vector x, Vector y);

    Object nrm2 (Vector x);

    Object asum (Vector x);

    void rot (Vector x, Vector y, double c, double s);

    void scal (Object alpha, Block x);

    void axpy (Object alpha, Block x, Block y);

    void mv (Object alpha, Matrix a, Vector x, Object beta, Vector y);

    void mv (Matrix a, Vector x);

    void rank (Object alpha, Vector x, Vector y, Matrix a);

    void mm (Object alpha, Matrix a, Matrix b, Object beta, Matrix c);

    void mm (Object alpha, Matrix a, Matrix b, Object beta, Matrix c, boolean right);

    void mm (Object alpha, Matrix a, Matrix b, boolean right);

}
