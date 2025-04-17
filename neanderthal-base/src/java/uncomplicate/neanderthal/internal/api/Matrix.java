//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package uncomplicate.neanderthal.internal.api;

public interface Matrix extends VectorSpace {

    long mrows ();

    long ncols ();

    VectorSpace row (long i);

    Object rows ();

    VectorSpace col (long j);

    Object cols ();

    VectorSpace dia ();

    VectorSpace dia (long k);

    Object dias ();

    Object boxedEntry (long i, long j);

    Matrix transpose ();

    Matrix submatrix (long i, long j, long k, long l);

}
