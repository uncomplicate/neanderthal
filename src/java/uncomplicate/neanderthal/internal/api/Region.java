//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package uncomplicate.neanderthal.internal.api;

public interface Region {

    boolean accessible (long i, long j);

    long colStart (long i);

    long colEnd (long i);

    long rowStart (long i);

    long rowEnd (long i);

    boolean isUpper ();

    boolean isLower ();

    boolean isDiagUnit ();

    int uplo ();

    int diag ();

    long surface ();

    long kl ();

    long ku ();

}
