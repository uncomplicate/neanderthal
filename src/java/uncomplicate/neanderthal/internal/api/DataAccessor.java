//   Copyright (c) Dragan Djuric. All rights reserved.
//   The use and distribution terms for this software are covered by the
//   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
//   which can be found in the file LICENSE at the root of this distribution.
//   By using this software in any fashion, you are agreeing to be bound by
//   the terms of this license.
//   You must not remove this notice, or any other, from this software.

package uncomplicate.neanderthal.internal.api;

import java.nio.ByteBuffer;

public interface DataAccessor {

    Object entryType ();

    long entryWidth ();

    long count (Object data);

    Object createDataSource (long n);

    Object initialize (Object data);

    Object initialize (Object data, Object value);

    Object wrapPrim (double data);

    Object castPrim (double data);

}
