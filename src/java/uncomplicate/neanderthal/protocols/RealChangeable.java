package uncomplicate.neanderthal.protocols;

import clojure.lang.IFn;

public interface RealChangeable {

    RealChangeable set (double val);

    RealChangeable set (long i, long j, double val);

    RealChangeable set (long i, double val);

    RealChangeable alter (long i, IFn fn);

    RealChangeable alter (long i, long j, IFn fn);
}
