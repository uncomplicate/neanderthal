package uncomplicate.neanderthal.protocols;

import clojure.lang.IFn;

public interface RealChangeable extends Changeable {

    RealChangeable set (double val);

    RealChangeable set (long i, long j, double val);

    RealChangeable set (long i, double val);

}
