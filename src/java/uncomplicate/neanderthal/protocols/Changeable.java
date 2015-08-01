package uncomplicate.neanderthal.protocols;

import clojure.lang.IFn;

public interface Changeable {

    Changeable setBoxed (Number val);

    Changeable setBoxed (long i, long j, Number val);

    Changeable setBoxed (long i, Number val);

    Changeable alter (long i, IFn fn);

    Changeable alter (long i, long j, IFn fn);
}
