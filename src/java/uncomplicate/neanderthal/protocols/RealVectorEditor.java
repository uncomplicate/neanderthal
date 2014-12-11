package uncomplicate.neanderthal.protocols;

public interface RealVectorEditor extends RealVector {

    RealVectorEditor setEntry (long i, double val);
    RealVectorEditor update (long i, Object fn);

}
