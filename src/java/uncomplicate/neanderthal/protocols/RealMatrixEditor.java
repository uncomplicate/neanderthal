package uncomplicate.neanderthal.protocols;

public interface RealMatrixEditor extends RealMatrix {

    RealMatrixEditor setEntry (long i, long j, double val);

    RealMatrixEditor update (long i, long j, Object fn);
}
