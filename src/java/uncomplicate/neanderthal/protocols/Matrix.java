package uncomplicate.neanderthal.protocols;

public interface Matrix {

    long mrows ();

    long ncols ();

    Vector row (long i);

    Vector col (long j);

    Object boxedEntry (long i, long j);

    Matrix transpose ();

    Matrix submatrix (long i, long j, long k, long l);
}
