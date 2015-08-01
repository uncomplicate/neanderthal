package uncomplicate.neanderthal.protocols;

public interface Vector {

    long dim ();

    Object boxedEntry (long i);

    Vector subvector (long k, long l);

}
